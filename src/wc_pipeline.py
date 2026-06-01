"""
International football match-outcome prediction pipeline.

Design choices (read these before changing anything):
  * Elo is computed over the FULL match history so ratings converge and reflect
    each team as it is *now* (Elo washes out old results automatically).
  * The supervised model is trained only on a RECENT window (default 10 years)
    plus exponential time-decay weights, because the feature->outcome
    relationship is best learned from the current era.
  * Validation is strictly time-based (train on the past, test on the future).
    Never shuffle a time series -- a random split leaks the future into training
    and gives a fake-good score.
  * Target is the 3-way outcome from the home team's perspective:
    0 = home win, 1 = draw, 2 = away win.

Data: martj42 "International football results from 1872 to 2026" (Kaggle).
Download (needs a Kaggle account + API token in ~/.kaggle/kaggle.json):
    pip install kaggle
    kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 --unzip
Then point DATA_PATH at results.csv.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import wc_squads as sq   # confederation() tournament->confed mapping (no cycle)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/results.csv"      # martj42 results.csv
DATA_START = "2000-01-01"      # use only matches on/after this date for EVERYTHING
#                                (Elo, form, H2H and modelling). None = full history.
TRAIN_YEARS = 10               # supervised model trains on the last N years
DECAY_HALFLIFE_DAYS = 365 * 2  # sample weight halves every ~3 years
FORM_WINDOW = 10               # matches in the rolling-form features
TEST_FRACTION = 0.2            # most-recent slice of the window held out for test
HOME_ADV_ELO = 0          # Elo points added to a non-neutral home side
ELO_START = 1500.0

# K-factor by competition tier (World Football Elo style, keyword-matched).
TOURNAMENT_K = {
    "world cup": 75, "olympic": 30, "confederations": 50,
    "uefa euro": 50, "copa am": 50, "african cup": 50, "afc asian": 50,
    "gold cup": 50, "uefa nations": 40, "qualification": 30, "qualifier": 30,
    "friendly": 10,
}
DEFAULT_K = 30


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    needed = {"date", "home_team", "away_team", "home_score", "away_score",
              "tournament", "neutral"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {missing}")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(bool)
    return df


def apply_name_aliases(df: pd.DataFrame, former_names_path: str):
    """Fold historical team names onto their current name BEFORE Elo/form, so a
    renamed team's rating history stays continuous (e.g. Dahomey -> Benin).

    The martj42 results.csv already stores every match under the CURRENT name
    (USSR games appear as 'Russia', etc.), so on that file this is a no-op and
    returns 0. It exists to keep the pipeline correct if pointed at a raw/older
    dump that still carries the old names.

    Mapping is DATE-BOUNDED to each rename's [start,end] window, matching the
    dataset's own semantics. Genuine break-ups that are NOT simple renames
    (Yugoslavia, Czechoslovakia, German DR) are deliberately absent from
    former_names.csv and are left as separate teams -- merging them would be
    wrong (their successors split rating history).

    Returns (df, n_remapped).
    """
    if not former_names_path or not os.path.exists(former_names_path):
        return df, 0
    fn = pd.read_csv(former_names_path, parse_dates=["start_date", "end_date"])
    df = df.copy()
    n_remapped = 0
    for r in fn.itertuples(index=False):
        in_window = (df["date"] >= r.start_date) & (df["date"] <= r.end_date)
        for col in ("home_team", "away_team"):
            hit = in_window & (df[col] == r.former)
            n_remapped += int(hit.sum())
            df.loc[hit, col] = r.current
    return df, n_remapped


# ---------------------------------------------------------------------------
# 2. Elo over the full history (no leakage: store the PRE-match rating)
# ---------------------------------------------------------------------------
def _k_factor(tournament: str) -> float:
    t = str(tournament).lower()
    for key, k in TOURNAMENT_K.items():
        if key in t:
            return k
    return DEFAULT_K


def _gd_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    elo_home = np.empty(len(df))
    elo_away = np.empty(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        rh = ratings.get(row.home_team, ELO_START)
        ra = ratings.get(row.away_team, ELO_START)
        # pre-match snapshot -> safe to use as a feature
        elo_home[i] = rh
        elo_away[i] = ra

        adv = 0.0 if row.neutral else HOME_ADV_ELO
        exp_home = 1.0 / (1.0 + 10 ** (-(rh + adv - ra) / 400.0))

        gd = row.home_score - row.away_score
        if gd > 0:
            score_home = 1.0
        elif gd == 0:
            score_home = 0.5
        else:
            score_home = 0.0

        k = _k_factor(row.tournament) * _gd_multiplier(gd)
        delta = k * (score_home - exp_home)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

    df = df.copy()
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = elo_home + \
        np.where(df["neutral"], 0.0, HOME_ADV_ELO) - elo_away
    return df, ratings


# ---------------------------------------------------------------------------
# 3. Rolling form (computed only from each team's PRIOR matches)
# ---------------------------------------------------------------------------
def add_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # Build a long table: one row per (team, match) so we can roll per team.
    home = pd.DataFrame({
        "midx": df.index, "date": df["date"], "team": df["home_team"],
        "gf": df["home_score"], "ga": df["away_score"], "side": "home",
    })
    away = pd.DataFrame({
        "midx": df.index, "date": df["date"], "team": df["away_team"],
        "gf": df["away_score"], "ga": df["home_score"], "side": "away",
    })
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select(
        [long.gf > long.ga, long.gf == long.ga], [3, 1], default=0)
    long["gd"] = long["gf"] - long["ga"]
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    g = long.groupby("team", group_keys=False)
    # shift(1) so the current match is never included in its own form. Roll each
    # single pre-computed column -> no apply over the grouping key. gf/ga split
    # the old goal-diff into OFFENSE (goals scored) and DEFENSE (goals conceded)
    # so the model can value attack and defense separately (and a GBM can later
    # interact home-attack x away-defense).
    def roll(s): return s.shift(1).rolling(window, min_periods=1).mean()
    long["form_pts"] = g["pts"].apply(roll)
    long["form_gf"] = g["gf"].apply(roll)   # offense: goals scored
    long["form_ga"] = g["ga"].apply(roll)   # defense: goals conceded

    cols = ["form_pts", "form_gf", "form_ga"]
    h = long[long.side == "home"].set_index("midx")[cols]
    a = long[long.side == "away"].set_index("midx")[cols]
    df = df.copy()
    df["form_pts_home"] = h["form_pts"].reindex(df.index).fillna(1.0)
    df["form_pts_away"] = a["form_pts"].reindex(df.index).fillna(1.0)
    df["form_gf_home"] = h["form_gf"].reindex(df.index).fillna(1.0)
    df["form_gf_away"] = a["form_gf"].reindex(df.index).fillna(1.0)
    df["form_ga_home"] = h["form_ga"].reindex(df.index).fillna(1.0)
    df["form_ga_away"] = a["form_ga"].reindex(df.index).fillna(1.0)
    return df


# ---------------------------------------------------------------------------
# 3b. Head-to-head: each team's record vs THIS specific opponent (causal)
# ---------------------------------------------------------------------------
def add_head_to_head(df: pd.DataFrame, shrink: float = 5.0) -> pd.DataFrame:
    """Per-match head-to-head features from the two teams' PRIOR meetings only
    (any competition, full history -> maximal sample). Captures a pair-specific
    edge BEYOND general strength (which Elo already has).

    Shrunk toward neutral by `shrink` pseudo-meetings so a 1-2 game sample does
    not produce an extreme signal:
      h2h_home_winrate : shrunk win rate of the CURRENT home team in the matchup
                         (toward 0.5)
      h2h_home_gd      : shrunk avg goal diff, current-home perspective (toward 0)
      h2h_logn         : log1p(prior meetings) -> a confidence/where-applicable cue
    All strictly causal: each row reads the accumulator BEFORE adding itself.
    """
    n = len(df)
    hw = np.empty(n)
    hg = np.empty(n)
    hn = np.empty(n)
    # (lo,hi) -> [meetings, lo_wins, draws, gd_sum_lo]
    stats: dict[tuple, list] = {}
    for i, (h, a, hs, as_) in enumerate(zip(
            df["home_team"].to_numpy(), df["away_team"].to_numpy(),
            df["home_score"].to_numpy(), df["away_score"].to_numpy())):
        lo, hi = (h, a) if h <= a else (a, h)
        m, lo_wins, draws, gd_lo = stats.get((lo, hi), (0, 0, 0, 0.0))
        if m > 0:
            home_wins = lo_wins if h == lo else (m - lo_wins - draws)
            home_gd = gd_lo if h == lo else -gd_lo
            hw[i] = (home_wins + 0.5 * shrink) / (m + shrink)
            hg[i] = home_gd / (m + shrink)
        else:
            hw[i] = 0.5
            hg[i] = 0.0
        hn[i] = np.log1p(m)
        # fold this match into the accumulator (from the 'lo' team's perspective)
        gd = hs - as_
        lo_gd = gd if h == lo else -gd
        lo_win = 1 if lo_gd > 0 else 0
        draw = 1 if gd == 0 else 0
        stats[(lo, hi)] = [m + 1, lo_wins +
                           lo_win, draws + draw, gd_lo + lo_gd]

    df = df.copy()
    df["h2h_home_winrate"] = hw
    df["h2h_home_gd"] = hg
    df["h2h_logn"] = hn
    return df


# ---------------------------------------------------------------------------
# 3c. Context features: confederation matchup, host/continental edge, importance
# ---------------------------------------------------------------------------
CONFEDS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]


def build_team_confederation(df: pd.DataFrame) -> dict:
    """team -> confederation, by the MODAL continental competition the team plays
    in (Euro/Copa/AFCON/Gold Cup/Asian Cup/... and their qualifiers). Static, no
    leakage. Teams seen only in World Cup/friendlies stay unknown -> all-zero
    one-hots (handled downstream)."""
    conf = df["tournament"].map(sq.confederation)
    cont = conf.isin(CONFEDS)
    rows = pd.concat([
        pd.DataFrame({"team": df["home_team"][cont], "conf": conf[cont]}),
        pd.DataFrame({"team": df["away_team"][cont], "conf": conf[cont]}),
    ])
    return (rows.groupby("team")["conf"]
            .agg(lambda s: s.value_counts().idxmax()).to_dict())


# Step 3 evaluation verdict: these confederation / host-continent / competition-
# importance features showed NO measurable benefit in the walk-forward -- not in
# aggregate and not even on the inter-confederation subset they were meant to help
# (all deltas within noise). Kept here, but OFF by default. Flip USE_CONTEXT_FEATURES
# to re-enable (build_dataset will then compute them).
USE_CONTEXT_FEATURES = False
CONTEXT_FEATURES = (
    ["comp_importance", "same_confed", "home_own_continent", "away_own_continent"]
    + [f"home_conf_{c}" for c in CONFEDS]
    + [f"away_conf_{c}" for c in CONFEDS]
)


def add_context_features(df: pd.DataFrame, team_conf: dict) -> pd.DataFrame:
    """Confederation identity (one-hot per side), a same-confederation flag, a
    continental-host edge (team playing inside its own confederation's region,
    from the venue `country`), and a competition-importance scalar (reuses the
    Elo K-factor). All known pre-match and for 2026 (venue = USA/CAN/MEX)."""
    df = df.copy()
    hc = df["home_team"].map(team_conf)
    ac = df["away_team"].map(team_conf)
    vc = (df["country"].map(team_conf) if "country" in df.columns
          else pd.Series(index=df.index, dtype=object))   # venue confederation

    df["comp_importance"] = (df["tournament"].map(_k_factor).astype(float)
                             / float(DEFAULT_K))           # ~2 WC, ~1.3 qual, ~0.7 friendly
    df["same_confed"] = ((hc == ac) & hc.notna()).astype(int)
    df["home_own_continent"] = ((hc == vc) & hc.notna()).astype(int)
    df["away_own_continent"] = ((ac == vc) & ac.notna()).astype(int)
    for c in CONFEDS:
        df[f"home_conf_{c}"] = (hc == c).astype(int)
        df[f"away_conf_{c}"] = (ac == c).astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. Target + assembly
# ---------------------------------------------------------------------------
FEATURES = [
    "elo_diff", "is_neutral",
    "form_pts_home", "form_pts_away",
    "form_gf_home", "form_gf_away",      # offense (goals scored)
    "form_ga_home", "form_ga_away",      # defense (goals conceded)
    "h2h_home_winrate", "h2h_home_gd", "h2h_logn",   # team-vs-team history
] + (CONTEXT_FEATURES if USE_CONTEXT_FEATURES else [])


def build_features(df: pd.DataFrame, former_names_path: str | None = None,
                   verbose: bool = True):
    """Run the full causal feature pipeline on an already-loaded results frame:
    date cutoff -> name aliases -> Elo -> form -> H2H -> (context) -> target.

    Split out from build_dataset so the SAME code can score a hypothetical
    fixture: append the fixture as a row and call this (see predict.MatchPredictor).
    Because every feature is a pre-match snapshot, a trailing fixture row gets
    correct features and its dummy result never affects earlier rows.
    Returns (df_with_features, final_ratings).
    """
    df = df.copy()
    # Restrict the WHOLE pipeline (Elo/form/H2H + modelling) to recent history.
    # Elo then cold-starts everyone at ELO_START on DATA_START and re-converges
    # within ~2 seasons -- well before the 8y test region, so no harm there.
    if DATA_START is not None:
        df = df[df["date"] >= pd.Timestamp(DATA_START)]
    df = df.reset_index(drop=True)              # clean RangeIndex for add_form
    # Normalize historical -> current team names BEFORE Elo/form so rating
    # history is continuous across renames (no-op on the martj42 file).
    df, n_alias = apply_name_aliases(df, former_names_path)
    if n_alias and verbose:
        print(f"Name aliases: remapped {n_alias:,} historical team entries "
              f"to current names.")
    df, ratings = add_elo(df)
    df = add_form(df, FORM_WINDOW)
    df = add_head_to_head(df)
    if USE_CONTEXT_FEATURES:
        df = add_context_features(df, build_team_confederation(df))
    df["is_neutral"] = df["neutral"].astype(int)
    gd = df["home_score"] - df["away_score"]
    df["target"] = np.select([gd > 0, gd == 0], [0, 1],
                             default=2)  # 0 H, 1 D, 2 A
    return df, ratings


def build_dataset(path: str):
    fn_path = os.path.join(os.path.dirname(path), "former_names.csv")
    return build_features(load_results(path), fn_path)


# ---------------------------------------------------------------------------
# 5. Train / evaluate with a time-based split + decay weights
# ---------------------------------------------------------------------------
def time_decay_weights(dates: pd.Series, halflife_days: float) -> np.ndarray:
    age = (dates.max() - dates).dt.days.to_numpy()
    return 0.5 ** (age / halflife_days)


def evaluate(path: str = DATA_PATH):
    df, ratings = build_dataset(path)

    cutoff = df["date"].max() - pd.DateOffset(years=TRAIN_YEARS)
    recent = df[df["date"] >= cutoff].reset_index(drop=True)
    print(f"Full history: {len(df):,} matches. "
          f"Recent {TRAIN_YEARS}y window used for modelling: {len(recent):,}.")

    split = int(len(recent) * (1 - TEST_FRACTION))
    train, test = recent.iloc[:split], recent.iloc[split:]
    print(f"Train: {len(train):,}  Test (most recent): {len(test):,}")

    Xtr, ytr = train[FEATURES].to_numpy(), train["target"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test["target"].to_numpy()
    w = time_decay_weights(train["date"], DECAY_HALFLIFE_DAYS)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),  # multinomial by default
    )
    model.fit(Xtr, ytr, logisticregression__sample_weight=w)

    proba = model.predict_proba(Xte)
    pred = proba.argmax(axis=1)
    labels = [0, 1, 2]

    # Baselines to beat.
    # (a) always predict the base rates seen in training
    base_rate = np.bincount(ytr, minlength=3) / len(ytr)
    base_proba = np.tile(base_rate, (len(yte), 1))
    # (b) Elo-only: map elo_diff to a P(home win) and split the rest as draw/away
    elo_p_home = 1 / (1 + 10 ** (-test["elo_diff"].to_numpy() / 400))
    elo_proba = np.column_stack([
        elo_p_home * 0.8,                    # crude: most non-draw mass to favourite
        np.full(len(yte), 0.0),
        (1 - elo_p_home) * 0.8,
    ])
    elo_proba[:, 1] = 1 - elo_proba[:, 0] - elo_proba[:, 2]
    elo_proba = np.clip(elo_proba, 1e-6, 1)
    elo_proba /= elo_proba.sum(axis=1, keepdims=True)

    print("\n--- Test metrics (lower log loss / Brier = better) ---")
    print(f"Model      log loss: {log_loss(yte, proba, labels=labels):.4f}   "
          f"acc: {accuracy_score(yte, pred):.3f}")
    print(
        f"Base-rate  log loss: {log_loss(yte, base_proba, labels=labels):.4f}")
    print(
        f"Elo-only   log loss: {log_loss(yte, elo_proba, labels=labels):.4f}")

    print("\nLearned coefficients (multinomial, standardized features):")
    lr = model.named_steps["logisticregression"]
    for cls, name in zip(range(3), ["P(home)", "P(draw)", "P(away)"]):
        terms = ", ".join(f"{f}={c:+.2f}" for f,
                          c in zip(FEATURES, lr.coef_[cls]))
        print(f"  {name}: {terms}")

    return model, df, ratings


# ---------------------------------------------------------------------------
# 6. Predict an arbitrary fixture from current ratings + latest form
# ---------------------------------------------------------------------------
def predict_match(model, df_full, ratings, home, away, neutral=True):
    """Predict a fixture using each team's current (converged) Elo and latest form."""
    eh = ratings.get(home, ELO_START)
    ea = ratings.get(away, ELO_START)
    elo_diff = eh + (0.0 if neutral else HOME_ADV_ELO) - ea

    def latest_form(team):
        rows = df_full[(df_full.home_team == team) |
                       (df_full.away_team == team)]
        if rows.empty:
            return 1.0, 0.0
        last = rows.iloc[-1]
        if last.home_team == team:
            return last.form_pts_home, last.form_gd_home
        return last.form_pts_away, last.form_gd_away

    fph, fgh = latest_form(home)
    fpa, fga = latest_form(away)
    x = np.array([[elo_diff, int(neutral), fph, fpa, fgh, fga]])
    p = model.predict_proba(x)[0]
    print(f"\n{home} vs {away} ({'neutral' if neutral else home + ' at home'})")
    print(f"  P({home} win) = {p[0]:.1%}   P(draw) = {p[1]:.1%}   "
          f"P({away} win) = {p[2]:.1%}")
    return p


if __name__ == "__main__":
    model, df, ratings = evaluate(DATA_PATH)
    # Example once you have real data:
    # predict_match(model, df, ratings, "Argentina", "France", neutral=True)
