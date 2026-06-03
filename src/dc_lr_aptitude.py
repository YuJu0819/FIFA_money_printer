"""
Where does each 1X2 model earn its keep? -- DC vs LR aptitude analysis.

Reuses the EXACT walk-forward folds of dixon_coles.backtest (DC = expanding
window on all matches, LR = rolling competitive-only), collecting per-match
out-of-fold probabilities for both models AND match metadata, then slices the
head-to-head per-match log loss by:
    * favourite strength      -> |elo_diff| buckets (mismatch vs coin-flip)
    * who is favoured / venue -> elo_diff sign x neutral
    * actual outcome class    -> home / draw / away
    * match type              -> World Cup / continental / qualifier / other
    * rating uncertainty      -> glicko win-expectancy extremity

Lower log loss = better. The "edge" column is LR - DC, so +ve = DC wins the bucket.

    python3 dc_lr_aptitude.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import wc_pipeline as wc
import wc_squads as sq
import walk_forward as wf
from dixon_coles import DixonColes

EPS = 1e-15
LABELS = [0, 1, 2]
CLASS_NAME = {0: "home", 1: "draw", 2: "away"}


def collect(path: str) -> pd.DataFrame:
    """One row per out-of-fold competitive match with both models' probs + meta."""
    df, _ = wc.build_dataset(path)                      # all matches (DC fitting)
    comp = sq.competitive_only(df).reset_index(drop=True)   # test population

    rows = []
    for tr, te, b0, b1 in wf.generate_folds(comp):
        # DC: expanding window on ALL matches before the block
        hist = df[df["date"] < b0]
        dc = DixonColes().fit(hist, ref_date=b0)
        P_dc = np.array([[*dc.predict(h, a, n).values()]
                         for h, a, n in zip(te["home_team"], te["away_team"],
                                            te["neutral"])])
        # LR: same fold's rolling competitive train block
        w = wc.time_decay_weights(tr["date"], wc.DECAY_HALFLIFE_DAYS)
        est = wf._fit(wf.make_lr(), tr[wc.FEATURES].to_numpy(),
                      tr["target"].to_numpy(), w)
        P_lr = wf.predict_proba_full(est, te[wc.FEATURES].to_numpy())

        y = te["target"].to_numpy()
        ll_dc = -np.log(np.clip(P_dc[np.arange(len(y)), y], EPS, 1))
        ll_lr = -np.log(np.clip(P_lr[np.arange(len(y)), y], EPS, 1))

        block = pd.DataFrame({
            "date": te["date"].to_numpy(),
            "tournament": te["tournament"].to_numpy(),
            "neutral": te["neutral"].to_numpy().astype(bool),
            "elo_diff": te["elo_diff"].to_numpy(),
            "elo_home": te["elo_home"].to_numpy(),
            "elo_away": te["elo_away"].to_numpy(),
            "glicko_exp": te["glicko_exp"].to_numpy(),
            "target": y,
            "ll_dc": ll_dc,
            "ll_lr": ll_lr,
        })
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def match_type(t: str) -> str:
    t = str(t).lower()
    if "world cup" in t and "qualif" not in t:
        return "World Cup finals"
    if "qualif" in t:
        return "Qualifier"
    if any(k in t for k in ("uefa", "copa", "african", "asian", "gold cup",
                            "nations league", "euro", "confederations")):
        return "Continental"
    return "Other competitive"


def _summary(g: pd.DataFrame) -> pd.Series:
    dc, lr = g["ll_dc"].mean(), g["ll_lr"].mean()
    return pd.Series({"n": len(g), "DC_ll": dc, "LR_ll": lr,
                      "edge(LR-DC)": lr - dc,
                      "winner": "DC" if dc < lr else "LR"})


def show(name: str, table: pd.DataFrame):
    print(f"\n=== {name} ===")
    print(table.to_string(float_format=lambda x: f"{x:+.4f}"))


def wc_focus(d: pd.DataFrame):
    """World-Cup-like view: NEUTRAL site only, sliced by the ABSOLUTE Elo level of
    the two teams (overall match strength tier) rather than just the gap."""
    n = d[d["neutral"]].copy()
    print(f"\n\n######## WORLD-CUP VIEW (neutral-site only): {len(n):,} matches ########")
    print(f"Pooled  DC {n['ll_dc'].mean():.4f}   LR {n['ll_lr'].mean():.4f}   "
          f"(edge {n['ll_lr'].mean()-n['ll_dc'].mean():+.4f})")

    n["elo_max"] = n[["elo_home", "elo_away"]].max(axis=1)
    n["elo_min"] = n[["elo_home", "elo_away"]].min(axis=1)
    n["elo_mean"] = n[["elo_home", "elo_away"]].mean(axis=1)
    n["abs_elo"] = n["elo_diff"].abs()

    # 1. Overall match tier = mean Elo of the two teams (both strong vs both weak)
    n["tier"] = pd.qcut(n["elo_mean"], 4,
                        labels=["both weak", "mid-low", "mid-high", "both strong"])
    show("WC | Match strength tier (mean Elo of the two teams)",
         n.groupby("tier", observed=True).apply(_summary))

    # 2. Strong-vs-weak grid: stronger team's level x weaker team's level
    n["strong_lvl"] = pd.qcut(n["elo_max"], 2, labels=["strong:lo", "strong:hi"])
    n["weak_lvl"] = pd.qcut(n["elo_min"], 2, labels=["weak:lo", "weak:hi"])
    grid = (n.groupby(["strong_lvl", "weak_lvl"], observed=True)
            .apply(_summary).reset_index())
    show("WC | Stronger team level x weaker team level", grid.set_index(
        ["strong_lvl", "weak_lvl"]))

    # 3. Absolute level x gap -- does the gap effect depend on the tier?
    n["gap"] = pd.cut(n["abs_elo"], [-1, 100, 250, 1e9],
                      labels=["close", "medium", "mismatch"])
    n["lvl2"] = pd.qcut(n["elo_mean"], 2, labels=["lower half", "upper half"])
    show("WC | Tier x rating gap",
         n.groupby(["lvl2", "gap"], observed=True).apply(_summary))

    # 4. Both elite? (both teams above a high Elo bar -- the latter-stage matchups)
    bar = n["elo_mean"].quantile(0.75)
    elite = n[(n["elo_min"] >= bar)]
    if len(elite) > 30:
        show(f"WC | Both teams elite (each Elo >= {bar:.0f}, top-quartile)",
             elite.assign(g="both elite").groupby("g").apply(_summary))


def main(path: str):
    d = collect(path)
    print(f"Out-of-fold competitive matches: {len(d):,}  "
          f"({d['date'].min().date()} -> {d['date'].max().date()})")
    print(f"Pooled log loss   DC {d['ll_dc'].mean():.4f}   "
          f"LR {d['ll_lr'].mean():.4f}   (edge {d['ll_lr'].mean()-d['ll_dc'].mean():+.4f})")

    # 1. Favourite strength: how lopsided the rating gap is
    d["abs_elo"] = d["elo_diff"].abs()
    d["mismatch"] = pd.cut(d["abs_elo"], [-1, 50, 150, 300, 1e9],
                           labels=["even (<50)", "slight (50-150)",
                                   "clear (150-300)", "blowout (>300)"])
    show("Favourite strength (|elo_diff|)",
         d.groupby("mismatch", observed=True).apply(_summary))

    # 2. Venue: neutral site vs a real home team
    show("Venue", d.groupby(d["neutral"].map({True: "neutral", False: "home/away"}))
         .apply(_summary))

    # 3. Actual outcome class -- where DC's draw structure should matter
    show("Actual outcome",
         d.groupby(d["target"].map(CLASS_NAME)).apply(_summary))

    # 4. Match type
    d["mtype"] = d["tournament"].map(match_type)
    show("Match type", d.groupby("mtype").apply(_summary))

    # 5. Rating-implied favouredness extremity (glicko win expectancy)
    d["gx"] = pd.cut(d["glicko_exp"], [-0.01, .35, .5, .65, 1.01],
                     labels=["under .35", ".35-.5", ".5-.65", ">.65"])
    show("Glicko win-expectancy (home)",
         d.groupby("gx", observed=True).apply(_summary))

    wc_focus(d)
    return d


if __name__ == "__main__":
    p = "../data/results.csv" if os.path.exists("../data/results.csv") else wc.DATA_PATH
    main(p)
