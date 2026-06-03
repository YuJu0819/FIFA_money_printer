"""
Betting layer: turn model probabilities + bookmaker odds into staking decisions.

Per outcome with model prob p and DECIMAL odds o:
    edge (EV per 1 staked) = p * o - 1
    full-Kelly fraction     = (p*o - 1) / (o - 1)     [0 if negative]
We bet only when edge >= EDGE_THRESHOLD (a margin to absorb model error), stake a
FRACTION of Kelly (full Kelly assumes perfectly-calibrated probs and is ruinous
under error), cap any single bet, and cap total simultaneous exposure.

Honest notes:
  * You supply the odds; the system finds where the MODEL disagrees with the BOOK.
    That disagreement is the signal, not guaranteed profit -- beating the vig is hard.
  * `edge` uses the raw odds you actually get paid. The book's implied probs sum to
    >1 (the overround / margin), shown as `book_p` so you see what you're up against.
  * Kelly trusts the probabilities. Calibrate the model (see calibrate step) before
    sizing real money on these stakes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

OUTCOMES = ["home_win", "draw", "away_win"]


def kelly_fraction(p: float, odds: float) -> float:
    """Full-Kelly stake fraction for a single bet (0 if not +EV)."""
    if odds <= 1:
        return 0.0
    return max((p * odds - 1.0) / (odds - 1.0), 0.0)


def implied_probs(odds: dict) -> dict:
    """Bookmaker implied probabilities (raw 1/odds; sum>1 by the overround)."""
    return {k: (1.0 / odds[k] if odds.get(k, 0) > 0 else np.nan) for k in OUTCOMES}


# -- devig: strip the bookmaker margin -> fair probabilities -------------------
def devig(odds: dict, method: str = "shin") -> dict:
    """Fair (vig-free) probabilities for a set of mutually-exclusive outcome odds.
    'multiplicative' just normalises 1/odds to sum to 1; 'shin' additionally
    corrects the favourite-longshot bias (assumes a little insider money)."""
    keys = [k for k in odds if odds.get(k, 0) > 1]
    inv = np.array([1.0 / odds[k] for k in keys])
    B = inv.sum()
    if len(keys) < 2 or B <= 0:
        return {k: float(v) for k, v in zip(keys, inv / B if B else inv)}
    if method == "shin":
        from scipy.optimize import brentq

        def shin_p(z):
            return (np.sqrt(z * z + 4 * (1 - z) * inv * inv / B) - z) / (2 * (1 - z))
        try:
            z = brentq(lambda z: shin_p(z).sum() - 1.0, 1e-9, 0.5)
            p = shin_p(z)
        except ValueError:
            p = inv / B                              # fall back to multiplicative
    else:
        p = inv / B
    return {k: float(pi) for k, pi in zip(keys, p)}


def _market_groups(odds: dict) -> list:
    """Split a flat odds dict into the mutually-exclusive markets to devig
    separately (1X2, each over/under line, totals buckets, each handicap line)."""
    g: dict = {}
    for k in odds:
        if k in ("home_win", "draw", "away_win"):
            g.setdefault("1x2", []).append(k)
        elif k.startswith(("over_", "under_")):
            g.setdefault("ou_" + k.split("_", 1)[1], []).append(k)
        elif k.startswith("goals_"):
            g.setdefault("totals", []).append(k)
        elif k.startswith("hcap_"):
            g.setdefault("hcap_" + k.split("_")[2].lstrip("+-"), []).append(k)
        else:
            g.setdefault(k, []).append(k)
    return list(g.values())


def fair_book(odds: dict, method: str = "shin") -> dict:
    """Per-outcome fair probability, devigging each market in `odds` separately."""
    fp = {}
    for keys in _market_groups(odds):
        fp.update(devig({k: odds[k] for k in keys}, method))
    return fp


def find_bets(label: str, probs: dict, odds: dict,
              kelly: float = 0.25, edge_threshold: float = 0.03,
              pushes: dict = None, devig_method: str = "shin") -> list:
    """All +EV outcomes for one match (before global capping). Works for ANY
    market in `odds` (1X2, over/under, goal buckets, handicap, ...) as long as
    `probs` has a model probability for that key. `pushes[key]` (e.g. an Asian
    handicap landing exactly on the line) returns the stake, so EV and Kelly use
    edge = p*o - (1 - push); full-Kelly = edge / (o - 1).

    `fair_p` is the bookmaker's vig-free probability (devigged); `vs_fair` =
    p_model - fair_p flags whether you genuinely disagree with the SHARP market
    (>0) or your raw edge is mostly the bookmaker's margin (<0)."""
    pushes = pushes or {}
    fair = fair_book(odds, devig_method)
    bets = []
    for k, o in odds.items():
        p = probs.get(k)
        if p is None or not o or o <= 1:
            continue
        q = float(pushes.get(k, 0.0))
        edge = p * o - (1.0 - q)
        if edge >= edge_threshold:
            kf = max(edge / (o - 1.0), 0.0)
            bets.append({
                "match": label, "bet": k, "p_model": p, "push": q,
                "fair_p": fair.get(k, np.nan), "book_p": 1.0 / o, "odds": o,
                "vs_fair": p - fair.get(k, np.nan), "edge": edge,
                "kelly_full": kf, "stake": kelly * kf,
            })
    return bets


def build_portfolio(matches: list, bankroll: float = 1000.0,
                    kelly: float = 0.5, edge_threshold: float = 0.05,
                    max_per_bet: float = 0.2, max_exposure: float = 0.7,
                    devig_method: str = "shin") -> pd.DataFrame:
    """matches: [{'label', 'probs': {...}, 'odds': {...}}, ...]
    Returns a DataFrame of recommended bets with stake fractions and EUR amounts.

    max_per_bet / max_exposure are fractions of bankroll (single bet / total).
    """
    rows = []
    for m in matches:
        rows += find_bets(m.get("label", f"{m.get('home','?')} v {m.get('away','?')}"),
                          m["probs"], m["odds"], kelly, edge_threshold,
                          m.get("pushes"), devig_method)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No +EV bets cleared the edge threshold "
              f"({edge_threshold:.0%}).")
        return df

    # cap any single bet, then scale down if total exposure exceeds the cap
    df["stake"] = df["stake"].clip(upper=max_per_bet)
    total = df["stake"].sum()
    if total > max_exposure:
        df["stake"] *= max_exposure / total
    df["stake_eur"] = (df["stake"] * bankroll).round(2)
    df["exp_profit_eur"] = (df["edge"] * df["stake"] * bankroll).round(2)
    df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    return df


def show_portfolio(df: pd.DataFrame, bankroll: float = 1000.0) -> None:
    if df is None or df.empty:
        return
    view = df[["match", "bet", "p_model", "fair_p", "vs_fair", "odds",
               "edge", "stake", "stake_eur"]].copy()
    for c in ["p_model", "fair_p", "vs_fair", "edge", "stake"]:
        view[c] = (view[c] * 100).round(1).astype(str) + "%"
    print(view.to_string(index=False))
    print(f"\n  bets: {len(df)}   total staked: "
          f"{df['stake_eur'].sum():.2f} ({df['stake'].sum():.1%} of {bankroll:.0f})"
          f"   expected profit: {df['exp_profit_eur'].sum():.2f}")


def portfolio_from_predictor(predictor, matches: list, **kw) -> pd.DataFrame:
    """matches: [{'home','away','neutral','odds':{...}, 'label'?}, ...]
    Fills in model probabilities via the predictor, then builds the portfolio."""
    enriched = []
    for m in matches:
        # ask the predictor for every market the odds reference (1X2, O/U, goal
        # buckets, handicap); pushes come back for handicap lines.
        probs, pushes = predictor.predict_markets(
            m["home"], m["away"], m.get("neutral", True),
            odds_keys=list(m["odds"].keys()), wc=m.get("wc", False))
        enriched.append({"label": m.get("label", f"{m['home']} v {m['away']}"),
                         "probs": probs, "odds": m["odds"], "pushes": pushes})
    bankroll = kw.get("bankroll", 1000.0)
    df = build_portfolio(enriched, **kw)
    show_portfolio(df, bankroll)
    return df


if __name__ == "__main__":
    import os
    from predict import MatchPredictor
    path = "../data/results.csv" if os.path.exists(
        "../data/results.csv") else None
    pr = MatchPredictor(path)
    # demo fixtures with made-up bookmaker odds (home_win / draw / away_win)
    matches = [
        {"home": "Spain", "away": "England", "neutral": True,
         "odds": {"home_win": 2.10, "draw": 3.30, "away_win": 4.20,
                  # secondary markets: totals buckets + Asian handicap +/-1
                  "goals_0_1": 3.40, "goals_2_3": 1.95, "goals_4plus": 4.50,
                  "hcap_home_-1": 3.60, "hcap_away_+1": 1.28}},
        {"home": "Brazil", "away": "Argentina", "neutral": True,
         "odds": {"home_win": 3.10, "draw": 3.10, "away_win": 2.40,
                  "hcap_away_-1": 4.20, "hcap_home_+1": 1.25}},
    ]
    print("\nPortfolio (1/4 Kelly, demo odds; incl. totals buckets + handicap):\n")
    portfolio_from_predictor(pr, matches, bankroll=1000.0, kelly=0.25,
                             edge_threshold=0.03)
