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


def find_bets(label: str, probs: dict, odds: dict,
              kelly: float = 0.25, edge_threshold: float = 0.03) -> list:
    """All +EV outcomes for one match (before global capping). Works for ANY
    market in `odds` (1X2, over_2.5, under_2.5, ...) as long as `probs` has a
    matching model probability for that key."""
    bets = []
    for k, o in odds.items():
        p = probs.get(k)
        if p is None or not o or o <= 1:
            continue
        edge = p * o - 1.0
        if edge >= edge_threshold:
            bets.append({
                "match": label, "bet": k, "p_model": p, "book_p": 1.0 / o,
                "odds": o, "edge": edge, "kelly_full": kelly_fraction(p, o),
                "stake": kelly * kelly_fraction(p, o),
            })
    return bets


def build_portfolio(matches: list, bankroll: float = 1000.0,
                    kelly: float = 0.5, edge_threshold: float = 0.05,
                    max_per_bet: float = 0.2, max_exposure: float = 0.7
                    ) -> pd.DataFrame:
    """matches: [{'label', 'probs': {...}, 'odds': {...}}, ...]
    Returns a DataFrame of recommended bets with stake fractions and EUR amounts.

    max_per_bet / max_exposure are fractions of bankroll (single bet / total).
    """
    rows = []
    for m in matches:
        rows += find_bets(m.get("label", f"{m.get('home','?')} v {m.get('away','?')}"),
                          m["probs"], m["odds"], kelly, edge_threshold)
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
    view = df[["match", "bet", "p_model", "book_p", "odds",
               "edge", "stake", "stake_eur", "exp_profit_eur"]].copy()
    for c in ["p_model", "book_p", "edge", "stake"]:
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
        # ask the predictor for every market the odds reference (1X2 + O/U lines)
        probs = predictor.predict_markets(m["home"], m["away"],
                                          m.get("neutral", True),
                                          odds_keys=list(m["odds"].keys()))
        enriched.append({"label": m.get("label", f"{m['home']} v {m['away']}"),
                         "probs": probs, "odds": m["odds"]})
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
                  "over_2.5": 2.05, "under_2.5": 1.75}},      # secondary market
        {"home": "France", "away": "Germany", "neutral": True,
         "odds": {"home_win": 1.80, "draw": 3.60, "away_win": 4.50}},
        {"home": "Brazil", "away": "Argentina", "neutral": True,
         "odds": {"home_win": 3.10, "draw": 3.10, "away_win": 2.40}},
    ]
    print("\nPortfolio (1/4 Kelly, demo odds):\n")
    portfolio_from_predictor(pr, matches, bankroll=1000.0)
