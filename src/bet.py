"""
bet.py -- read a fixtures+odds CSV and print the recommended betting portfolio.

CSV columns:
  home, away                     (required)
  neutral                        (optional, default 1)   1/0 or true/false
  wc                             (optional, default 0)   1 => apply WC goal-scale
  <market odds columns>          decimal odds; leave blank where you have no price:
    home_win, draw, away_win,
    over_2.5, under_2.5,
    goals_0_1, goals_2_3, goals_4plus,
    hcap_home_-1, hcap_away_+1, hcap_home_+1, hcap_away_-1

Usage (from src/):
  python bet.py fixtures.csv --bankroll 1000 --kelly 0.25 --edge 0.03
  python bet.py fixtures.csv --mv country_market_value.csv     # use market value
"""
from __future__ import annotations
import argparse
import os
import pandas as pd

MARKET_COLS = ["home_win", "draw", "away_win",
               "over_2.5", "under_2.5",
               "goals_0_1", "goals_2_3", "goals_4plus",
               "hcap_home_-1", "hcap_away_+1", "hcap_home_+1", "hcap_away_-1"]


def _bool(x, default):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="fixtures+odds CSV")
    ap.add_argument("--results", default="../data/results.csv",
                    help="path to results.csv (default ../data/results.csv)")
    ap.add_argument("--mv", default=None,
                    help="country market-value CSV (enables the MV feature)")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    ap.add_argument("--edge", type=float, default=0.03, help="min edge to bet")
    ap.add_argument("--max-per-bet", type=float, default=0.05)
    ap.add_argument("--max-exposure", type=float, default=0.25)
    ap.add_argument("--devig", default="shin", choices=["shin", "multiplicative"])
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"fixtures file not found: {args.csv}")
    df = pd.read_csv(args.csv)
    for req in ("home", "away"):
        if req not in df.columns:
            raise SystemExit(f"CSV must have a '{req}' column")
    present = [c for c in MARKET_COLS if c in df.columns]
    if not present:
        raise SystemExit(f"CSV has no market-odds columns. Expected any of: {MARKET_COLS}")

    matches = []
    for _, r in df.iterrows():
        odds = {c: float(r[c]) for c in present
                if pd.notna(r[c]) and float(r[c]) > 1}
        if not odds:
            continue
        matches.append({"home": str(r["home"]).strip(),
                        "away": str(r["away"]).strip(),
                        "neutral": _bool(r.get("neutral"), True),
                        "wc": _bool(r.get("wc"), False),
                        "odds": odds})
    if not matches:
        raise SystemExit("No fixtures with usable odds (>1) found.")

    print(f"Loaded {len(matches)} fixture(s) from {args.csv}. Training model "
          f"(market value {'ON' if args.mv else 'off'}) ...\n")
    from predict import MatchPredictor
    from portfolio import portfolio_from_predictor
    pr = MatchPredictor(args.results, country_series=args.mv)

    for m in matches:
        for t in (m["home"], m["away"]):
            if not pr.known_team(t):
                print(f"  warning: unknown team {t!r} -> default rating used")

    print()
    portfolio_from_predictor(pr, matches, bankroll=args.bankroll, kelly=args.kelly,
                             edge_threshold=args.edge, max_per_bet=args.max_per_bet,
                             max_exposure=args.max_exposure, devig_method=args.devig)
    print("\n  (vs_fair = model prob - devigged market prob; >0 = you disagree with "
          "the SHARP market, not just the vig)")


if __name__ == "__main__":
    main()
