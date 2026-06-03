# FIFA Money Printer

Predict international football match outcomes (aimed at the 2026 World Cup) and turn
calibrated probabilities into betting signals across 1X2, totals, and Asian-handicap
markets. The model is a **logistic-regression + Dixon-Coles goals-model blend** with
Elo, form, head-to-head, and Glicko features.

See **[SUMMARY.pdf](SUMMARY.pdf)** for the full story (methodology, results, and every
idea that was tested — both what worked and what didn't).

## How the model works (best configuration)

Validated SOTA: **0.839 out-of-sample log loss** (3-way), **beats the Elo baseline in 16 of
17 walk-forward folds**, and well-calibrated (predicted draw rate 0.221 vs 0.220 actual).

**Data.** International matches since 2000. Elo / form / H2H / Glicko are computed over the
full history; only friendlies are dropped from the model's training rows (kept in the ratings).

**Features (12, all causal pre-match snapshots):**
- `elo_diff` — competition-weighted Elo (the dominant signal)
- `is_neutral`
- rolling form (last 10 games): points, plus an offense/defense split (goals scored / conceded), home & away
- head-to-head vs the *specific* opponent: shrunk win rate, goal difference, log(#meetings)
- `glicko_exp` — an uncertainty-shrunk Glicko win expectancy (discounts the rating gap when a team is rarely rated)

**Two models, blended 50/50:**
1. **Logistic regression** on the 12 features, trained on competitive matches with exponential
   time-decay weights. Linear, calibrated, and — proven by experiment — hard to beat here.
2. **Dixon-Coles goals model**: per-team attack/defense + home edge by time-weighted maximum
   likelihood from goals (fit on *all* history with a ~5-year half-life, since international
   teams play rarely), with the low-score correction. Produces the full scoreline matrix.

The blend wins because the two are **decorrelated** (a feature classifier vs a goals process):
averaging captures both, and the goals model repairs the draw structure the classifier misses.
1X2 log loss went **0.848** (LR alone) → **0.842** (blend) → **0.839** (Dixon-Coles on
expanding history).

**Secondary markets** (over/under, total-goals buckets, 3-way handicap) are derived from the
scoreline matrix **reconciled** to the blend's better 1X2 — so they inherit the blend's
calibration *plus* Dixon-Coles' "win by 1 vs win by 2+" margin detail. `wc=True` applies a
~+8% World-Cup goal-scale (a real, measured effect) that sharpens totals/handicap at WC finals.

**Live prediction** uses a synthetic-row trick: the fixture is appended to the results and the
exact training feature pipeline is re-run, so live and training features can never drift.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get the data (not stored in git)
1. `data/results.csv` — martj42 "International football results" (Kaggle). `download.py`
   fetches it (needs a Kaggle API token in `~/.kaggle/kaggle.json`).
2. Optional, for the market-value feature: the salimt `football-datasets` dump
   (`kaggle datasets download -d xfkzujqjvx97n/football-datasets`) into `data/`.

The whole `data/` folder is git-ignored — regenerate it from the downloads above.

## Run (from `src/`)

```bash
cd src
python walk_forward.py        # walk-forward backtest (LR, no market value)
python run_with_mv.py         # backtest WITH market value (--injuries, --model both)
python dixon_coles.py         # Dixon-Coles vs LR vs blend
python predict.py             # demo predictions (LR / DC / blend + secondary markets)
python bet.py fixtures_example.csv   # betting portfolio from a fixtures+odds CSV
```

> Always run from `src/` (modules import each other by name and use `../data/...`).
> Use the venv's Python (`../.venv/bin/python`) — a different numpy build prints harmless
> matmul warnings.

## Predict a fixture (Python)

```python
from predict import MatchPredictor
pr = MatchPredictor("../data/results.csv")          # or (..., "country_market_value.csv") for MV
pr.predict("Brazil", "Argentina", neutral=True)     # {'home_win', 'draw', 'away_win'}
pr.total_buckets("Brazil", "Argentina", wc=True)    # 0-1 / 2-3 / 4+  (wc=True: WC goal-scale)
pr.handicap("Brazil", "Argentina", line=-1)         # Asian handicap +/-1 (with push)
pr.predict("France", "Spain", unavailable={"France": ["Kylian Mbappé"]})   # drop a player
```

## Bet from a CSV — `bet.py`

```bash
python bet.py fixtures.csv --bankroll 1000 --kelly 0.25 --edge 0.03
python bet.py fixtures.csv --mv country_market_value.csv   # use the market-value feature
```

CSV columns (`fixtures_example.csv` is a template):

| column | meaning |
|---|---|
| `home`, `away` | team names (martj42 spelling) |
| `neutral` | 1/0 (default 1) |
| `wc` | 1 => apply World Cup goal-scale (default 0) |
| odds columns | decimal odds; blank where you have none |

Odds columns: `home_win`, `draw`, `away_win`, `over_2.5`, `under_2.5`,
`goals_0_1`, `goals_2_3`, `goals_4plus`, `hcap_home_-1`, `hcap_away_+1`,
`hcap_home_+1`, `hcap_away_-1`.

Output shows, per bet: your model probability, the **devigged** fair market probability,
`vs_fair` (model − fair; >0 means you disagree with the *sharp* market, not just the vig),
the edge, and a fractional-Kelly stake.

## Modules (`src/`)

| module | role |
|---|---|
| `wc_pipeline.py` | features (Elo, form, H2H, Glicko) + `build_features` / `build_dataset` |
| `dixon_coles.py` | Dixon-Coles goals model → scoreline matrix → 1X2 + secondary markets |
| `walk_forward.py` | walk-forward backtest, model registry, IS/OOS gap |
| `wc_squads.py` / `wc_market_value.py` / `wc_squad_dataset.py` | competitive filter + market-value feature |
| `predict.py` | `MatchPredictor` (blend, secondary markets, absence override) |
| `portfolio.py` | devig + push-aware fractional-Kelly staking |
| `bet.py` | CSV → portfolio CLI |
| `run_with_mv.py` | one-command runner |
| `chemistry.py` | causal chemistry feature (tested null; reference tool) |

## Honest notes
- The model is **at its information ceiling** (~0.84 log loss, beats Elo 16/17). Many
  "beyond strength" feature ideas were tested and ruled out — see SUMMARY.pdf.
- **Betting reality**: you supply the odds; the system finds where your model disagrees
  with the market. Without **historical odds** we can't yet measure realized profit (CLV /
  betting backtest) — that's the one open dependency.
- Stakes default to conservative (¼-Kelly, 5% max/bet, 25% max exposure). Don't crank to
  full Kelly — calibrated ≠ infallible.
