# FIFA Money Printer — Project Summary

## Goal
Predict the per-game outcome (home win / draw / away win) of international football
matches — aimed at the 2026 World Cup — and turn calibrated probabilities into
betting signals (1X2 **and** secondary markets) with a staked portfolio.

## Data
- **data/results.csv** — ~49k international matches 1872–2026 (martj42, Kaggle). The backbone.
- **salimt Transfermarkt dump** — player market values, national-team membership, profiles,
  and **injuries** (dated spells). Used for the optional squad-value feature.
- Auxiliary: former_names, goalscorers, shootouts (mostly unused).
- (`data/` is git-ignored; regenerate via download.py + Kaggle.)

## Modules (src/, 9 files)
| Module | Role |
|---|---|
| wc_pipeline.py | Causal features: Elo, form (off/def split), H2H, Glicko; build_features/build_dataset |
| wc_squads.py | competitive_only filter; confederation tagging + MV-coverage filter |
| wc_market_value.py | Leakage-safe as-of join of a country value series onto matches |
| wc_squad_dataset.py | Builds the country squad-value series offline; SquadValuer for live overrides |
| dixon_coles.py | Dixon-Coles goals model -> scoreline matrix -> 1X2 + secondary markets |
| walk_forward.py | Walk-forward backtest, model registry (LR/HGB), compare_models, IS/OOS/gap |
| predict.py | MatchPredictor: LR+DC blend, secondary markets, absence override (synthetic-row) |
| portfolio.py | EV + fractional-Kelly staking across any market (1X2, O/U, ...) |
| run_with_mv.py | One-command end-to-end runner |

## Model features (12, all causal pre-match snapshots)
- `elo_diff` — competition-weighted Elo (the dominant signal)
- `is_neutral`
- Form (last 10): `form_pts`, plus offense/defense split `form_gf` / `form_ga`, home & away
- Head-to-head (pair-specific, shrunk): `h2h_home_winrate`, `h2h_home_gd`, `h2h_logn`
- `glicko_exp` — uncertainty-shrunk win expectancy (Glicko-1 rating + RD)
- (+ `mv_log_diff`, `mv_missing` when market value is enabled — opt-in)

## The model: LR + Dixon-Coles blend
- **Logistic regression** on the 12 features (linear, calibrated, hard to beat here).
- **Dixon-Coles** goals model: per-team attack/defense + home edge by time-weighted MLE
  from goals, with the low-score correction. Gives the full scoreline matrix.
- **Blend** (0.5 each) is the production 1X2 model — LR and a goals process are
  decorrelated, so averaging captures both, and DC repairs the draw structure.

## Validation (walk-forward, post-2000, rolling 8y train -> 6mo test blocks)
Each game scored out-of-sample by the model trained only on its past. Tracks OOS log
loss, an in-sample reference + IS->OOS gap (overfit alarm), Brier, accuracy, folds-beating-Elo.

## Performance — the ceiling-breaking progression
| model | OOS log loss | notes |
|---|---|---|
| Pure Elo baseline | 0.894 | 60.1% accuracy; never predicts a draw |
| LR (original, 6 feat) | 0.8484 | 15/17 folds beat Elo |
| + off/def form + H2H | 0.8449 | the one feature gain |
| + Dixon-Coles blend | 0.8419 | fixes draws; unlocks secondary markets |
| + Glicko-1 `glicko_exp` | 0.8405 | LR alone 0.8466; 16/17 folds beat Elo |
| **+ DC expanding history + ~5y decay** | **0.8393** | DC was data-starved by the rolling window |

Net gain **0.8484 -> 0.8393** (~1.1%). The wins came from richer TARGET (goals model)
and more DATA (Glicko ratings; DC on all history), never from a fancier model class.
The model is **well-calibrated** (draw class 0.221 predicted vs 0.220 actual).

## Why we're at the ceiling — the error budget
40% of matches carry 68% of all log loss, in two near-irreducible buckets:
| match type | % matches | % of total log loss | mean log loss |
|---|---|---|---|
| Draws | 22% | 37% | 1.43 |
| Upsets (favourite lost) | 18% | 31% | 1.49 |
| Clean favourite wins | 60% | 31% | 0.44 |

The model is excellent on "chalk" and the residual lives in genuinely surprising events
that need information no historical feature contains (lineups, motivation, one-offs).

## What was ruled out (honest negative results)
- **Gradient boosting (HGB) and EBM** — both overfit; LR wins (signal-limited, not capacity-limited).
- **DC rho / bivariate Poisson** — rho not at bound; conditional goal correlation ~0 (no
  positive dependence for bivariate Poisson to model). DC already fits the draw structure.
- **Glicko-2** — volatility signal useless (sigma ~constant); worse than Glicko-1 in the blend.
- **Confederation / host / competition-importance features** — no measurable effect.
- **Team-level H2H aggregates** (mean/std) — re-encode strength; nil.
- **Market value** — ~77% redundant with Elo (corr 0.77); residual-after-Elo only +0.105
  correlation with outcome. Opt-in, Europe-centric, marginal.
- **Injury availability** — historical backtest lift negligible (top-30 mean robust); real
  value is live (auto-excludes current injuries / manual absence override).
- **Draw fixes**: closeness feature (softmax already handles decay) and adaptive LR/DC
  blend (optimal weight ~flat by |elo|) — both null.
- **Pairwise "bogey team" effect beyond Elo** — corr(prior pair residual, current) = +0.017;
  when model and H2H disagree the MODEL is right 62% of the time. H2H post-processing hurts.

## Prediction + betting layer
- `MatchPredictor.predict(home, away, neutral, unavailable={team:[players]})` -> 1X2 blend.
  Uses the synthetic-row method (appends the fixture, re-runs build_features) so live and
  training features can never drift. `predict_components`, `over_under`, `score_matrix`,
  `expected_goals`, `squad(team)` expose the goals model and secondary markets.
- `portfolio.py` — per outcome, edge = p_model*odds - 1; bets above a threshold sized by
  fractional Kelly with exposure caps; works for ANY market (1X2, over/under, ...).

## How to run (from src/)
```
python walk_forward.py             # backtest (LR, no MV)
python run_with_mv.py              # backtest WITH market value (--injuries, --model both)
python dixon_coles.py              # DC vs LR vs blend
python predict.py                  # demo predictions (LR / DC / blend + secondary markets)
python portfolio.py                # demo portfolio (edit fixtures + odds for real use)
```

## Honest limitations & next steps
- **Information ceiling**: ~0.84 log loss / 61% accuracy is good for 3-way football; further
  accuracy needs new information that predicts surprises, not more features.
- **1X2 on major matches is an efficient market.** The realistic edge is in (a) the
  secondary markets the goals model unlocks, and (b) lower-profile games — which is exactly
  where squad coverage thins out.
- **Open / blocking**: a historical odds dataset (opening + closing lines) is required to
  measure Closing Line Value, backtest the betting strategy, devig properly, and build a
  model-vs-market blend (the legitimate "second opinion" — external info the model lacks).
