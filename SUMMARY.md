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
| predict.py | MatchPredictor: LR+DC blend, secondary markets (blend-reconciled), absence override |
| portfolio.py | Devig (Shin) + EV + push-aware fractional-Kelly across any market |
| run_with_mv.py | One-command end-to-end runner |
| chemistry.py | Causal club-chemistry feature (built, tested NULL; reference tool) |

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

## Trials ledger — what worked and what didn't
Every idea below was tested with a proper causal backtest (not intuition). The pattern is the
headline result: **gains came from a richer target and more data; everything that tried to add
"a new signal beyond team strength" failed**, because Elo already integrates whatever shows up
in results.

| Trial | Outcome |
|---|---|
| Offense/defense form split + head-to-head | ✅ 0.8484 -> 0.8449 |
| Dixon-Coles goals model, blended with LR | ✅ 0.8419; fixes draws, unlocks secondary markets |
| Glicko-1 uncertainty (`glicko_exp`) | ✅ 0.8405; 16/17 folds beat Elo |
| Dixon-Coles on expanding history + ~5y decay | ✅ 0.8393 (current SOTA) |
| World Cup goal-scale (secondary markets) | ✅ real, ~2.6 SE effect; sharpens totals/handicap |
| Blend-reconciled secondary markets; devig + `vs_fair` | ✅ betting-layer correctness |
| Gradient boosting (HGB) / EBM | ❌ overfit; LR wins (signal-limited, not capacity-limited) |
| DC rho widening / bivariate Poisson | ❌ rho not at bound; conditional goal corr ~0 |
| Glicko-2 volatility | ❌ sigma ~constant; worse than Glicko-1 in the blend |
| Confederation / host / competition-importance | ❌ no measurable effect |
| Team-level H2H aggregates (mean/std) | ❌ re-encode strength; nil |
| Closeness feature / adaptive LR-DC blend (for draws) | ❌ both null |
| Pairwise "bogey team" beyond Elo; H2H post-processing | ❌ corr +0.02; post-process hurts |
| Momentum / win-streak | ❌ residual corr ~0; captured by Elo+form |
| Market value as a strength feature | ⚠️ ~77% redundant with Elo; opt-in, marginal |
| Injury availability (historical) | ⚠️ negligible backtest lift; real value is LIVE |
| Squad chemistry (causal, from transfer_history) | ❌ orthogonal to strength but residual corr -0.007 |
| Squad-value concentration (Gini / CV / top-1) | ❌ distribution shape; residual corr ~0 |

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
- **Momentum / win-streak** — residual corr ~0 (-0.02); already captured by Elo + form.
- **Squad CHEMISTRY** (causal, dated from transfer_history) — orthogonal to strength (first
  non-redundant idea!) but corr(chem_diff, residual) = -0.007; small-nation/few-clubs artifact,
  not a winning signal.
- **Squad-value CONCENTRATION** (Gini / CV / top-1 share) — distribution shape, also null
  (residual corr ~0). Pattern confirmed: orthogonal != predictive; Elo integrates everything
  that shows up in results.

One thing that DID test real beyond the model: **World Cup goal-scale** -- WC finals score
~8% more than DC predicts (validated, ~2.6 SE / 444 matches), so `wc=True` applies a 1.08
scaling that sharpens totals / correct-score (1X2 unchanged).

## Prediction + betting layer
- `MatchPredictor.predict(home, away, neutral, unavailable={team:[players]})` -> 1X2 blend.
  Synthetic-row method (appends the fixture, re-runs build_features) so live and training
  features can never drift. `unavailable=` drops named players (absence override); `wc=True`
  applies the World Cup goal-scale.
- **Markets** (all from the Dixon-Coles scoreline matrix): 1X2, over/under, total-goals
  buckets (0-1 / 2-3 / 4+), Asian handicap +/-1 (with push). Secondary markets are derived
  from a matrix RECONCILED to the blend's better 1X2 -- so the handicap inherits the
  blend's calibration (0.8393) plus DC's "win by 1 vs by 2+" margin shape.
- `portfolio.py` — **devig** (Shin / multiplicative) gives the bookmaker's fair probability;
  `vs_fair` = p_model - fair flags genuine disagreement with the sharp market vs a vig
  illusion. **Push-aware** EV/Kelly (edge = p*o - (1-push)); fractional Kelly + exposure caps;
  any market.

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
