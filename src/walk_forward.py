"""
Walk-forward validation on top of wc_pipeline.

Why this and not the single split:
  A single train/test split tells you how the model did on ONE recent slice,
  which can be lucky or unlucky. Walk-forward retrains the model repeatedly,
  always training on the past and testing on the immediately following block,
  then rolls forward. You get:
    * an out-of-fold (OOF) score where each game is scored by the one fold model
      trained on its past (disjoint blocks) -> a robust, honest aggregate,
    * a per-fold breakdown -> shows whether the edge over the Elo baseline is
      consistent over time or driven by one good era,
    * a plot of model-vs-baseline log loss per fold.

No leakage:
  Features (Elo, form) are already causal in wc_pipeline -- each row only sees
  prior matches -- so the feature table is built ONCE. Only the model and its
  StandardScaler are refit per fold, on that fold's training rows alone.

Competitive-only modelling:
  Elo and form are still computed over the FULL history (so ratings converge),
  but friendlies are dropped from the train/test rows. Friendlies are noisy
  (experimental line-ups) and, crucially, lack accessible squad data -- keeping
  them would force a mostly-missing market-value feature that adds no signal.

Market value (optional):
  Pass a country_series (or CSV path) and the loop attaches a strictly-pre-match
  squad-value feature (mv_log_diff + an mv_missing flag) and switches to the
  median-imputing model. A coverage guard drops the feature if too few matches
  actually have data, so a sparse series can't quietly corrupt the model.

Two schemes:
  expanding -> train on everything before the test block (more data each fold)
  rolling   -> train only on a fixed trailing window (adapts faster, less data)

Run:
    python3 walk_forward.py          # competitive-only, no market value
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score

import wc_pipeline as wc  # build_dataset, FEATURES, time_decay_weights, constants
import wc_squads as sq    # competitive_only, EXTRA_FEATURES
import wc_market_value as mvmod  # attach_market_value

LABELS = [0, 1, 2]  # home win / draw / away win

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
N_TEST_YEARS = 8        # how far back the walk-forward test region reaches
# size of each test block (12 = test one year at a time)
STEP_MONTHS = 6
SCHEME = "rolling"    # "expanding" or "rolling"
ROLL_TRAIN_YEARS = 8    # trailing train window when SCHEME == "rolling"
MIN_TRAIN = 500         # skip a fold if the train set is smaller than this
COMPETITIVE_ONLY = True  # drop friendlies from modelling (keep them in Elo/form)
MV_MIN_COVERAGE = 0.50   # disable market value if < this fraction of rows have it
# When market value is ON, drop whole confederations whose MV coverage is below
# this (keeps UEFA/CONMEBOL/CONCACAF/CAF/World Cup; drops AFC/OFC/regional).
# Set to 0.0 to keep every confederation.
MV_CONFED_MIN_COVERAGE = 0.50


def make_lr():
    """Logistic regression. Median-impute first so it tolerates the NaNs market
    value introduces (a no-op when no feature is missing), then standardize."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def make_hgb():
    """Gradient-boosted trees. Handles NaN natively (no imputer) and is scale-
    invariant; captures interactions LR cannot.

    NOTE (Step 2 finding): on the current NUMERIC features and ~4-5k-row folds,
    even this heavily-regularized config does NOT beat logistic regression
    (OOS ~0.855 vs LR 0.845) -- the bottleneck is signal in the features, not
    model capacity, so the trees only find interactions that don't generalize.
    LR stays the default. HGB is kept for Step 3 (categorical confederation/host
    features, which trees handle natively) and Step 4 (ensembling). Config is the
    most-regularized point on the frontier (depth 2, big leaves, strong L2, early
    stopping) to keep the IS->OOS gap small."""
    return make_pipeline(
        HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.03,
            max_iter=500, max_depth=2, min_samples_leaf=300,
            l2_regularization=10.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=0,
        ),
    )


MODELS = {"lr": make_lr, "hgb": make_hgb}
make_model = make_lr   # back-compat alias


def _fit(est, Xtr, ytr, w):
    """Fit a pipeline passing sample_weight to whatever its final step is named
    (logisticregression / histgradientboostingclassifier / ...)."""
    step = est.steps[-1][0]
    est.fit(Xtr, ytr, **{f"{step}__sample_weight": w})
    return est


def multiclass_brier(y_true, proba, n_classes=3):
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def predict_proba_full(model, X, labels=LABELS) -> np.ndarray:
    """predict_proba scattered back onto the full label set.

    If a fold's training block happens to contain only a subset of the three
    outcomes, sklearn returns one probability column per *seen* class. We map
    those columns onto a fixed (n, len(labels)) matrix so downstream log_loss
    with labels=LABELS always lines up; any class never seen in training gets
    probability 0 (a true penalty, not a crash).
    """
    proba_seen = model.predict_proba(X)
    full = np.zeros((X.shape[0], len(labels)))
    col_of = {c: j for j, c in enumerate(labels)}
    for k, c in enumerate(model.classes_):
        full[:, col_of[c]] = proba_seen[:, k]
    return full


def elo_only_proba(elo_diff: np.ndarray, draw_rate: float) -> np.ndarray:
    """Elo-only 3-way baseline.

    The home/away split comes from the Elo win expectancy. The draw mass is the
    empirical draw rate of THIS fold's training block (causal -- it never looks
    at the test outcomes) rather than a hardcoded 0.2 constant. The remaining
    (1 - draw_rate) is allocated to home/away in proportion to the Elo
    expectancy, so a bigger rating gap -> more mass on the favourite.
    """
    elo_diff = np.asarray(elo_diff, dtype=float)
    # Elo expected score for home
    p_home = 1 / (1 + 10 ** (-elo_diff / 400))
    decisive = 1.0 - draw_rate
    p = np.column_stack([
        decisive * p_home,
        np.full(len(elo_diff), draw_rate),
        decisive * (1 - p_home),
    ])
    p = np.clip(p, 1e-6, 1)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------
def generate_folds(df: pd.DataFrame):
    """Yield (train, test, block_start, block_end) tuples, oldest first."""
    max_date = df["date"].max()
    block_start = (max_date - pd.DateOffset(years=N_TEST_YEARS)).normalize()

    while block_start < max_date:
        block_end = block_start + pd.DateOffset(months=STEP_MONTHS)
        test = df[(df.date >= block_start) & (df.date < block_end)]

        if SCHEME == "rolling":
            train_lo = block_start - pd.DateOffset(years=ROLL_TRAIN_YEARS)
            train = df[(df.date < block_start) & (df.date >= train_lo)]
        else:  # expanding
            train = df[df.date < block_start]

        if len(train) >= MIN_TRAIN and len(test) > 0:
            yield train, test, block_start, block_end
        block_start = block_end


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------
def _resolve_country_series(country_series):
    """Accept a DataFrame, a CSV path, or None."""
    if country_series is None:
        return None
    if isinstance(country_series, pd.DataFrame):
        return country_series
    return mvmod.load_country_series_from_csv(country_series)


def walk_forward(path: str = wc.DATA_PATH, country_series=None,
                 model_name: str = "lr", verbose: bool = True):
    def say(*a):
        if verbose:
            print(*a)
    if model_name not in MODELS:
        raise ValueError(f"unknown model_name={model_name!r}; use {list(MODELS)}")
    make = MODELS[model_name]

    # Elo + form are built over the FULL history (ratings need every match).
    df, _ = wc.build_dataset(path)
    n_full = len(df)

    # Restrict MODELLING to competitive matches (Elo/form already baked in).
    if COMPETITIVE_ONLY:
        df = sq.competitive_only(df).reset_index(drop=True)

    # Optionally attach a strictly-pre-match squad-value feature.
    cs = _resolve_country_series(country_series)
    use_mv = cs is not None
    features = list(wc.FEATURES)
    if use_mv:
        df = mvmod.attach_market_value(df, cs)
        # Coverage must be measured over the matches that actually enter folds
        # (the rolling train window + test region), NOT all of history -- else
        # ancient uncovered matches drag the fraction down and wrongly trip the
        # guard even when the recent modelling era is fully covered.
        earliest_block = (df["date"].max()
                          - pd.DateOffset(years=N_TEST_YEARS)).normalize()
        earliest_used = (earliest_block - pd.DateOffset(years=ROLL_TRAIN_YEARS)
                         if SCHEME == "rolling" else df["date"].min())

        # Drop whole confederations whose MV coverage is below the threshold, so
        # modelling focuses on regions where the feature is actually informative.
        if MV_CONFED_MIN_COVERAGE > 0:
            say(f"Confederation MV filter (>= {MV_CONFED_MIN_COVERAGE:.0%} "
                f"coverage in modelling window):")
            df = sq.keep_high_mv_confederations(
                df, MV_CONFED_MIN_COVERAGE, verbose=verbose,
                window_mask=(df["date"] >= earliest_used)).reset_index(drop=True)
            say()

        win = df[df["date"] >= earliest_used]
        coverage = 1.0 - float(win["mv_missing"].mean())
        if coverage < MV_MIN_COVERAGE:
            say(f"Market value coverage {coverage:.1%} (over modelling "
                f"window) < {MV_MIN_COVERAGE:.0%} -> disabling the feature to "
                f"avoid a mostly-imputed (corrupting) column.\n")
            use_mv = False
        else:
            features = list(wc.FEATURES) + list(mvmod.EXTRA_FEATURES)
            say(f"Market value ON: coverage {coverage:.1%} over modelling "
                f"window (since {earliest_used.date()}).")

    say(f"Modelling table: {len(df):,}/{n_full:,} matches "
        f"({'competitive only' if COMPETITIVE_ONLY else 'all'}). model={model_name}, "
        f"Scheme={SCHEME}, test region={N_TEST_YEARS}y, block={STEP_MONTHS}mo, "
        f"features={len(features)}.\n")

    folds = list(generate_folds(df))
    if not folds:
        print("No folds met the criteria -- the test region is empty or every "
              "candidate train block is below MIN_TRAIN. Check N_TEST_YEARS, "
              "STEP_MONTHS and MIN_TRAIN against the size of the data.")
        return None

    rows = []
    oof_proba, oof_y = [], []          # each game scored by its own fold's model
    oof_elo = []

    for train, test, b0, b1 in folds:
        Xtr, ytr = train[features].to_numpy(), train["target"].to_numpy()
        Xte, yte = test[features].to_numpy(), test["target"].to_numpy()

        # decay weights relative to THIS fold's most recent training match
        w = wc.time_decay_weights(train["date"], wc.DECAY_HALFLIFE_DAYS)

        est = _fit(make(), Xtr, ytr, w)
        proba = predict_proba_full(est, Xte)
        proba_tr = predict_proba_full(est, Xtr)     # in-sample (same train rows)

        # draw rate learned from this fold's training block only (no peeking)
        draw_rate = float((ytr == 1).mean())
        elo_p = elo_only_proba(test["elo_diff"].to_numpy(), draw_rate)

        ll_is = log_loss(ytr, proba_tr, labels=LABELS)   # in-sample (optimistic)
        ll_model = log_loss(yte, proba, labels=LABELS)   # out-of-sample
        ll_elo = log_loss(yte, elo_p, labels=LABELS)
        row = {
            "fold": f"{b0.date()}–{b1.date()}",
            "n_train": len(train), "n_test": len(test),
            "ll_is": ll_is,                             # in-sample reference
            "ll_oos": ll_model,                         # out-of-sample (the real one)
            "gap": ll_model - ll_is,                    # +ve = optimism/overfit
            "ll_elo": ll_elo,
            "edge": ll_elo - ll_model,                  # +ve = model beats Elo
            "acc": accuracy_score(yte, proba.argmax(1)),
        }
        if use_mv:  # per-fold train coverage -> spot folds the feature can't help
            row["mv_cov"] = 1.0 - float(train["mv_missing"].mean())
        rows.append(row)
        oof_proba.append(proba)
        oof_y.append(yte)
        oof_elo.append(elo_p)

    res = pd.DataFrame(rows)
    oof_proba = np.vstack(oof_proba)
    oof_y = np.concatenate(oof_y)
    oof_elo = np.vstack(oof_elo)

    # ---- per-fold table ----
    say(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- pooled OOF summary ----
    # Every test block is disjoint, so each match here is scored by exactly the
    # fold model that was trained on its past -- never a single global model.
    # Pooled log loss therefore equals the n_test-weighted mean of the per-fold
    # losses above; it is an honest aggregate of real out-of-fold predictions.
    say("\n=== Out-of-fold totals (each game scored by its own fold's model) ===")
    say(f"  matches scored   : {len(oof_y):,}")
    say(f"  model OOS log loss: {log_loss(oof_y, oof_proba, labels=LABELS):.4f}")
    say(f"  Elo   OOS log loss: {log_loss(oof_y, oof_elo, labels=LABELS):.4f}")
    say(f"  model Brier      : {multiclass_brier(oof_y, oof_proba):.4f}")
    say(f"  model accuracy   : {accuracy_score(oof_y, oof_proba.argmax(1)):.3f}")
    # In-sample reference: mean of per-fold IS log loss vs OOS. A small gap means
    # the model is not overfitting (expected for logistic regression); a large
    # gap would warn that a more complex model is memorising the train block.
    say(f"  in-sample log loss: {res['ll_is'].mean():.4f}  (avg of folds)")
    say(f"  IS->OOS gap      : {res['gap'].mean():+.4f}  "
        f"(optimism; near 0 = not overfitting)")
    wins = (res['edge'] > 0).sum()
    say(f"  folds beating Elo: {wins}/{len(res)}  "
        f"(consistency matters more than the average)")

    if verbose:
        plot_folds(res)
    return res, oof_proba, oof_y


def compare_models(path: str = wc.DATA_PATH, country_series=None,
                   models=("lr", "hgb")):
    """Run several models through the SAME walk-forward and print one comparison
    table (pooled out-of-sample). Each model is scored on identical folds."""
    tag = "with market value" if country_series is not None else "no market value"
    print(f"Model comparison ({tag}), pooled out-of-fold:\n")
    hdr = (f"{'model':<6}{'OOS_ll':>9}{'Brier':>8}{'acc':>7}"
           f"{'IS_ll':>8}{'gap':>8}{'beatElo':>9}")
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for m in models:
        res, oofp, oofy = walk_forward(path, country_series,
                                       model_name=m, verbose=False)
        ll = log_loss(oofy, oofp, labels=LABELS)
        br = multiclass_brier(oofy, oofp)
        ac = accuracy_score(oofy, oofp.argmax(1))
        wins, nf = int((res["edge"] > 0).sum()), len(res)
        print(f"{m:<6}{ll:>9.4f}{br:>8.4f}{ac:>7.3f}"
              f"{res['ll_is'].mean():>8.4f}{res['gap'].mean():>+8.4f}"
              f"{f'{wins}/{nf}':>9}")
        results[m] = (res, oofp, oofy)
    return results


def plot_folds(res: pd.DataFrame, path="walkforward_logloss.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(res))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, res["ll_oos"], "o-", label="Model (out-of-sample)", color="#008ABC")
    if "ll_is" in res:
        ax.plot(x, res["ll_is"], "^:", label="Model (in-sample)", color="#7FC6E0")
    ax.plot(x, res["ll_elo"], "s--",
            label="Elo-only baseline", color="#999999")
    ax.set_xticks(list(x))
    ax.set_xticklabels(res["fold"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log loss (lower = better)")
    ax.set_title("Walk-forward: does the model beat Elo in every fold?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved per-fold plot -> {path}")


if __name__ == "__main__":
    walk_forward(wc.DATA_PATH)
