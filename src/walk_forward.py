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

Two schemes:
  expanding -> train on everything before the test block (more data each fold)
  rolling   -> train only on a fixed trailing window (adapts faster, less data)

Run:
    python3 wc_walkforward.py        # uses DATA_PATH from wc_pipeline
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score

import wc_pipeline as wc  # build_dataset, FEATURES, time_decay_weights, constants

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


def make_model():
    """Same model as the single-split pipeline; rebuilt fresh for each fold."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )


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
def walk_forward(path: str = wc.DATA_PATH):
    df, _ = wc.build_dataset(path)
    print(f"Feature table: {len(df):,} matches. "
          f"Scheme={SCHEME}, test region={N_TEST_YEARS}y, "
          f"block={STEP_MONTHS}mo.\n")

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
        Xtr, ytr = train[wc.FEATURES].to_numpy(), train["target"].to_numpy()
        Xte, yte = test[wc.FEATURES].to_numpy(), test["target"].to_numpy()

        # decay weights relative to THIS fold's most recent training match
        w = wc.time_decay_weights(train["date"], wc.DECAY_HALFLIFE_DAYS)

        model = make_model()
        model.fit(Xtr, ytr, logisticregression__sample_weight=w)
        proba = predict_proba_full(model, Xte)

        # draw rate learned from this fold's training block only (no peeking)
        draw_rate = float((ytr == 1).mean())
        elo_p = elo_only_proba(test["elo_diff"].to_numpy(), draw_rate)

        ll_model = log_loss(yte, proba, labels=LABELS)
        ll_elo = log_loss(yte, elo_p, labels=LABELS)
        rows.append({
            "fold": f"{b0.date()}–{b1.date()}",
            "n_train": len(train), "n_test": len(test),
            "ll_model": ll_model, "ll_elo": ll_elo,
            "edge": ll_elo - ll_model,                  # +ve = model beats Elo
            "acc": accuracy_score(yte, proba.argmax(1)),
        })
        oof_proba.append(proba)
        oof_y.append(yte)
        oof_elo.append(elo_p)

    res = pd.DataFrame(rows)
    oof_proba = np.vstack(oof_proba)
    oof_y = np.concatenate(oof_y)
    oof_elo = np.vstack(oof_elo)

    # ---- per-fold table ----
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- pooled OOF summary ----
    # Every test block is disjoint, so each match here is scored by exactly the
    # fold model that was trained on its past -- never a single global model.
    # Pooled log loss therefore equals the n_test-weighted mean of the per-fold
    # losses above; it is an honest aggregate of real out-of-fold predictions.
    print("\n=== Out-of-fold totals (each game scored by its own fold's model) ===")
    print(f"  matches scored : {len(oof_y):,}")
    print(
        f"  model log loss : {log_loss(oof_y, oof_proba, labels=LABELS):.4f}")
    print(f"  Elo   log loss : {log_loss(oof_y, oof_elo, labels=LABELS):.4f}")
    print(f"  model Brier    : {multiclass_brier(oof_y, oof_proba):.4f}")
    print(
        f"  model accuracy : {accuracy_score(oof_y, oof_proba.argmax(1)):.3f}")
    wins = (res['edge'] > 0).sum()
    print(f"  folds beating Elo: {wins}/{len(res)}  "
          f"(consistency matters more than the average)")

    plot_folds(res)
    return res, oof_proba, oof_y


def plot_folds(res: pd.DataFrame, path="walkforward_logloss.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(res))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, res["ll_model"], "o-", label="Model", color="#008ABC")
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
