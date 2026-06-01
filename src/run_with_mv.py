"""
One-command runner: market-value model end to end.

It (1) builds the per-country squad-value series from the salimt CSVs in DATA_DIR
(or reuses a cached CSV), then (2) runs the walk-forward backtest with that series
attached as the mv_log_diff / mv_missing features.

Run from the src/ folder:
    python run_with_mv.py                 # build series, then backtest with MV
    python run_with_mv.py --no-mv         # baseline: same backtest, no MV
    python run_with_mv.py --cache mv.csv  # reuse/save the series at mv.csv

DATA_DIR defaults to ../data (where results.csv and the salimt player_* folders live).
"""
from __future__ import annotations
import argparse
import os

import wc_market_value as mvmod
import wc_squad_dataset as sd
import walk_forward as wf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="../data",
                    help="folder with results.csv and the salimt player_* folders")
    ap.add_argument("--cache", default="country_market_value.csv",
                    help="where to cache the built country value series")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the series even if the cache exists")
    ap.add_argument("--no-mv", action="store_true",
                    help="run the baseline backtest without market value")
    ap.add_argument("--model", default="lr", choices=["lr", "hgb", "both"],
                    help="model: logistic regression, gradient boosting, or both")
    args = ap.parse_args()

    results = os.path.join(args.data_dir, "results.csv")

    def run(cs):
        if args.model == "both":
            wf.compare_models(results, country_series=cs)
        else:
            wf.walk_forward(results, country_series=cs, model_name=args.model)

    if args.no_mv:
        print(">> Baseline backtest (no market value)\n")
        run(None)
        return

    # 1) build (or load) the country squad-value series
    if args.rebuild or not os.path.exists(args.cache):
        print(">> Building country squad-value series from salimt CSVs ...")
        cs = sd.build_series_from_dir(args.data_dir)
        cs.to_csv(args.cache, index=False)
        print(f"   cached -> {args.cache}\n")
    else:
        print(f">> Reusing cached series: {args.cache} "
              f"(pass --rebuild to regenerate)\n")
    cs = mvmod.load_country_series_from_csv(args.cache)

    # 2) backtest with the market-value features attached
    print(">> Walk-forward backtest WITH market value\n")
    run(cs)


if __name__ == "__main__":
    main()
