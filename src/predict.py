"""
Live match prediction -- the foundation for the betting layer.

MatchPredictor fits the model once on all competitive matches up to the latest
date, then scores any hypothetical fixture by the SYNTHETIC-ROW method: it appends
the fixture as one row to the raw results and re-runs the exact same feature
pipeline (wc_pipeline.build_features). Because every feature is a pre-match
snapshot, the trailing fixture row gets correct Elo/form/H2H/market-value features
and its dummy result never affects earlier rows. This means ZERO duplicated
feature logic -- prediction and training features can never drift apart.

Usage:
    from predict import MatchPredictor
    pr = MatchPredictor("../data/results.csv")                  # no market value
    pr = MatchPredictor("../data/results.csv", "country_market_value.csv")  # +MV
    pr.predict("Brazil", "Argentina", neutral=True)
    # -> {'home_win': .., 'draw': .., 'away_win': ..}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import wc_pipeline as wc
import wc_squads as sq
import wc_market_value as mvmod
import walk_forward as wf


class MatchPredictor:
    def __init__(self, data_path: str = None, country_series=None,
                 model_name: str = "lr", train_years: int | None = None):
        data_path = data_path or wc.DATA_PATH
        self.raw = wc.load_results(data_path)
        self._fn_path = os.path.join(os.path.dirname(data_path), "former_names.csv")

        # one canonical feature build for training
        self.df, self.ratings = wc.build_features(self.raw, self._fn_path)
        self.model_name = model_name

        self.use_mv = country_series is not None
        self.features = list(wc.FEATURES)
        self.cs = None
        if self.use_mv:
            self.cs = (mvmod.load_country_series_from_csv(country_series)
                       if isinstance(country_series, str) else country_series)
            self.df = mvmod.attach_market_value(self.df, self.cs)
            self.features += list(mvmod.EXTRA_FEATURES)

        self._fit(model_name, train_years)

    def _fit(self, model_name, train_years):
        train = sq.competitive_only(self.df)
        if train_years:
            cut = train["date"].max() - pd.DateOffset(years=train_years)
            train = train[train["date"] >= cut]
        Xtr = train[self.features].to_numpy()
        ytr = train["target"].to_numpy()
        w = wc.time_decay_weights(train["date"], wc.DECAY_HALFLIFE_DAYS)
        self.model = wf._fit(wf.MODELS[model_name](), Xtr, ytr, w)
        self.n_train = len(train)
        self.trained_through = train["date"].max()

    # -- prediction via the synthetic-row method ----------------------------
    def _feature_row(self, home, away, neutral, date, tournament, country):
        """Append the fixture to the raw results, re-run build_features (same code
        as training), and return the fixture row's feature Series."""
        when = (pd.Timestamp(date) if date is not None
                else self.raw["date"].max() + pd.Timedelta(days=1))
        fixture = {
            "date": when, "home_team": home, "away_team": away,
            "home_score": 0, "away_score": 0,          # dummy; never used for features
            "tournament": tournament, "neutral": bool(neutral),
            "country": country if country is not None else ("" if neutral else home),
            "_is_fixture": True,
        }
        raw = self.raw.copy()
        raw["_is_fixture"] = False
        aug = pd.concat([raw, pd.DataFrame([fixture])], ignore_index=True)

        feat, _ = wc.build_features(aug, self._fn_path, verbose=False)
        if self.use_mv:
            feat = mvmod.attach_market_value(feat, self.cs)
        return feat[feat["_is_fixture"]].iloc[-1]

    def predict(self, home: str, away: str, neutral: bool = True,
                date=None, tournament: str = "FIFA World Cup",
                country: str | None = None) -> dict:
        """Return {'home_win','draw','away_win'} for the fixture.

        `date` (default: day after the latest result) sets the as-of point for
        Elo/form/H2H/market value. `tournament`/`country` only matter if context
        features are enabled (off by default)."""
        row = self._feature_row(home, away, neutral, date, tournament, country)
        x = np.array([[float(row[f]) for f in self.features]])
        p = wf.predict_proba_full(self.model, x)[0]   # ordered [home, draw, away]
        return {"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])}

    def known_team(self, name: str) -> bool:
        return name in self.ratings


if __name__ == "__main__":
    path = "../data/results.csv" if os.path.exists("../data/results.csv") else wc.DATA_PATH
    pr = MatchPredictor(path)
    print(f"Trained on {pr.n_train:,} competitive matches through "
          f"{pr.trained_through.date()} (model={pr.model_name}).\n")
    for h, a in [("Brazil", "Argentina"), ("France", "Germany"),
                 ("Spain", "England"), ("United States", "Mexico"),
                 ("Japan", "Brazil")]:
        p = pr.predict(h, a, neutral=True)
        print(f"  {h} vs {a} (neutral): "
              f"{h} {p['home_win']:.1%} | draw {p['draw']:.1%} | "
              f"{a} {p['away_win']:.1%}")
