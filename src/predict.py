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
from dixon_coles import DixonColes


class MatchPredictor:
    def __init__(self, data_path: str = None, country_series=None,
                 model_name: str = "lr", train_years: int | None = None,
                 blend: float = 0.5, fit_goals: bool = True,
                 squad_data_dir: str | None = None):
        """blend = weight on the LR classifier in the LR/Dixon-Coles 1X2 blend
        (0.5 was the backtest optimum). fit_goals=False -> pure LR, no goals model
        and no secondary markets. squad_data_dir (with market value ON) enables
        the live `unavailable=` absence override."""
        data_path = data_path or wc.DATA_PATH
        self.raw = wc.load_results(data_path)
        self._fn_path = os.path.join(os.path.dirname(data_path), "former_names.csv")

        # one canonical feature build for training
        self.df, self.ratings = wc.build_features(self.raw, self._fn_path)
        self.model_name = model_name
        self.blend = blend

        self.use_mv = country_series is not None
        self.features = list(wc.FEATURES)
        self.cs = None
        if self.use_mv:
            self.cs = (mvmod.load_country_series_from_csv(country_series)
                       if isinstance(country_series, str) else country_series)
            self.df = mvmod.attach_market_value(self.df, self.cs)
            self.features += list(mvmod.EXTRA_FEATURES)

        self._fit(model_name, train_years)
        # goals model: fit on ALL matches up to the latest date (ratings need them)
        self.dc = (DixonColes().fit(self.df, ref_date=self.df["date"].max())
                   if fit_goals else None)
        # player-level valuer for the live absence override (MV must be on)
        self.valuer = None
        if self.use_mv and squad_data_dir is not None:
            from wc_squad_dataset import SquadValuer
            self.valuer = SquadValuer(squad_data_dir)

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

    def _lr_from_row(self, row) -> dict:
        x = np.array([[float(row[f]) for f in self.features]])
        p = wf.predict_proba_full(self.model, x)[0]   # ordered [home, draw, away]
        return {"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])}

    def predict_lr(self, home, away, neutral=True, date=None,
                   tournament="FIFA World Cup", country=None) -> dict:
        """LR-classifier 1X2 probabilities (synthetic-row features)."""
        return self._lr_from_row(
            self._feature_row(home, away, neutral, date, tournament, country))

    def _override_mv(self, row, home, away, date, unavailable):
        """Replace the fixture's market-value feature using the SquadValuer with
        the named players dropped from each team's top-N."""
        when = pd.Timestamp(date) if date is not None else self.raw["date"].max()
        vh = self.valuer.value(home, when, unavailable.get(home, []))
        va = self.valuer.value(away, when, unavailable.get(away, []))
        lh = np.log10(vh) if vh and vh > 0 else np.nan
        la = np.log10(va) if va and va > 0 else np.nan
        row = row.copy()
        row["mv_log_diff"] = lh - la
        row["mv_missing"] = int(np.isnan(lh - la))
        return row

    def predict(self, home: str, away: str, neutral: bool = True,
                date=None, tournament: str = "FIFA World Cup",
                country: str | None = None, unavailable: dict | None = None) -> dict:
        """1X2 probabilities -- the LR/Dixon-Coles blend by default (backtest
        best: 0.8484 -> 0.8419). Pure LR if fit_goals=False or blend>=1.

        unavailable = {team: [player names/ids]} drops those players from the
        team's squad-value (needs market value ON + squad_data_dir)."""
        row = self._feature_row(home, away, neutral, date, tournament, country)
        if unavailable and self.valuer is not None and self.use_mv:
            row = self._override_mv(row, home, away, date, unavailable)
        lr = self._lr_from_row(row)
        if self.dc is None or self.blend >= 1.0:
            return lr
        dc = self.dc.predict(home, away, neutral)
        b = self.blend
        return {k: b * lr[k] + (1 - b) * dc[k] for k in lr}

    def squad(self, team, date=None, exclude=()):
        """The available top-N players (id, name, value) the valuer would use."""
        when = pd.Timestamp(date) if date is not None else self.raw["date"].max()
        import wc_squad_dataset as sd
        pc = sd.NAME_ALIAS.get(team, team)
        ex = set()
        for x in exclude:
            ex |= self.valuer.resolve(x, self.valuer.players_by_country.get(pc, set()))
        return self.valuer.squad(pc, when, ex)

    def predict_components(self, home, away, neutral=True, **kw) -> dict:
        """{'lr':..,'dc':..,'blend':..} -- to see each model's view."""
        out = {"lr": self.predict_lr(home, away, neutral)}
        if self.dc is not None:
            out["dc"] = self.dc.predict(home, away, neutral)
            out["blend"] = self.predict(home, away, neutral, **kw)
        return out

    # -- secondary markets (Dixon-Coles scoreline matrix) -------------------
    # For World Cup fixtures pass wc=True (applies dixon_coles.WC_GOAL_SCALE),
    # which corrects DC's ~8% goal under-prediction at WC finals -- improves
    # totals / correct-score (1X2 is barely affected).
    def _gs(self, wc):
        from dixon_coles import WC_GOAL_SCALE
        return WC_GOAL_SCALE if wc else 1.0

    def _reconcile(self, M, blend):
        """Rescale the DC scoreline matrix so its W/D/L marginals equal the LR+DC
        blend -- injects the better-calibrated 1X2 into the secondary markets
        (esp. handicap/margin) while keeping DC's within-outcome score shape."""
        import dixon_coles as dcm
        dc12 = dcm.onextwo_from_matrix(M)
        n = M.shape[0]
        diff = np.subtract.outer(np.arange(n), np.arange(n))
        M2 = M.copy()
        for region, key in ((diff > 0, "home_win"), (diff == 0, "draw"),
                            (diff < 0, "away_win")):
            if dc12[key] > 0:
                M2[region] *= blend[key] / dc12[key]
        return M2 / M2.sum()

    def _market_matrix(self, home, away, neutral, wc, reconcile, blend=None):
        M = self.dc.score_matrix(home, away, neutral, goal_scale=self._gs(wc))
        if reconcile:
            M = self._reconcile(M, blend or self.predict(home, away, neutral))
        return M

    def predict_markets(self, home, away, neutral=True, odds_keys=None,
                        wc=False, reconcile=True):
        """Returns (probs, pushes). Handles, by odds key:
          home_win/draw/away_win    -- 1X2 blend
          over_X / under_X          -- totals line X
          goals_0_1/2_3/4plus       -- total-goals buckets
          hcap_home_-1 / hcap_draw_-1 / hcap_away_+1 / ... -- Taiwan 3-way handicap
        Taiwan handicap is a 3-way market: on an integer line the adjusted tie is
        a separate bettable `hcap_draw_*` outcome, NOT a stake refund. `pushes` is
        therefore always empty (kept for the caller's tuple unpack); 3-way EV uses
        no push term. reconcile=True derives the secondary markets from the
        blend-rescaled scoreline matrix (better-calibrated 1X2 baked in)."""
        import dixon_coles as dcm
        probs = self.predict(home, away, neutral)
        pushes = {}
        if self.dc is None or not odds_keys:
            return probs, pushes
        M = self._market_matrix(home, away, neutral, wc, reconcile, blend=probs)
        for k in odds_keys:
            if k.startswith(("over_", "under_")):
                try:
                    line = float(k.split("_", 1)[1])
                except ValueError:
                    continue
                probs.update(dcm.ou_from_matrix(M, line))
            elif k in ("goals_0_1", "goals_2_3", "goals_4plus"):
                probs.update(dcm.totals_from_matrix(M))
            elif k.startswith("hcap_"):
                try:
                    _, side, lv = k.split("_")
                    lineval = float(lv)
                except ValueError:
                    continue
                # home & draw keys are quoted on the HOME line; away flips sign
                hline = -lineval if side == "away" else lineval
                h = dcm.hcap_from_matrix(M, hline)
                # 3-way outcomes: home cover / handicap draw / away cover.
                # h["push"] (adjusted-tie mass) is the draw probability here.
                probs[k] = {"home": h["home_cover"], "away": h["away_cover"],
                            "draw": h["push"]}.get(side)
        return probs, pushes

    def score_matrix(self, home, away, neutral=True, wc=False, reconcile=True):
        return self._market_matrix(home, away, neutral, wc, reconcile)

    def over_under(self, home, away, line=2.5, neutral=True, wc=False, reconcile=True):
        import dixon_coles as dcm
        return dcm.ou_from_matrix(self._market_matrix(home, away, neutral, wc, reconcile), line)

    def handicap(self, home, away, line=-1.0, neutral=True, wc=False, reconcile=True):
        import dixon_coles as dcm
        return dcm.hcap_from_matrix(self._market_matrix(home, away, neutral, wc, reconcile), line)

    def total_buckets(self, home, away, neutral=True, wc=False, reconcile=True):
        import dixon_coles as dcm
        return dcm.totals_from_matrix(self._market_matrix(home, away, neutral, wc, reconcile))

    def expected_goals(self, home, away, neutral=True, wc=False):
        return self.dc.expected_goals(home, away, neutral, goal_scale=self._gs(wc))

    def known_team(self, name: str) -> bool:
        return name in self.ratings


if __name__ == "__main__":
    path = "../data/results.csv" if os.path.exists("../data/results.csv") else wc.DATA_PATH
    pr = MatchPredictor(path)
    print(f"Trained on {pr.n_train:,} competitive matches through "
          f"{pr.trained_through.date()} | LR+DC blend (w_lr={pr.blend}).\n")
    for h, a in [("Brazil", "Argentina"), ("Spain", "England"),
                 ("United States", "Mexico")]:
        c = pr.predict_components(h, a, neutral=True)
        ou = pr.over_under(h, a, 2.5, neutral=True)
        xg = pr.expected_goals(h, a, neutral=True)
        print(f"  {h} vs {a} (neutral):")
        for tag in ("lr", "dc", "blend"):
            q = c[tag]
            print(f"    {tag:<5} {q['home_win']:.0%} / {q['draw']:.0%} / {q['away_win']:.0%}")
        print(f"    xG {xg['home_xg']:.2f}-{xg['away_xg']:.2f} | "
              f"O2.5 {ou['over_2.5']:.0%} / U2.5 {ou['under_2.5']:.0%}\n")
