"""
Dixon-Coles goals model (Stage 1 of the "push past the ceiling" plan).

Estimates per-team ATTACK and DEFENSE strengths + a home-advantage term by
time-weighted maximum likelihood from match GOALS (team-level, fully covered --
no Europe/market-value gap), with the Dixon-Coles low-score correction that fixes
the 0-0 / 1-1 / draw structure. From the fitted rates it builds the full scoreline
probability matrix, which yields:
  * 1X2 (home/draw/away)  -- to compare against the LR classifier,
  * totals (over/under), Asian handicaps, correct score -- the softer secondary
    markets 1X2 can't reach.

Why this is the right Stage 1: our LR residual is concentrated in DRAWS
(per-class log loss draw 1.43 vs home 0.57), exactly the structure a goals model
repairs; and it uses only goals, so it generalises to every match.

    from dixon_coles import DixonColes, backtest
    backtest("../data/results.csv")          # DC vs LR vs blend, same folds
    dc = DixonColes().fit(history_df, ref_date=pd.Timestamp("2026-06-01"))
    dc.predict("Brazil", "Argentina", neutral=True)     # {home_win, draw, away_win}
    dc.score_matrix(...)                                 # full P(x goals, y goals)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def _log_pois(k, lam):
    return k * np.log(lam) - lam - gammaln(k + 1)


# -- markets derived from a scoreline matrix M (home goals x away goals) -------
def _grids(n):
    i = np.arange(n)
    return np.subtract.outer(i, i), np.add.outer(i, i)   # home-away diff, total


def onextwo_from_matrix(M) -> dict:
    diff, _ = _grids(M.shape[0])
    return {"home_win": float(M[diff > 0].sum()), "draw": float(M[diff == 0].sum()),
            "away_win": float(M[diff < 0].sum())}


def ou_from_matrix(M, line=2.5) -> dict:
    _, tot = _grids(M.shape[0])
    over = float(M[tot > line].sum())
    return {f"over_{line}": over, f"under_{line}": 1 - over}


def totals_from_matrix(M) -> dict:
    _, tot = _grids(M.shape[0])
    return {"goals_0_1": float(M[tot <= 1].sum()),
            "goals_2_3": float(M[(tot >= 2) & (tot <= 3)].sum()),
            "goals_4plus": float(M[tot >= 4].sum())}


def hcap_from_matrix(M, line=-1.0) -> dict:
    diff, _ = _grids(M.shape[0])
    adj = diff + line
    return {"home_cover": float(M[adj > 0].sum()),
            "push": float(M[adj == 0].sum()) if float(line) == int(line) else 0.0,
            "away_cover": float(M[adj < 0].sum())}


class DixonColes:
    def __init__(self, xi: float = 0.0004, reg: float = 0.5,
                 cap_goals: int = 10, max_iter: int = 400):
        # xi: time-decay/day (~4.7yr half-life). International teams play rarely,
        #     so DC is data-starved -- fit it on ALL history (expanding window)
        #     and use a LONG half-life; this beat the old ~2yr setting in the
        #     walk-forward (blend 0.8405 -> 0.8393).
        # reg: L2 shrinkage on attack/defense (helps rarely-playing teams)
        # cap_goals: clip goals when fitting so blowouts (e.g. 31-0) don't dominate
        self.xi, self.reg, self.cap, self.max_iter = xi, reg, cap_goals, max_iter

    # -- fit -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame, ref_date=None):
        d = df.dropna(subset=["home_score", "away_score"])
        teams = sorted(set(d["home_team"]) | set(d["away_team"]))
        self.teams = teams
        idx = {t: i for i, t in enumerate(teams)}
        T = len(teams)

        hi = d["home_team"].map(idx).to_numpy()
        ai = d["away_team"].map(idx).to_numpy()
        x = np.minimum(d["home_score"].to_numpy(), self.cap).astype(float)
        y = np.minimum(d["away_score"].to_numpy(), self.cap).astype(float)
        neutral = d["neutral"].to_numpy().astype(bool)
        ref = pd.Timestamp(ref_date) if ref_date is not None else d["date"].max()
        age = (ref - d["date"]).dt.days.to_numpy().clip(min=0)
        w = np.exp(-self.xi * age)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        def unpack(p):
            inter, ha, rho = p[0], p[1], p[2]
            att = p[3:3 + T]
            dfn = p[3 + T:3 + 2 * T]
            return inter, ha, rho, att - att.mean(), dfn - dfn.mean()

        def nll(p):
            inter, ha, rho, att, dfn = unpack(p)
            lam = np.exp(inter + att[hi] - dfn[ai] + np.where(neutral, 0.0, ha))
            mu = np.exp(inter + att[ai] - dfn[hi])
            ll = _log_pois(x, lam) + _log_pois(y, mu)
            tau = np.ones(len(x))
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            ll = ll + np.log(np.clip(tau, 1e-9, None))
            return -(w * ll).sum() + self.reg * (att @ att + dfn @ dfn)

        p0 = np.concatenate([[0.3, 0.25, -0.05], np.zeros(2 * T)])
        bounds = [(-2, 2), (-1, 1), (-0.2, 0.2)] + [(-3, 3)] * (2 * T)
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter})
        self.inter, self.ha, self.rho, att, dfn = unpack(res.x)
        self.att = dict(zip(teams, att))
        self.dfn = dict(zip(teams, dfn))
        self.converged = res.success
        return self

    # -- predict -------------------------------------------------------------
    def rates(self, home, away, neutral):
        ah, dh = self.att.get(home, 0.0), self.dfn.get(home, 0.0)
        aa, da = self.att.get(away, 0.0), self.dfn.get(away, 0.0)
        lam = np.exp(self.inter + ah - da + (0.0 if neutral else self.ha))
        mu = np.exp(self.inter + aa - dh)
        return float(lam), float(mu)

    def score_matrix(self, home, away, neutral=True, max_goals=10,
                     goal_scale=1.0) -> np.ndarray:
        """`goal_scale` multiplies both teams' expected goals. World Cup finals
        score ~8% more than DC (fit on all internationals) predicts -- and DC
        over-predicts 0-0/1-1 there -- so pass ~1.08 for WC games (validated at
        ~2.6 SE on 444 WC matches). Mainly improves totals/correct-score; 1X2 is
        barely affected since both rates scale together."""
        lam, mu = self.rates(home, away, neutral)
        lam, mu = lam * goal_scale, mu * goal_scale
        ph = np.exp(_log_pois(np.arange(max_goals + 1), lam))
        pa = np.exp(_log_pois(np.arange(max_goals + 1), mu))
        M = np.outer(ph, pa)
        M[0, 0] *= 1 - lam * mu * self.rho
        M[0, 1] *= 1 + lam * self.rho
        M[1, 0] *= 1 + mu * self.rho
        M[1, 1] *= 1 - self.rho
        M = np.clip(M, 0, None)
        return M / M.sum()

    def predict(self, home, away, neutral=True, max_goals=10, goal_scale=1.0) -> dict:
        """1X2 probabilities from the scoreline matrix."""
        M = self.score_matrix(home, away, neutral, max_goals, goal_scale)
        return {"home_win": float(np.tril(M, -1).sum()),
                "draw": float(np.trace(M)),
                "away_win": float(np.triu(M, 1).sum())}

    # -- secondary markets (the real payoff) --------------------------------
    def over_under(self, home, away, line=2.5, neutral=True, max_goals=10,
                   goal_scale=1.0) -> dict:
        return ou_from_matrix(
            self.score_matrix(home, away, neutral, max_goals, goal_scale), line)

    def expected_goals(self, home, away, neutral=True, goal_scale=1.0) -> dict:
        lam, mu = self.rates(home, away, neutral)
        return {"home_xg": lam * goal_scale, "away_xg": mu * goal_scale,
                "total_xg": (lam + mu) * goal_scale}

    def handicap(self, home, away, line=-1.0, neutral=True, max_goals=10,
                 goal_scale=1.0) -> dict:
        """Asian handicap applied to the HOME team (line<0 => home favoured, must
        win by more than |line|). Cover probabilities with the PUSH (exact line,
        stake returned) separated out -- needed for correct EV."""
        return hcap_from_matrix(
            self.score_matrix(home, away, neutral, max_goals, goal_scale), line)

    def total_buckets(self, home, away, neutral=True, max_goals=10,
                      goal_scale=1.0) -> dict:
        """Total-goals buckets 0-1 / 2-3 / 4+ (mutually exclusive, sum to 1)."""
        return totals_from_matrix(
            self.score_matrix(home, away, neutral, max_goals, goal_scale))


WC_GOAL_SCALE = 1.08   # World Cup finals score ~8% more than the all-matches fit


# ---------------------------------------------------------------------------
# Backtest: DC vs LR on identical walk-forward folds (+ blends)
# ---------------------------------------------------------------------------
def backtest(path: str = None):
    import wc_pipeline as wc
    import wc_squads as sq
    import walk_forward as wf
    from sklearn.metrics import log_loss, accuracy_score

    path = path or "../data/results.csv"
    df, _ = wc.build_dataset(path)                 # all matches (for DC fitting)
    comp = sq.competitive_only(df).reset_index(drop=True)   # test population

    oof_dc, oof_lr, oof_y = [], [], []
    for tr, te, b0, b1 in wf.generate_folds(comp):
        # DC trains on ALL matches before the block (EXPANDING window -- it is
        # data-starved on international football and a long time-decay handles
        # recency; this beat the rolling window in the walk-forward).
        hist = df[df["date"] < b0]
        dc = DixonColes().fit(hist, ref_date=b0)
        P = np.array([[*dc.predict(h, a, n).values()]
                      for h, a, n in zip(te["home_team"], te["away_team"],
                                         te["neutral"])])
        oof_dc.append(P)
        oof_y.append(te["target"].to_numpy())
        # LR on the same fold, for a head-to-head reference
        w = wc.time_decay_weights(tr["date"], wc.DECAY_HALFLIFE_DAYS)
        est = wf._fit(wf.make_lr(), tr[wc.FEATURES].to_numpy(),
                      tr["target"].to_numpy(), w)
        oof_lr.append(wf.predict_proba_full(est, te[wc.FEATURES].to_numpy()))

    DC, LR, Y = np.vstack(oof_dc), np.vstack(oof_lr), np.concatenate(oof_y)
    L = lambda P: log_loss(Y, P, labels=[0, 1, 2])
    A = lambda P: accuracy_score(Y, P.argmax(1))
    dll = lambda P: -np.log(np.clip(P[Y == 1, 1], 1e-15, 1)).mean()  # draw-class ll

    print(f"Test matches: {len(Y):,}\n")
    print(f"  {'model':<14}{'log loss':>9}{'acc':>7}{'draw_ll':>9}")
    print("  " + "-" * 39)
    print(f"  {'LR':<14}{L(LR):>9.4f}{A(LR):>7.3f}{dll(LR):>9.3f}")
    print(f"  {'Dixon-Coles':<14}{L(DC):>9.4f}{A(DC):>7.3f}{dll(DC):>9.3f}")
    for wlr in (0.3, 0.5, 0.7):
        B = wlr * LR + (1 - wlr) * DC
        print(f"  {'blend '+str(wlr)+'LR':<14}{L(B):>9.4f}{A(B):>7.3f}{dll(B):>9.3f}")
    return DC, LR, Y


if __name__ == "__main__":
    import os
    backtest("../data/results.csv" if os.path.exists("../data/results.csv") else None)
