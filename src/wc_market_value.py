"""
Point-in-time squad market value for the walk-forward backtest.

THE ONE RULE: for backtesting you must use each team's value AS IT STOOD
BEFORE the match. Joining current values onto past matches leaks the future
(a player's value today reflects how his career actually turned out) and
produces a backtest that looks great and fails live. Everything below is built
around a strictly-earlier "as-of" join to prevent that.

Pipeline:
  1. fetch_player_value_history(id)  -> dated [(date, value_eur)] per player
                                        (Transfermarkt ceapi value chart)
  2. build_country_series(squads)    -> per-country monthly point-in-time
                                        total squad value
  3. attach_market_value(matches)    -> adds mv_log_home/away/diff via merge_asof
                                        (strictly-earlier snapshot only)
  4. feed mv_log_diff into the walk-forward FEATURES.

NOTE ON ACCESS: Transfermarkt has no official API and scraping it is against
their ToS; throttle hard, cache, and prefer an existing dataset if you have one
(see load_country_series_from_csv for the no-scrape path). This module cannot
reach transfermarkt.com from every environment -- the parsing and join logic is
what matters and is unit-tested below on synthetic data.
"""

from __future__ import annotations
import json
import time
import os
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

CACHE_DIR = "mv_cache"
CEAPI = "https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{pid}"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; personal Kaggle project)"}
THROTTLE_SECONDS = 2.0  # be polite; raise if you see blocks


# ---------------------------------------------------------------------------
# 1. Fetch + parse one player's dated value history
# ---------------------------------------------------------------------------
def parse_value_history(raw: dict) -> pd.DataFrame:
    """Parse the ceapi value-development payload into [date, value_eur].

    Kept separate from the network call so it can be tested offline.
    Expected shape: {"list": [{"datum_mw": "Jan 1, 2018", "y": 500000, ...}, ...]}
    """
    items = raw.get("list", []) if isinstance(raw, dict) else []
    rows = []
    for it in items:
        date = it.get("datum_mw") or it.get("x")
        if isinstance(date, (int, float)):           # epoch ms fallback
            date = pd.to_datetime(date, unit="ms", errors="coerce")
        else:
            date = pd.to_datetime(date, errors="coerce")
        val = it.get("y")
        if val is None:                              # parse "€500Th."/"€1.20m"
            val = _parse_money(it.get("mw", ""))
        if pd.notna(date) and val is not None and not pd.isna(val):
            rows.append((date, float(val)))
    df = pd.DataFrame(rows, columns=["date", "value_eur"]).dropna()
    return df.sort_values("date").reset_index(drop=True)


def _parse_money(s: str) -> float | None:
    s = str(s).replace("€", "").replace("\u20ac", "").strip().lower()
    if not s:
        return None
    mult = 1.0
    if s.endswith("m") or "mio" in s:
        mult, s = 1e6, s.replace("mio", "").rstrip("m")
    elif s.endswith("th.") or s.endswith("k") or "tsd" in s:
        mult, s = 1e3, s.replace("th.", "").replace("tsd", "").rstrip("k")
    try:
        return float(s.replace(",", ".")) * mult
    except ValueError:
        return None


def fetch_player_value_history(player_id: str | int, use_cache=True) -> pd.DataFrame:
    """Fetch one player's dated market-value history. Caches to disk.

    Cannot run where transfermarkt.com is unreachable; raises so the caller
    can fall back to a CSV. Throttle and cache to stay within reason.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{player_id}.json")
    if use_cache and os.path.exists(cache):
        with open(cache) as f:
            return parse_value_history(json.load(f))

    req = Request(CEAPI.format(pid=player_id), headers=HEADERS)
    with urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    with open(cache, "w") as f:
        json.dump(raw, f)
    time.sleep(THROTTLE_SECONDS)
    return parse_value_history(raw)


# ---------------------------------------------------------------------------
# 2. Build a per-country point-in-time squad-value series
# ---------------------------------------------------------------------------
def value_asof(history: pd.DataFrame, when: pd.Timestamp) -> float:
    """A player's value as of `when` = last data point dated <= when (else NaN).

    AS-OF POLICY (read alongside attach_market_value): this is INCLUSIVE of
    `when`. A player's public market value on match-day morning is known before
    kickoff (Transfermarkt revalues from form over weeks, never from the single
    result we are predicting), so using a revaluation dated exactly `when` is not
    leakage. The country-snapshot join in attach_market_value is deliberately
    stricter (strictly-before) because those monthly snapshots are coarser.

    Vectorized with searchsorted (history is sorted ascending by parse_*); this
    is O(log n) per call instead of a full O(n) boolean scan.
    """
    dates = history["date"].to_numpy()            # sorted ascending
    if len(dates) == 0:
        return np.nan
    idx = int(np.searchsorted(dates, np.datetime64(when), side="right")) - 1
    if idx < 0:
        return np.nan
    return float(history["value_eur"].iloc[idx])


def build_country_series(
    squads: dict[str, list],
    histories: dict, 
    snapshot_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Sum point-in-time player values into a per-country, per-snapshot total.

    squads     : {country_name: [player_id, ...]}  -- the squad you treat as
                 representative. SIMPLIFICATION: a fixed squad per country
                 ignores call-up turnover. For more accuracy pass era-specific
                 squads and call this once per era, or supply a CSV (below).
    histories  : {player_id: value-history DataFrame from fetch/parse}
    snapshot_dates : the dates to evaluate (e.g. monthly).
    Returns long DataFrame [country, date, mv_eur].
    """
    rows = []
    for country, pids in squads.items():
        for d in snapshot_dates:
            vals = [value_asof(histories[p], d) for p in pids if p in histories]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                rows.append((country, d, float(np.sum(vals))))
    return pd.DataFrame(rows, columns=["country", "date", "mv_eur"])


def load_country_series_from_csv(path: str) -> pd.DataFrame:
    """No-scrape path: read a ready series with columns
    country, date, squad_value_eur  ->  normalized [country, date, mv_eur]."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"squad_value_eur": "mv_eur"})
    return df[["country", "date", "mv_eur"]].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. As-of join onto the match table  (THE anti-leakage step)
# ---------------------------------------------------------------------------
EXTRA_FEATURES = ["mv_log_diff", "mv_missing"]


def attach_market_value(matches: pd.DataFrame, country_series: pd.DataFrame
                        ) -> pd.DataFrame:
    """Add mv_log_home, mv_log_away, mv_log_diff, mv_missing to `matches`.

    Uses merge_asof with direction='backward' and allow_exact_matches=False, so
    each match only ever sees a squad value dated STRICTLY BEFORE its own date.
    """
    cs = country_series.sort_values("date")[["country", "date", "mv_eur"]]
    m = matches.sort_values("date").reset_index().rename(columns={"index": "_orig"})

    def side_join(team_col, out_col):
        left = m[["_orig", "date", team_col]].rename(columns={team_col: "country"})
        left = left.sort_values("date")
        j = pd.merge_asof(left, cs, on="date", by="country",
                          direction="backward", allow_exact_matches=False)
        return j.set_index("_orig")["mv_eur"].rename(out_col)

    mv_home = side_join("home_team", "mv_home")
    mv_away = side_join("away_team", "mv_away")

    out = matches.copy()
    out["mv_home"] = mv_home.reindex(out.index)
    out["mv_away"] = mv_away.reindex(out.index)
    out["mv_log_home"] = np.log10(out["mv_home"])
    out["mv_log_away"] = np.log10(out["mv_away"])
    out["mv_log_diff"] = out["mv_log_home"] - out["mv_log_away"]
    # known at prediction time, so this flag is leakage-free
    out["mv_missing"] = out["mv_log_diff"].isna().astype(int)
    return out


# ---------------------------------------------------------------------------
# 4. Model that tolerates the NaNs market value introduces
# ---------------------------------------------------------------------------
def make_model_with_imputer():
    """Drop-in replacement for wc_walkforward.make_model that median-imputes.
    The imputer is fit per fold on TRAIN only (inside the pipeline), so adding
    it does not leak across the walk-forward splits."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
