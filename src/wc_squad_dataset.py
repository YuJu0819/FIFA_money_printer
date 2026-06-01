"""
Zero-scrape squad value from ready CSVs (salimt Transfermarkt datalake on Kaggle:
kaggle datasets download -d xfkzujqjvx97n/football-datasets).

It uses two flat tables, joined on player_id:
  * player_market_value          : {player_id, date_unix, value}  -> dated values
                                    (date_unix is actually an ISO date string)
  * player_national_performances : {player_id, team_id, debut, career_state}
                                    -> which national team_id a player belongs to

REAL-DATA NOTES (this dump differs from the idealized schema):
  * There is NO team name and NO usable debut date in the membership table
    (`debut` is 100% empty), and the national team_id resolves in none of the
    team_* tables. So team_id -> country is recovered OFFLINE by build_team_name_map
    (modal player citizenship from player_profiles). resolve_team_names (network)
    is kept only as a fallback; its public API is currently unreliable.

WHAT THIS IS (and isn't):
  It builds each nation's value as the sum, at each date, of its *player pool's*
  point-in-time values -- not the exact matchday XI. "Currently in the pool" is
  approximated as: still actively valued (a market-value update within
  `liveness_months`). The debut gate is applied only where a date exists (it does
  not in this dump). It's a roster-POOL team-strength proxy -- close to
  Transfermarkt's own "national team value" -- good as a feature; it is NOT a
  claim about who started a given match.

Output: long DataFrame [country, date, mv_eur], i.e. exactly what
wc_market_value.attach_market_value() expects. Full offline recipe:
    mv   = load_market_values(".../player_market_value.csv")
    pool = load_national_pool(".../player_national_performances.csv")
    tmap = build_team_name_map(pool, ".../player_profiles.csv")   # offline names
    cs   = build_country_value_series(mv, pool, tmap)
    cs   = align_names(cs)                              # profiles -> martj42 names
    df   = wc_market_value.attach_market_value(df, cs) # leakage-safe as-of join
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# martj42 name -> salimt/profiles citizenship spelling, only where they differ.
# (Verified against player_profiles.citizenship; the salimt dump uses a
# Transfermarkt/German-ish convention for some nations.) Extend as needed.
NAME_ALIAS = {
    # major / mid-tier nations with a different spelling in player_profiles
    "Ivory Coast": "Cote d'Ivoire",
    "South Korea": "Korea, South",
    "North Korea": "Korea, North",
    "China PR": "China",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Curaçao": "Curacao",
    "Hong Kong": "Hongkong",
    "Turkey": "Türkiye",
    "Gambia": "The Gambia",
    "Republic of Ireland": "Ireland",
    "Taiwan": "Chinese Taipei",
    "New Caledonia": "Neukaledonien",
    # small islands / nations (no-op until their pool clears the size threshold)
    "Saint Kitts and Nevis": "St. Kitts & Nevis",
    "Saint Martin": "Saint-Martin",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent & Grenadinen",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "South Sudan": "Southern Sudan",
}


def _to_datetime_unix(s: pd.Series) -> pd.Series:
    """Parse the market-value date column. Despite the name `date_unix`, the
    salimt file actually stores ISO date strings ("2023-12-19"); we still handle
    real epoch (seconds or milliseconds, by magnitude) in case a future dump
    changes format."""
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().gt(10**11).mean() > 0.5 else "s"
        return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def load_market_values(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"value": "value_eur"})
    df["date"] = _to_datetime_unix(df["date_unix"])
    df = df.dropna(subset=["date", "value_eur"])
    return df[["player_id", "date", "value_eur"]].sort_values(["player_id", "date"])


def load_national_pool(path: str) -> pd.DataFrame:
    """player_national_team_performances: real columns are player_id, team_id,
    debut, career_state (NOT team_name/first_game_date). Country name is NOT in
    this file -- resolve team_id -> name separately (resolve_team_names)."""
    df = pd.read_csv(path)
    df["debut"] = pd.to_datetime(df.get("debut"), errors="coerce")
    keep = [c for c in ["player_id", "team_id",
                        "debut", "career_state"] if c in df.columns]
    return df[keep]


def resolve_team_names(team_ids, api_base="https://transfermarkt-api.fly.dev",
                       fetch_json=None) -> dict:
    """Map the ~200 distinct national team_ids -> country name via the felipeall
    API (GET /clubs/{id}/profile -> {"name": ...}). One light, cached call per
    team_id. This is the only network step left, and it's tiny vs per-player
    scraping. Offline alternative: parse team_url for /verein/{id} out of any
    table that carries national-team rows, then join names yourself."""
    import json
    from urllib.request import Request, urlopen
    if fetch_json is None:
        def fetch_json(url):
            req = Request(url, headers={"User-Agent": "research-kaggle/1.0"})
            with urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
    out = {}
    for tid in sorted(set(team_ids)):
        try:
            out[tid] = fetch_json(
                f"{api_base}/clubs/{tid}/profile").get("name", "")
        except Exception as e:
            print(f"  name lookup failed for team_id {tid}: {e}")
    return out


def build_team_name_map(national_pool: pd.DataFrame, player_profiles_path: str,
                        min_purity: float = 0.6, min_players: int = 5) -> dict:
    """OFFLINE national team_id -> country name, no network (preferred over
    resolve_team_names, whose third-party API is unreliable).

    Names each national team_id by the MODAL first-citizenship of the players who
    appear for it (player_profiles.citizenship). National squads are >90% single-
    citizenship, so the mode is the country (verified: Brazil/Germany/... resolve
    at 0.94-1.00 purity). Output names are in the profiles spelling; align_names()
    then maps them to martj42. A team_id is dropped if it has < min_players capped
    players or its modal citizenship share < min_purity (too ambiguous to trust).

    Note: senior and youth teams of one nation both resolve to that nation; that
    is fine -- build_country_value_series dedupes players per country, so a youth
    cap never double-counts, and youth-only players add negligible value.
    """
    pp = pd.read_csv(player_profiles_path, low_memory=False)
    cz = pp["citizenship"].astype(str).str.split("  ").str[0].str.strip()
    pp = pd.DataFrame({"player_id": pp["player_id"], "cz1": cz})
    pp = pp[pp["cz1"].ne("") & pp["cz1"].ne("nan")]
    j = national_pool[["player_id", "team_id"]].merge(
        pp, on="player_id", how="left")
    j = j.dropna(subset=["cz1"])

    out = {}
    for tid, s in j.groupby("team_id")["cz1"]:
        vc = s.value_counts()
        if len(s) >= min_players and (vc.iloc[0] / vc.sum()) >= min_purity:
            out[int(tid)] = vc.index[0]
    return out


def build_country_value_series(
    market_values: pd.DataFrame,
    national_pool: pd.DataFrame,
    team_name_map: dict,
    freq: str = "MS",
    liveness_months: int = 18,
    agg: str = "top_n_mean",
    top_n: int = 30,
) -> pd.DataFrame:
    """Per-country point-in-time squad value on a monthly grid.

    national_pool : player_id, team_id, debut  (from load_national_pool)
    team_name_map : {team_id: country_name}    (from build_team_name_map)
    Liveness is by market-value recency (point-in-time), NOT career_state.

    agg :
      "top_n_mean" (default) -- mean of the `top_n` most valuable LIVE players at
            each date. This is a squad-QUALITY proxy: it tracks the players a
            nation would realistically field and is not inflated by how deep the
            capped-player pool is. Nations with < top_n live players average what
            they have.
      "sum"  -- total value of all live players (pool-DEPTH + quality mixed).

    `mv_eur` holds whichever aggregate was chosen; attach_market_value only takes
    its log and a difference, so the choice of average vs sum is a modelling
    decision, not a units issue.
    """
    pool = national_pool.copy()
    pool["country"] = pool["team_id"].map(team_name_map)
    unmapped = pool["country"].isna().mean()
    if unmapped:
        print(f"  note: {unmapped:.0%} of pool rows have no name for their team_id "
              f"-> dropped. Extend team_name_map if teams you need are missing.")
    pool = pool.dropna(subset=["country"])
    pool = pool.rename(columns={"debut": "first_game_date"})
    # One nation can have several team_ids (senior + U21/U23/...), all resolving
    # to the same country. Keep one row per (country, player) so a player who
    # appears for both senior and youth is summed only once.
    pool = pool.sort_values("first_game_date").drop_duplicates(
        ["country", "player_id"])
    # Normalize EVERY datetime to int64 nanoseconds. (This pandas build can use
    # microsecond resolution, and Timestamp.value is unit-dependent, so mixing
    # them silently breaks comparisons -- force ns everywhere.)
    grid = pd.date_range(market_values["date"].min(),
                         market_values["date"].max(), freq=freq).as_unit("ns")
    grid_i64 = grid.asi8
    live_ns = np.int64(liveness_months) * 30 * 24 * 3600 * 10**9

    mv = market_values.assign(
        _ns=market_values["date"].dt.as_unit("ns").astype("int64"))
    mv_by_player = {pid: (g["_ns"].to_numpy(), g["value_eur"].to_numpy())
                    for pid, g in mv.groupby("player_id")}

    rows = []
    for country, grp in pool.groupby("country"):
        # One row per player: their LIVE as-of value over the grid (NaN = not in
        # the active pool at that date).
        cols = []
        for pid, fg in zip(grp["player_id"], grp["first_game_date"]):
            pv = mv_by_player.get(pid)
            if pv is None:
                continue
            dates_i64, vals = pv
            idx = np.searchsorted(dates_i64, grid_i64,
                                  side="right") - 1  # last <= grid
            valid = idx >= 0
            asof_val = np.where(valid, vals[idx.clip(0)], np.nan)
            last_update = np.where(valid, dates_i64[idx.clip(0)], 0)
            live = valid & ((grid_i64 - last_update) <=
                            live_ns)       # still active
            if pd.notna(fg):
                live &= grid_i64 >= pd.Timestamp(
                    fg).as_unit("ns").value  # debuted
            cols.append(np.where(live, asof_val, np.nan))
        if not cols:
            continue
        M = np.vstack(cols)                       # (n_players, n_grid)

        if agg == "sum":
            aggv = np.nansum(M, axis=0)           # 0 where no live player
        elif agg == "top_n_mean":
            # mean of the top_n largest live values per column, no-NaN-warning.
            Mf = np.where(np.isnan(M), -np.inf, M)
            k = min(top_n, M.shape[0])
            top = np.sort(Mf, axis=0)[::-1][:k]   # (k, n_grid), -inf = padding
            mask = np.isfinite(top)
            cnt = mask.sum(axis=0)
            ssum = np.where(mask, top, 0.0).sum(axis=0)
            aggv = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
        else:
            raise ValueError(f"unknown agg={agg!r}; use 'top_n_mean' or 'sum'")

        keep = aggv > 0
        for d, v in zip(grid[keep], aggv[keep]):
            rows.append((country, d, float(v)))
    return pd.DataFrame(rows, columns=["country", "date", "mv_eur"])


def align_names(country_series: pd.DataFrame, alias: dict | None = None
                ) -> pd.DataFrame:
    """Rename profiles/salimt country names to martj42 names (reverse of
    NAME_ALIAS)."""
    alias = alias or NAME_ALIAS
    rev = {v: k for k, v in alias.items()}
    out = country_series.copy()
    out["country"] = out["country"].map(lambda c: rev.get(c, c))
    return out


# ---------------------------------------------------------------------------
# Recipe runner: build the leakage-safe country value series from the salimt
# CSVs and write it to disk.  Run:  python wc_squad_dataset.py [DATA_DIR] [OUT]
# ---------------------------------------------------------------------------
def build_series_from_dir(data_dir: str = "../data",
                          freq: str = "MS", liveness_months: int = 18,
                          min_players: int = 5, min_purity: float = 0.6,
                          agg: str = "top_n_mean", top_n: int = 30
                          ) -> pd.DataFrame:
    """End-to-end: load the three salimt CSVs, resolve team names offline, build
    the per-country point-in-time value series, and align to martj42 names."""
    import os
    mv_path = os.path.join(data_dir, "player_market_value",
                           "player_market_value.csv")
    np_path = os.path.join(data_dir, "player_national_performances",
                           "player_national_performances.csv")
    pp_path = os.path.join(data_dir, "player_profiles", "player_profiles.csv")

    mv = load_market_values(mv_path)
    pool = load_national_pool(np_path)
    tmap = build_team_name_map(pool, pp_path,
                               min_purity=min_purity, min_players=min_players)
    print(f"market values: {len(mv):,} rows | pool: {len(pool):,} rows | "
          f"team_ids named: {len(tmap)} ({len(set(tmap.values()))} countries)")
    cs = build_country_value_series(mv, pool, tmap, freq=freq,
                                    liveness_months=liveness_months,
                                    agg=agg, top_n=top_n)
    cs = align_names(cs)
    label = f"top-{top_n} mean" if agg == "top_n_mean" else agg
    print(f"country value series ({label}): {len(cs):,} rows, "
          f"{cs['country'].nunique()} countries, "
          f"{cs['date'].min().date()}..{cs['date'].max().date()}")
    return cs


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    out = sys.argv[2] if len(sys.argv) > 2 else "country_market_value.csv"
    cs = build_series_from_dir(data_dir)
    cs.to_csv(out, index=False)
    print(f"\nwrote -> {out}")
    print("\nUse it in the walk-forward with:")
    print("    import wc_market_value as mv, walk_forward as wf")
    print(f"    cs = mv.load_country_series_from_csv('{out}')"
          "  # has country,date,mv_eur")
    print("    wf.walk_forward('../data/results.csv', country_series=cs)")
    # peek: top nations by latest squad value (top-30 mean -> per-player, in EUR m)
    latest = (cs.sort_values("date").groupby("country").tail(1)
              .sort_values("mv_eur", ascending=False).head(10))
    print("\nTop 10 nations by latest squad value (EUR m, avg of top-30):")
    for _, r in latest.iterrows():
        print(f"  {r['country']:<16} {r['mv_eur']/1e6:6.1f}")
