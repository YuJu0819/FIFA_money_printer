"""
Per-match squad valuation for competitive internationals (option 1, extended
to qualifiers). Builds on wc_market_value.py (reuses value_asof + the fetcher).

WHY TWO INPUT MODES:
  * Tournaments publish one squad list -> "editions" mode is enough.
  * Qualifiers have a different call-up per match and NO single squad list, so
    real qualifier rosters need per-match lineup data -> "lineups" mode.
  martj42 carries neither lineups nor squads, so you supply one of these:

  editions : long DataFrame [country, start, end, player_id]
             one row per player per call-up window. A "World Cup 2022" edition
             is one window; a "WC2026 qualifying campaign" is another (use the
             broad pool called up across the campaign). Coarser but light.
  lineups  : long DataFrame [date, team, player_id]
             one row per player per match. Accurate per-qualifier rosters, but
             you need the data (Transfermarkt national-team match pages, etc.).

LEAKAGE NOTES:
  * Player VALUE is always taken as-of the match date via value_asof, i.e. the
    last revaluation on/before kickoff -> no value leakage.
  * Roster IDENTITY: editions mode is leakage-free (the window is known ahead).
    lineups mode uses the REALIZED lineup, which is mild optimism (you wouldn't
    know the exact XI pre-kickoff). For a strictly pre-match version, build the
    roster from players capped in the team's PRIOR N matches instead -- see
    recent_caps_squads().
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import wc_market_value as mvmod   # value_asof, fetch_player_value_history

# Substring match on the martj42 `tournament` column. Drop these from MODELLING
# (they stay in the Elo/form history, which is computed earlier over everything).
DROP_FROM_MODELLING = ["friendly"]


# ---------------------------------------------------------------------------
# Competition filtering
# ---------------------------------------------------------------------------
def is_competitive(tournament: str) -> bool:
    t = str(tournament).lower()
    return not any(k in t for k in DROP_FROM_MODELLING)


def competitive_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep World Cup, continental championships, ALL qualifiers, Nations League,
    etc.; drop friendlies. Apply to the modelling window, NOT before Elo/form."""
    return df[df["tournament"].map(is_competitive)].copy()


# ---------------------------------------------------------------------------
# Per-match squad value: EDITIONS mode (tournaments + qualifying campaigns)
# ---------------------------------------------------------------------------
def squad_value_from_editions(df: pd.DataFrame, editions: pd.DataFrame,
                              histories: dict):
    """For each match, squad = players whose [start,end] window for that country
    contains the match date; value summed as-of the date. Returns (home, away)
    Series aligned to df.index."""
    ed = editions.copy()
    ed["start"] = pd.to_datetime(ed["start"])
    ed["end"] = pd.to_datetime(ed["end"])
    # index by country for speed
    by_country = {c: g for c, g in ed.groupby("country")}

    def value(team, date):
        g = by_country.get(team)
        if g is None:
            return np.nan
        sel = g[(g["start"] <= date) & (g["end"] >= date)]
        vals = [mvmod.value_asof(histories[p], date)
                for p in sel["player_id"] if p in histories]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan

    home = df.apply(lambda r: value(r["home_team"], r["date"]), axis=1)
    away = df.apply(lambda r: value(r["away_team"], r["date"]), axis=1)
    return home, away


# ---------------------------------------------------------------------------
# Per-match squad value: LINEUPS mode (accurate per-qualifier rosters)
# ---------------------------------------------------------------------------
def squad_value_from_lineups(df: pd.DataFrame, lineups: pd.DataFrame,
                             histories: dict):
    lu = lineups.copy()
    lu["date"] = pd.to_datetime(lu["date"])
    grp = lu.groupby(["date", "team"])["player_id"].apply(list)

    def value(team, date):
        pids = grp.get((date, team), [])
        vals = [mvmod.value_asof(histories[p], date)
                for p in pids if p in histories]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan

    home = df.apply(lambda r: value(r["home_team"], r["date"]), axis=1)
    away = df.apply(lambda r: value(r["away_team"], r["date"]), axis=1)
    return home, away


def recent_caps_squads(df: pd.DataFrame, lineups: pd.DataFrame, n_matches: int = 10):
    """Build a STRICTLY pre-match roster per (team,date): the distinct players the
    team capped in its previous `n_matches` matches. Use the result as `lineups`
    input to squad_value_from_lineups for a leakage-free roster identity."""
    lu = lineups.copy()
    lu["date"] = pd.to_datetime(lu["date"])
    match_players = lu.groupby(["team", "date"])["player_id"].apply(set).reset_index()
    rows = []
    for team, g in match_players.groupby("team"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(len(g)):
            prior = g.iloc[max(0, i - n_matches):i]
            pool = set().union(*prior["player_id"]) if len(prior) else set()
            for p in pool:
                rows.append((g.loc[i, "date"], team, p))
    return pd.DataFrame(rows, columns=["date", "team", "player_id"])


# ---------------------------------------------------------------------------
# Finalize -> the columns the walk-forward consumes
# ---------------------------------------------------------------------------
EXTRA_FEATURES = ["mv_log_diff", "mv_missing"]


def finalize(df: pd.DataFrame, mv_home: pd.Series, mv_away: pd.Series
             ) -> pd.DataFrame:
    out = df.copy()
    out["mv_home"] = pd.Series(mv_home, index=df.index)
    out["mv_away"] = pd.Series(mv_away, index=df.index)
    out["mv_log_home"] = np.log10(out["mv_home"])
    out["mv_log_away"] = np.log10(out["mv_away"])
    out["mv_log_diff"] = out["mv_log_home"] - out["mv_log_away"]
    out["mv_missing"] = out["mv_log_diff"].isna().astype(int)
    return out
