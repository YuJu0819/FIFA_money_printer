"""
Causal club-chemistry feature.

Reconstruct each player's club SPELLS from dated transfers (transfer_history),
find the EARLIEST date each pair of players were club teammates (overlapping
spells at the same club), then score a nation's squad by how many of its top-N
most-valuable players had already played together BEFORE a given date.

Backtest-safe: only partnerships formed strictly before a match count, so there
is no leakage (the static teammate table could not give this -- it has no dates).

VERDICT (tested): chemistry is orthogonal to strength (unlike momentum/H2H), but
the causal feature does NOT predict outcomes -- corr(chem_diff, model residual) =
-0.007, and base+chem_diff is slightly worse (0.8466 -> 0.8472). The orthogonal
part is a "small-nation, few-source-clubs" artifact, not a winning signal. Kept
as a reference tool; not wired into the model.

    spells = load_club_spells("../data/transfer_history/...csv", keep_players=pool_ids)
    onset  = build_teammate_onset(spells)          # {(a,b): earliest_ns}
    cs     = build_chem_series(SquadValuer(...), onset)   # [country, date, chem]
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# to_team values that are not real clubs (end-of-career etc.)
NON_CLUBS = {"Retired", "Without Club", "Career break", "Unknown", "---",
             "Career Break", "Ban", "Suspended"}


def load_club_spells(path: str, keep_players=None) -> pd.DataFrame:
    """Per-player club spells [player_id, club, start, end] from transfers: a
    player joins to_team on transfer_date and stays until the next transfer."""
    th = pd.read_csv(path, usecols=["player_id", "transfer_date", "to_team_id",
                                    "to_team_name"], low_memory=False)
    th["transfer_date"] = pd.to_datetime(th["transfer_date"], errors="coerce")
    th = th.dropna(subset=["transfer_date", "to_team_id"])
    if keep_players is not None:
        th = th[th["player_id"].isin(keep_players)]
    th = th.sort_values(["player_id", "transfer_date"])
    th["start"] = th["transfer_date"]
    th["end"] = th.groupby("player_id")["transfer_date"].shift(-1)
    th["end"] = th["end"].fillna(pd.Timestamp("2100-01-01"))
    th = th[~th["to_team_name"].isin(NON_CLUBS)]
    return th.rename(columns={"to_team_id": "club"})[
        ["player_id", "club", "start", "end"]]


def build_teammate_onset(spells: pd.DataFrame) -> dict:
    """{(lo_id, hi_id): earliest_ns} -- first time each pair overlapped at a club.
    Vectorized per club (broadcast the spell intervals)."""
    s = spells.copy()
    # force NANOseconds (astype int64 alone yields us in pandas 2.x -> 1000x off
    # vs pd.Timestamp(d).value, which is always ns, in build_chem_series)
    s["s_ns"] = s["start"].dt.as_unit("ns").astype("int64")
    s["e_ns"] = s["end"].dt.as_unit("ns").astype("int64")
    onset: dict = {}
    for club, g in s.groupby("club"):
        pid = g["player_id"].to_numpy()
        st = g["s_ns"].to_numpy()
        en = g["e_ns"].to_numpy()
        n = len(pid)
        if n < 2:
            continue
        # overlap_{ij} = max(st_i,st_j) < min(en_i,en_j)
        lo = np.maximum(st[:, None], st[None, :])
        hi = np.minimum(en[:, None], en[None, :])
        ov = lo < hi
        iu = np.triu_indices(n, k=1)
        mask = ov[iu]
        for i, j, o in zip(pid[iu[0]][mask], pid[iu[1]][mask], lo[iu][mask]):
            if i == j:
                continue
            key = (int(i), int(j)) if i < j else (int(j), int(i))
            if key not in onset or o < onset[key]:
                onset[key] = int(o)
    return onset


def build_chem_series(valuer, onset: dict, freq: str = "MS",
                      top_n: int = 30, start="2004-01-01") -> pd.DataFrame:
    """Per-country, per-date chemistry = fraction of top-N squad PAIRS who had
    played together before that date. Output [country, date, chem] in the
    profiles spelling (align_names afterwards)."""
    from itertools import combinations
    grid = pd.date_range(start, pd.Timestamp.today().normalize(), freq=freq)
    rows = []
    for country in valuer.players_by_country:
        for d in grid:
            sq = valuer.squad(country, d)[:top_n]
            ids = [int(i) for i, _, _ in sq]
            if len(ids) < 10:
                continue
            dns = pd.Timestamp(d).value
            npair = conn = 0
            for a, b in combinations(ids, 2):
                npair += 1
                key = (a, b) if a < b else (b, a)
                o = onset.get(key)
                if o is not None and o < dns:
                    conn += 1
            if npair:
                rows.append((country, d, conn / npair))
    return pd.DataFrame(rows, columns=["country", "date", "chem"])


if __name__ == "__main__":
    import wc_squad_dataset as sd
    pool = sd.load_national_pool("../data/player_national_performances/"
                                 "player_national_performances.csv")
    pool_ids = set(pool["player_id"].unique())
    print(f"national-pool players: {len(pool_ids):,}")
    spells = load_club_spells("../data/transfer_history/transfer_history.csv",
                              keep_players=pool_ids)
    print(f"club spells (pool players): {len(spells):,} | "
          f"clubs: {spells['club'].nunique():,}")
    onset = build_teammate_onset(spells)
    print(f"teammate pairs with a dated onset: {len(onset):,}")
    od = pd.to_datetime(pd.Series(list(onset.values())))
    print(f"onset date range: {od.min().date()} .. {od.max().date()} "
          f"| median {od.median().date()}")
