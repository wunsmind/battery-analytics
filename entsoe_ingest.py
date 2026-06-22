"""ENTSO-E Transparency Platform day-ahead price ingestion.

Pulls wholesale day-ahead prices (currency/MWh) per bidding zone — the "pure"
market price (≈ Nord Pool), distinct from Tibber's home consumer price. Deep
history back to ~2015; quarter-hourly since 2025-10-01, hourly before.

Module is `entsoe_ingest` (not `entsoe`) so it doesn't shadow the entsoe-py lib.
"""

from __future__ import annotations

import pandas as pd
from entsoe import EntsoePandasClient

# Swedish bidding zones (entsoe-py Area codes). SE_3/SE_4 carry most commercial
# BESS activity; SE_1/SE_2 are the northern surplus zones.
SWEDISH_ZONES = ["SE_1", "SE_2", "SE_3", "SE_4"]

# ENTSO-E mandatory data starts 2015-01-05.
EARLIEST = "2015-01-05"


def _resolution_tag(minutes: float) -> str:
    return "QUARTER_HOURLY" if 10 <= minutes <= 20 else "HOURLY"


def fetch_day_ahead(token: str, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Day-ahead prices for one zone over [start, end). Series in currency/MWh."""
    client = EntsoePandasClient(api_key=token)
    return client.query_day_ahead_prices(zone, start=start, end=end)


def to_records(zone: str, series: pd.Series, currency: str = "EUR",
               source: str = "entsoe") -> list[dict]:
    """Convert a price series to upsert-ready rows, tagging each row's resolution.

    Resolution is derived per-row from the gap to the next point, so a window that
    spans the 2025-10-01 hourly→15-min switch is labelled correctly.
    """
    if series is None or series.empty:
        return []
    series = series[~series.index.duplicated(keep="last")].sort_index()
    idx = series.index
    # gap to the *next* point (the interval this price covers), in minutes
    gap_next = pd.Series(idx, index=range(len(idx))).diff().shift(-1).dt.total_seconds().div(60)
    if len(gap_next) > 1:
        gap_next.iloc[-1] = gap_next.iloc[-2]
    else:
        gap_next.iloc[-1] = 60.0
    out = []
    for ts, val, mins in zip(idx, series.to_numpy(), gap_next.to_numpy()):
        if pd.isna(val):
            continue
        out.append({
            "zone": zone,
            "starts_at": ts.isoformat(),
            "resolution": _resolution_tag(float(mins)),
            "price": float(val),
            "currency": currency,
            "source": source,
        })
    return out


def iter_year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end) spanning [start, end] in <=1-year windows.

    Chunking keeps each ENTSO-E request bounded and makes backfills resumable
    (storage is idempotent, so re-running only refreshes)."""
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        yield cur, nxt
        cur = nxt
