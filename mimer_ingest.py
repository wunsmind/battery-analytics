"""Svenska kraftnät (Mimer) ancillary-service price ingestion.

Mimer (https://mimer.svk.se) is SvK's public market portal; it publishes cleared
volumes and marginal prices for the Nordic reserves — FCR-N, FCR-D up/down,
(and aFRR / mFRR on separate controllers). This module pulls FCR via the public
CSV export, the half of the revenue stack that `optimizer.markets` models but had
no price feed behind it.

No API key — the endpoint is open. Data is internal market data (reproducible from
a public source), so it lives in the git-ignored market.db like ENTSO-E history.

Verified live-response facts (2026-06, FCR endpoint):
  - GET /PrimaryRegulation/DownloadText?periodFrom=&periodTo=&auctionTypeId=&productTypeId=
  - periodFrom/periodTo are MM/DD/YYYY HH:MM:SS in *Europe/Stockholm* local time.
  - CSV is UTF-8 BOM, ';'-delimited, decimal comma, trailing ';' per row.
  - `Datum` is Europe/Stockholm local time, hourly.
  - One system-wide marginal price column per product (EUR/MW); the per-zone
    columns (`SE3 FCRN`, ...) are cleared *volumes* in MW, not prices.

aFRR / mFRR live on sibling controllers (e.g. /Afrr, /Mfrr) with the same
DownloadText pattern but different columns — left as a TODO until SE aFRR opens
(PICASSO) since it carries no battery revenue for SE today.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

BASE_URL = "https://mimer.svk.se"
FCR_PATH = "/PrimaryRegulation/DownloadText"
MIMER_TZ = "Europe/Stockholm"
CURRENCY = "EUR"
SOURCE = "svk_mimer"

# Mimer publishes FCR for the Swedish zones plus the cooperating DK2 zone.
ZONE_COLUMN_SUFFIX = {        # our zone id -> Mimer per-zone volume column token
    "SE_1": "SE1",
    "SE_2": "SE2",
    "SE_3": "SE3",
    "SE_4": "SE4",
    "DK_2": "DK2",
}

# Each FCR product: optimizer.markets id -> (price column, per-zone volume token).
# Volume column is f"{zone_token} {vol_token}", e.g. "SE3 FCRN", "SE1 FCRD upp".
FCR_PRODUCTS = {
    "fcr_n":      ("FCR-N Pris (EUR/MW)",     "FCRN"),
    "fcr_d_up":   ("FCR-D upp Pris (EUR/MW)", "FCRD upp"),
    "fcr_d_down": ("FCR-D ned Pris (EUR/MW)", "FCRD ned"),
}


def _resolution_tag(minutes: float) -> str:
    return "QUARTER_HOURLY" if 10 <= minutes <= 20 else "HOURLY"


def fetch_fcr_csv(
    period_from: pd.Timestamp,
    period_to: pd.Timestamp,
    auction_type_id: int = 1,    # 1 = "Totalt" (combined D-1/D-2 auctions)
    product_type_id: int = 0,    # 0 = all FCR products
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> str:
    """Raw FCR CSV text for [period_from, period_to). Times are sent as Stockholm
    local wall-clock (tz-aware inputs are converted, naive ones assumed Stockholm)."""
    fmt = "%m/%d/%Y %H:%M:%S"
    params = {
        "periodFrom": _to_stockholm_wallclock(period_from).strftime(fmt),
        "periodTo": _to_stockholm_wallclock(period_to).strftime(fmt),
        "auctionTypeId": auction_type_id,
        "productTypeId": product_type_id,
    }
    get = (session or requests).get
    resp = get(BASE_URL + FCR_PATH, params=params, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8-sig"   # strip BOM
    return resp.text


def parse_fcr_csv(text: str) -> list[dict]:
    """Parse Mimer FCR CSV into upsert-ready reserve_prices rows (capacity only).

    Emits one row per (product, zone, hour): the shared marginal price plus that
    zone's cleared volume. Zones/products absent from the CSV are skipped."""
    text = text.lstrip("﻿")
    if not text.strip():
        return []
    df = pd.read_csv(io.StringIO(text), sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")          # trailing ';' -> empty column
    if "Datum" not in df.columns or df.empty:
        return []

    # Mimer appends a "Summa" (total) footer row — keep only real timestamps.
    dt = pd.to_datetime(df["Datum"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df[dt.notna()].reset_index(drop=True)
    dt = dt[dt.notna()].reset_index(drop=True)
    if df.empty:
        return []

    idx = _localize_stockholm(dt)
    res = _resolution_series(idx)

    rows: list[dict] = []
    for product_id, (price_col, vol_token) in FCR_PRODUCTS.items():
        if price_col not in df.columns:
            continue
        price = pd.to_numeric(df[price_col], errors="coerce")
        for zone, ztok in ZONE_COLUMN_SUFFIX.items():
            vol_col = f"{ztok} {vol_token}"
            if vol_col not in df.columns:
                continue
            vol = pd.to_numeric(df[vol_col], errors="coerce")
            for ts, p, v, r in zip(idx, price, vol, res):
                if pd.isna(p) and pd.isna(v):
                    continue
                rows.append({
                    "product_id": product_id,
                    "zone": zone,
                    "starts_at": ts.isoformat(),
                    "resolution": r,
                    "price_type": "capacity",
                    "price": None if pd.isna(p) else float(p),
                    "volume_mw": None if pd.isna(v) else float(v),
                    "currency": CURRENCY,
                    "source": SOURCE,
                })
    return rows


def iter_month_chunks(start: pd.Timestamp, end: pd.Timestamp, months: int = 3):
    """Yield (chunk_start, chunk_end) windows of <=`months`, for resumable backfills."""
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=months), end)
        yield cur, nxt
        cur = nxt


def _to_stockholm_wallclock(ts: pd.Timestamp) -> pd.Timestamp:
    """Stockholm-local naive timestamp (the wall-clock Mimer's query expects)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(MIMER_TZ)
    else:
        ts = ts.tz_localize(MIMER_TZ)
    return ts.tz_localize(None)


def _localize_stockholm(naive: pd.Series) -> pd.DatetimeIndex:
    """Localize naive Stockholm timestamps to tz-aware, tolerating the DST fold."""
    idx = pd.DatetimeIndex(naive)
    try:
        return idx.tz_localize(MIMER_TZ, ambiguous="infer",
                               nonexistent="shift_forward")
    except ValueError:
        # DST fold that 'infer' can't resolve (e.g. a non-contiguous window) —
        # fall back to treating the ambiguous hour as DST (ambiguous=True).
        return idx.tz_localize(MIMER_TZ, ambiguous=True,
                               nonexistent="shift_forward")


def _resolution_series(idx: pd.DatetimeIndex) -> list[str]:
    """Per-row resolution tag from the gap to the next point (handles a mid-window
    hourly->15-min switch, mirroring entsoe_ingest)."""
    if len(idx) == 0:
        return []
    gaps = pd.Series(idx).diff().shift(-1).dt.total_seconds().div(60)
    if len(gaps) > 1:
        gaps.iloc[-1] = gaps.iloc[-2]
    else:
        gaps.iloc[-1] = 60.0
    return [_resolution_tag(float(g) if pd.notna(g) else 60.0) for g in gaps]
