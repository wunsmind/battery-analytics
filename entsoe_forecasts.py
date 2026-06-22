"""ENTSO-E day-ahead *forecast* ingestion: wind/solar generation + load.

Where `entsoe_ingest` pulls the realised day-ahead *price*, this pulls the
exogenous drivers a trader actually has **at the day-ahead gate**: the published
day-ahead forecasts of wind+solar generation and of total load, per bidding zone.

Why this is better than the Open-Meteo point weather we already fold in:
  - **Gate-aligned, no optimism.** The weather backtest uses *archived actual*
    weather as a proxy for the delivery-day forecast (a mild look-ahead). These
    ENTSO-E series are the genuine forecast *as published before the gate*, so a
    backtest on them is honest about what was knowable.
  - **System-level, not a proxy point.** Total zonal wind generation (MW) is the
    quantity that actually clears the market, vs wind speed at one coordinate.
  - **Cross-border.** SE_4 prices are driven as much by German/Danish wind
    (via the Baltic Cable / Kontek / Öresund interconnectors) as by Swedish wind.
    ENTSO-E publishes DE_LU / DK_1 / DK_2 wind forecasts; Open-Meteo would need a
    hand-picked grid of foreign points to approximate the same signal.

Availability is uneven and handled gracefully (skip-and-continue):
  - DE_LU / DK_1 / DK_2: wind (on+offshore) + solar, deep history, 15-min (DE) /
    hourly (DK).
  - SE zones: load forecast for all; wind only `Wind Onshore` for some (SE_3),
    and `query_wind_and_solar_forecast` raises NoMatchingDataError for others.

Module is `entsoe_forecasts` (mirrors `entsoe_ingest`): fetch + record shaping
here, the CLI/backfill driver in `fetch_forecasts.py`.
"""

from __future__ import annotations

import pandas as pd
from entsoe import EntsoePandasClient

from entsoe_ingest import _resolution_tag

# Zones we know publish useful day-ahead forecasts. The northern SE zones add
# little (almost no wind, weakly coupled), so the defaults target SE_3/SE_4 and
# their interconnector neighbours.
WIND_SOLAR_ZONES = ["DE_LU", "DK_1", "DK_2", "SE_3", "SE_4"]
LOAD_ZONES = ["SE_3", "SE_4"]

# ENTSO-E generation/load forecasts are mandated from 2015, but coverage by zone
# only firms up around 2016; callers chunk and skip gaps, so this is just a floor.
EARLIEST = "2016-01-01"


def _median_gap_minutes(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 60.0
    return float(pd.Series(idx).diff().dropna().dt.total_seconds().median() / 60.0)


def fetch_wind_solar(token: str, zone: str, start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
    """Day-ahead wind (on+offshore summed) and solar generation forecast, in MW.

    Returns a frame with whatever of {wind, solar} the zone publishes; empty if
    none. Onshore/offshore are summed into a single `wind` (their split isn't a
    price driver — total injected wind is).
    """
    client = EntsoePandasClient(api_key=token)
    df = client.query_wind_and_solar_forecast(zone, start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    wind_cols = [c for c in df.columns if "Wind" in c]
    if wind_cols:
        out["wind"] = df[wind_cols].sum(axis=1, min_count=1)
    if "Solar" in df.columns:
        out["solar"] = df["Solar"]
    out.index.name = "starts_at"
    return out


def fetch_load(token: str, zone: str, start: pd.Timestamp,
               end: pd.Timestamp) -> pd.DataFrame:
    """Day-ahead total load forecast, in MW. Returns a frame with a `load` column."""
    client = EntsoePandasClient(api_key=token)
    df = client.query_load_forecast(zone, start=start, end=end)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # entsoe-py returns a single-column DataFrame ('Forecasted Load'); be lenient.
    col = df["Forecasted Load"] if hasattr(df, "columns") else df
    out = pd.DataFrame({"load": col})
    out.index.name = "starts_at"
    return out


def to_records(zone: str, frame: pd.DataFrame, source: str = "entsoe") -> list[dict]:
    """Convert a wide forecast frame (columns = kinds) to long upsert rows.

    Each (zone, starts_at, kind) becomes one row; resolution is tagged from the
    series' median step so a frame spanning an hourly→15-min switch is labelled by
    its dominant cadence (forecasts are uniform within a fetch window).
    """
    if frame is None or frame.empty:
        return []
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    res = _resolution_tag(_median_gap_minutes(frame.index))
    out = []
    for ts, row in frame.iterrows():
        iso = ts.isoformat()
        for kind in frame.columns:
            val = row[kind]
            if pd.isna(val):
                continue
            out.append({"zone": zone, "starts_at": iso, "kind": kind,
                        "resolution": res, "value": float(val), "source": source})
    return out
