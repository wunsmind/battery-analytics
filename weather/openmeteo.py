"""Open-Meteo weather ingestion for Swedish bidding zones.

Pulls the price-relevant hourly weather from Open-Meteo's free historical archive
(no API key, data back to 1940):
  - temperature  → heating/cooling demand
  - 100 m wind   → wind generation (depresses SE3/SE4 day-ahead prices most)
  - solar radiation + cloud cover → PV generation (growing midday-price driver)
  - precipitation → hydro-inflow proxy (~45% of Nordic supply is hydro)

One representative point per zone — the population/load centre, which for SE4
(Malmö) also sits near the major onshore/offshore wind. A multi-point average
would be more faithful but a single point already carries most of the signal;
see ROADMAP for the refinement.

Module is `weather.openmeteo` (mirrors `entsoe_ingest`): the fetch + record
shaping live here, the CLI/backfill driver lives in `fetch_weather.py`.
"""

from __future__ import annotations

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Representative (lat, lon) per bidding zone — load centres; SE4 also near wind.
ZONE_COORDS = {
    "SE_1": (65.58, 22.15),   # Luleå
    "SE_2": (62.39, 17.31),   # Sundsvall
    "SE_3": (59.33, 18.07),   # Stockholm
    "SE_4": (55.60, 13.00),   # Malmö
}

# Open-Meteo hourly variables we request -> our column names.
_VARS = {
    "temperature_2m": "temp_c",          # °C
    "wind_speed_100m": "wind_100m",      # km/h, turbine hub height
    "shortwave_radiation": "solar_rad",  # W/m², PV driver
    "cloud_cover": "cloud_cover",        # %, total cover
    "precipitation": "precip",           # mm, rain + snow water-equivalent
}

# Open-Meteo archive data starts 1940; prices start 2015, so this is the floor.
EARLIEST = "2015-01-01"


def fetch_weather(lat: float, lon: float, start: pd.Timestamp,
                  end: pd.Timestamp, timeout: float = 60.0) -> pd.DataFrame:
    """Hourly weather for one point over [start, end], indexed in UTC.

    Returns a DataFrame with columns temp_c, wind_100m (empty if no data).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "hourly": ",".join(_VARS),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=list(_VARS.values()))
    idx = pd.to_datetime(times, utc=True)
    df = pd.DataFrame(
        {col: hourly.get(api, []) for api, col in _VARS.items()}, index=idx
    )
    df.index.name = "starts_at"
    return df


def to_records(zone: str, df: pd.DataFrame, source: str = "open-meteo") -> list[dict]:
    """Convert a weather DataFrame to upsert-ready rows for `weather`."""
    if df is None or df.empty:
        return []
    cols = [c for c in _VARS.values() if c in df.columns]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = []
    for ts, row in df.iterrows():
        vals = {c: (None if pd.isna(row[c]) else float(row[c])) for c in cols}
        if all(v is None for v in vals.values()):  # skip fully-empty hours
            continue
        out.append({"zone": zone, "starts_at": ts.isoformat(), "source": source, **vals})
    return out


def iter_year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end) over [start, end] in <=1-year windows.

    Keeps each archive request bounded and backfills resumable (upserts are
    idempotent), mirroring the ENTSO-E ingester."""
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        yield cur, nxt
        cur = nxt
