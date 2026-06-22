"""SQLite persistence for spot prices.

Tibber only ever returns today + tomorrow, so we accumulate history by upserting
on every fetch. The primary key (home_id, starts_at, resolution) makes re-running
idempotent and lets HOURLY and QUARTER_HOURLY data coexist (both have a :00 row
for the same instant but different values). We re-write values on conflict because
a given interval can be revised (e.g. tomorrow's prices appear, then become today).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from tibber.client import PriceHour

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    home_id    TEXT NOT NULL,
    starts_at  TEXT NOT NULL,           -- ISO-8601 with tz offset
    resolution TEXT NOT NULL,           -- HOURLY | QUARTER_HOURLY
    total      REAL,
    energy     REAL,                    -- spot/wholesale component
    tax        REAL,
    level      TEXT,
    currency   TEXT,
    fetched_at TEXT NOT NULL,           -- when we last wrote this row (UTC ISO)
    PRIMARY KEY (home_id, starts_at, resolution)
);
"""

# Bidding-zone wholesale prices (ENTSO-E day-ahead). Separate from `prices`
# because these are zone-level market prices (EUR/MWh), not a home's consumer
# price — and they're internal market data (see data-licensing memory).
ZONE_SCHEMA = """
CREATE TABLE IF NOT EXISTS zone_prices (
    zone       TEXT NOT NULL,           -- e.g. SE_3
    starts_at  TEXT NOT NULL,           -- ISO-8601 with tz offset
    resolution TEXT NOT NULL,           -- HOURLY | QUARTER_HOURLY
    price      REAL,                    -- day-ahead price, currency/MWh
    currency   TEXT,                    -- e.g. EUR
    source     TEXT,                    -- e.g. entsoe
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (zone, starts_at, resolution)
);
"""

# Weather per bidding zone (Open-Meteo hourly archive). One representative point
# per zone; the price-relevant drivers are wind (depresses SE3/SE4 prices) and
# temperature (heating demand). Hourly only — quarter-hourly features upsample.
WEATHER_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather (
    zone        TEXT NOT NULL,          -- e.g. SE_3
    starts_at   TEXT NOT NULL,          -- ISO-8601 UTC, top of hour
    temp_c      REAL,                   -- 2 m air temperature, °C
    wind_100m   REAL,                   -- 100 m wind speed, km/h (turbine hub height)
    solar_rad   REAL,                   -- shortwave radiation, W/m² (PV driver)
    cloud_cover REAL,                   -- total cloud cover, %
    precip      REAL,                   -- precipitation, mm (rain+snow; hydro proxy)
    source      TEXT,                   -- e.g. open-meteo
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (zone, starts_at)
);
"""

# Columns added after the initial (temp, wind) weather table shipped. ALTER is
# idempotent-guarded so older DBs upgrade in place without losing rows.
_WEATHER_ADDED = ("solar_rad", "cloud_cover", "precip")


def _ensure_weather_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(weather)")}
    for name in _WEATHER_ADDED:
        if name not in cols:
            conn.execute(f"ALTER TABLE weather ADD COLUMN {name} REAL")

# Legacy table (pre resolution column) -> migrate into `prices`, tagged HOURLY.
MIGRATE = """
INSERT OR IGNORE INTO prices
    (home_id, starts_at, resolution, total, energy, tax, level, currency, fetched_at)
SELECT home_id, starts_at, 'HOURLY', total, energy, tax, level, currency, fetched_at
FROM hourly_prices;
DROP TABLE hourly_prices;
"""


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    has_legacy = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hourly_prices'"
    ).fetchone()
    if has_legacy:
        conn.executescript(MIGRATE)


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA)
        conn.executescript(ZONE_SCHEMA)
        conn.executescript(WEATHER_SCHEMA)
        _ensure_weather_columns(conn)
        _migrate_legacy(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_prices(db_path: str, rows: Iterable[PriceHour]) -> int:
    """Insert or update price intervals. Returns the number of rows written."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = [
        (r.home_id, r.starts_at, r.resolution, r.total, r.energy, r.tax,
         r.level, r.currency, fetched_at)
        for r in rows
    ]
    if not records:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO prices
                (home_id, starts_at, resolution, total, energy, tax, level, currency, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(home_id, starts_at, resolution) DO UPDATE SET
                total=excluded.total,
                energy=excluded.energy,
                tax=excluded.tax,
                level=excluded.level,
                currency=excluded.currency,
                fetched_at=excluded.fetched_at;
            """,
            records,
        )
    return len(records)


def load_prices(
    db_path: str, home_id: str | None = None, resolution: str | None = None
) -> pd.DataFrame:
    """Load stored prices as a DataFrame, parsed and sorted by time."""
    clauses, params = [], []
    if home_id:
        clauses.append("home_id = ?")
        params.append(home_id)
    if resolution:
        clauses.append("resolution = ?")
        params.append(resolution)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM prices" + where, conn, params=tuple(params))
    if not df.empty:
        df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, format="ISO8601")
        df = df.sort_values("starts_at").reset_index(drop=True)
    return df


def list_homes(db_path: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT home_id FROM prices ORDER BY home_id").fetchall()
    return [r[0] for r in rows]


def list_resolutions(db_path: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT resolution FROM prices ORDER BY resolution"
        ).fetchall()
    return [r[0] for r in rows]


# ---- zone prices (ENTSO-E) -------------------------------------------------

def upsert_zone_prices(db_path: str, rows: Iterable[dict]) -> int:
    """Insert/update zone day-ahead prices. Rows are dicts with keys:
    zone, starts_at, resolution, price, currency, source."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = [
        (r["zone"], r["starts_at"], r["resolution"], r.get("price"),
         r.get("currency"), r.get("source"), fetched_at)
        for r in rows
    ]
    if not records:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO zone_prices
                (zone, starts_at, resolution, price, currency, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(zone, starts_at, resolution) DO UPDATE SET
                price=excluded.price,
                currency=excluded.currency,
                source=excluded.source,
                fetched_at=excluded.fetched_at;
            """,
            records,
        )
    return len(records)


def load_zone_prices(
    db_path: str, zone: str | None = None, resolution: str | None = None
) -> pd.DataFrame:
    clauses, params = [], []
    if zone:
        clauses.append("zone = ?")
        params.append(zone)
    if resolution:
        clauses.append("resolution = ?")
        params.append(resolution)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM zone_prices" + where, conn, params=tuple(params))
    if not df.empty:
        df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, format="ISO8601")
        df = df.sort_values("starts_at").reset_index(drop=True)
    return df


def list_zones(db_path: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT zone FROM zone_prices ORDER BY zone").fetchall()
    return [r[0] for r in rows]


def list_zone_resolutions(db_path: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT resolution FROM zone_prices ORDER BY resolution"
        ).fetchall()
    return [r[0] for r in rows]


# ---- weather (Open-Meteo) --------------------------------------------------

def upsert_weather(db_path: str, rows: Iterable[dict]) -> int:
    """Insert/update hourly weather. Rows are dicts with keys:
    zone, starts_at, temp_c, wind_100m, solar_rad, cloud_cover, precip, source."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = [
        (r["zone"], r["starts_at"], r.get("temp_c"), r.get("wind_100m"),
         r.get("solar_rad"), r.get("cloud_cover"), r.get("precip"),
         r.get("source"), fetched_at)
        for r in rows
    ]
    if not records:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO weather
                (zone, starts_at, temp_c, wind_100m, solar_rad, cloud_cover,
                 precip, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(zone, starts_at) DO UPDATE SET
                temp_c=excluded.temp_c,
                wind_100m=excluded.wind_100m,
                solar_rad=excluded.solar_rad,
                cloud_cover=excluded.cloud_cover,
                precip=excluded.precip,
                source=excluded.source,
                fetched_at=excluded.fetched_at;
            """,
            records,
        )
    return len(records)


def load_weather(db_path: str, zone: str | None = None) -> pd.DataFrame:
    """Load stored weather as a time-indexed DataFrame (UTC), or empty if none."""
    where, params = ("", ())
    if zone:
        where, params = (" WHERE zone = ?", (zone,))
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM weather" + where, conn, params=params)
    if not df.empty:
        df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, format="ISO8601")
        df = df.sort_values("starts_at").reset_index(drop=True)
    return df
