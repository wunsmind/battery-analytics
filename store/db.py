"""SQLite persistence for hourly prices.

Tibber only ever returns today + tomorrow, so we accumulate history by upserting
on every fetch. The primary key (home_id, starts_at) makes re-running the fetcher
idempotent: the same hour is updated, never duplicated. We re-write values on
conflict because a given hour can be revised (e.g. tomorrow's prices appear, then
later become "today").
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from tibber.client import PriceHour

SCHEMA = """
CREATE TABLE IF NOT EXISTS hourly_prices (
    home_id    TEXT NOT NULL,
    starts_at  TEXT NOT NULL,           -- ISO-8601 with tz offset
    total      REAL,
    energy     REAL,                    -- spot/wholesale component
    tax        REAL,
    level      TEXT,
    currency   TEXT,
    fetched_at TEXT NOT NULL,           -- when we last wrote this row (UTC ISO)
    PRIMARY KEY (home_id, starts_at)
);
"""


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_prices(db_path: str, rows: Iterable[PriceHour]) -> int:
    """Insert or update price hours. Returns the number of rows written."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = [
        (r.home_id, r.starts_at, r.total, r.energy, r.tax, r.level, r.currency, fetched_at)
        for r in rows
    ]
    if not records:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO hourly_prices
                (home_id, starts_at, total, energy, tax, level, currency, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(home_id, starts_at) DO UPDATE SET
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


def load_prices(db_path: str, home_id: str | None = None) -> pd.DataFrame:
    """Load all stored prices as a DataFrame, parsed and sorted by time."""
    with connect(db_path) as conn:
        query = "SELECT * FROM hourly_prices"
        params: tuple = ()
        if home_id:
            query += " WHERE home_id = ?"
            params = (home_id,)
        df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, format="ISO8601")
        df = df.sort_values("starts_at").reset_index(drop=True)
    return df


def list_homes(db_path: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT home_id FROM hourly_prices ORDER BY home_id"
        ).fetchall()
    return [r[0] for r in rows]
