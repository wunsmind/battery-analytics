#!/usr/bin/env python3
"""Fetch the latest Tibber prices and store them in SQLite.

Run anytime — it's idempotent. Best run once daily after ~13:00 CET (when
tomorrow's Nord Pool day-ahead prices publish) to capture the full curve.

    python fetch.py

Cron example (14:05 every day, logging to fetch.log):
    5 14 * * *  cd /path/to/battery-analytics && /path/to/.venv/bin/python fetch.py >> fetch.log 2>&1
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from store.db import upsert_prices
from tibber.client import TibberError, fetch_prices


def main() -> int:
    load_dotenv()
    token = os.getenv("TIBBER_TOKEN")
    db_path = os.getenv("DB_PATH", "prices.db")

    if not token or token == "your-token-here":
        print("ERROR: set TIBBER_TOKEN in .env (copy from .env.example).", file=sys.stderr)
        return 1

    # Capture both resolutions every run. Quarter-hourly is the native market unit
    # (15-min since 2025-10-01) and Tibber only serves it for ~7 days, so it must be
    # fetched regularly to build history without gaps.
    total_written = 0
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for resolution in ("HOURLY", "QUARTER_HOURLY"):
        try:
            rows = fetch_prices(token, resolution=resolution)
        except TibberError as e:
            print(f"ERROR ({resolution}): {e}", file=sys.stderr)
            return 1
        written = upsert_prices(db_path, rows)
        total_written += written
        print(f"[{stamp}] {resolution}: fetched {len(rows)}, upserted {written}")

    print(f"[{stamp}] total upserted {total_written} into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
