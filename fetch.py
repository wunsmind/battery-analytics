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

    try:
        rows = fetch_prices(token)
    except TibberError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    written = upsert_prices(db_path, rows)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] fetched {len(rows)} hours, upserted {written} into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
