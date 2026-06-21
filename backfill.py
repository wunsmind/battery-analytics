#!/usr/bin/env python3
"""One-off backfill of recent history from Tibber's priceInfoRange.

Tibber caps lookback at ~31 days (HOURLY) / ~7 days (QUARTER_HOURLY), so this
gives an immediate head start beyond the today+tomorrow that fetch.py captures.
For deeper history (back to 2015, all bidding zones), use an ENTSO-E ingester.

    python backfill.py                 # ~31 days, hourly
    python backfill.py QUARTER_HOURLY  # ~7 days, quarter-hourly

Idempotent — re-running just upserts the same hours.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from store.db import upsert_prices
from tibber.client import TibberError, fetch_price_range


def main() -> int:
    load_dotenv()
    token = os.getenv("TIBBER_TOKEN")
    db_path = os.getenv("DB_PATH", "prices.db")
    resolution = sys.argv[1].upper() if len(sys.argv) > 1 else "HOURLY"

    if not token or token == "your-token-here":
        print("ERROR: set TIBBER_TOKEN in .env (copy from .env.example).", file=sys.stderr)
        return 1

    try:
        rows = fetch_price_range(token, resolution=resolution)
    except (TibberError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    written = upsert_prices(db_path, rows)
    if rows:
        span = f"{rows[0].starts_at} -> {rows[-1].starts_at}"
    else:
        span = "no data returned"
    print(f"backfill[{resolution}]: {len(rows)} intervals upserted into {db_path} ({span})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
