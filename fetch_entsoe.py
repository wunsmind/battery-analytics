#!/usr/bin/env python3
"""Backfill / update ENTSO-E day-ahead prices into SQLite (zone_prices table).

Examples:
    python fetch_entsoe.py                      # last 30 days, SE_3 + SE_4
    python fetch_entsoe.py --zones SE_3 SE_4 SE_1 SE_2
    python fetch_entsoe.py --start 2015-01-05   # deep backfill to ~now
    python fetch_entsoe.py --start 2024-01-01 --end 2025-01-01

Idempotent — re-running refreshes existing rows; safe to resume after interrupt.
Chunks by year so long backfills make steady, resumable progress.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from entsoe_ingest import SWEDISH_ZONES, fetch_day_ahead, iter_year_chunks, to_records
from store.db import upsert_zone_prices

TZ = "Europe/Stockholm"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill ENTSO-E day-ahead prices.")
    p.add_argument("--zones", nargs="+", default=["SE_3", "SE_4"],
                   help=f"Bidding zones (default: SE_3 SE_4). All: {' '.join(SWEDISH_ZONES)}")
    p.add_argument("--start", help="ISO date, e.g. 2015-01-05 (default: 30 days ago)")
    p.add_argument("--end", help="ISO date (default: now)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    load_dotenv()
    token = os.getenv("ENTSOE_TOKEN")
    db_path = os.getenv("DB_PATH", "prices.db")

    if not token or token == "your-entsoe-token-here":
        print("ERROR: set ENTSOE_TOKEN in .env (see .env.example).", file=sys.stderr)
        return 1

    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ).normalize()
    start = (pd.Timestamp(args.start, tz=TZ) if args.start
             else end - pd.Timedelta(days=30))

    total = 0
    for zone in args.zones:
        zone_total = 0
        for c_start, c_end in iter_year_chunks(start, end):
            try:
                series = fetch_day_ahead(token, zone, c_start, c_end)
            except Exception as e:  # noqa: BLE001 - report and continue other chunks
                print(f"  {zone} {c_start.date()}–{c_end.date()}: ERROR {e}", file=sys.stderr)
                continue
            written = upsert_zone_prices(db_path, to_records(zone, series))
            zone_total += written
            print(f"  {zone} {c_start.date()}–{c_end.date()}: {written} rows")
        print(f"{zone}: {zone_total} rows total")
        total += zone_total

    print(f"Done: {total} rows upserted into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
