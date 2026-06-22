#!/usr/bin/env python3
"""Backfill / update Open-Meteo weather into SQLite (weather table).

Examples:
    python fetch_weather.py                       # last 30 days, SE_3 + SE_4
    python fetch_weather.py --zones SE_3 SE_4 SE_1 SE_2
    python fetch_weather.py --start 2015-01-01    # deep backfill to ~now

No API key required. Idempotent — re-running refreshes existing rows; safe to
resume after interrupt. Chunks by year for steady, resumable progress.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from store.db import upsert_weather
from weather.openmeteo import ZONE_COORDS, fetch_weather, iter_year_chunks, to_records

TZ = "Europe/Stockholm"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Open-Meteo weather.")
    p.add_argument("--zones", nargs="+", default=["SE_3", "SE_4"],
                   help=f"Bidding zones (default: SE_3 SE_4). All: {' '.join(ZONE_COORDS)}")
    p.add_argument("--start", help="ISO date, e.g. 2015-01-01 (default: 30 days ago)")
    p.add_argument("--end", help="ISO date (default: today)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    load_dotenv()
    # Shares the large, git-ignored zone DB (weather is reproducible via this CLI).
    db_path = os.getenv("ZONE_DB_PATH", "market.db")

    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ).normalize()
    start = (pd.Timestamp(args.start, tz=TZ) if args.start
             else end - pd.Timedelta(days=30))

    total = 0
    for zone in args.zones:
        coords = ZONE_COORDS.get(zone)
        if coords is None:
            print(f"  {zone}: no coordinates configured, skipping", file=sys.stderr)
            continue
        lat, lon = coords
        zone_total = 0
        for c_start, c_end in iter_year_chunks(start, end):
            try:
                df = fetch_weather(lat, lon, c_start, c_end)
            except Exception as e:  # noqa: BLE001 - report and continue other chunks
                print(f"  {zone} {c_start.date()}–{c_end.date()}: ERROR {e}", file=sys.stderr)
                continue
            written = upsert_weather(db_path, to_records(zone, df))
            zone_total += written
            print(f"  {zone} {c_start.date()}–{c_end.date()}: {written} rows")
        print(f"{zone}: {zone_total} rows total")
        total += zone_total

    print(f"Done: {total} rows upserted into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
