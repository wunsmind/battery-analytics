#!/usr/bin/env python3
"""Backfill / update ENTSO-E day-ahead forecasts into SQLite (zone_forecasts).

Pulls the gate-aligned exogenous drivers: wind+solar generation forecast and
total load forecast, per bidding zone (see entsoe_forecasts for the why).

Examples:
    python fetch_forecasts.py                       # last 30 days, default zones
    python fetch_forecasts.py --start 2023-01-01    # deep backfill to ~now
    python fetch_forecasts.py --wind-zones DE_LU DK_2 --load-zones SE_4

Idempotent — re-running refreshes existing rows; safe to resume after interrupt.
Chunks by year. Zones that don't publish a series are reported and skipped.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from entsoe_forecasts import (
    LOAD_ZONES,
    WIND_SOLAR_ZONES,
    fetch_load,
    fetch_wind_solar,
    to_records,
)
from entsoe_ingest import iter_year_chunks
from store.db import upsert_zone_forecasts

TZ = "Europe/Stockholm"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill ENTSO-E day-ahead forecasts.")
    p.add_argument("--wind-zones", nargs="*", default=WIND_SOLAR_ZONES,
                   help=f"Zones for wind/solar forecast (default: {' '.join(WIND_SOLAR_ZONES)}; "
                        "pass with no values to skip)")
    p.add_argument("--load-zones", nargs="*", default=LOAD_ZONES,
                   help=f"Zones for load forecast (default: {' '.join(LOAD_ZONES)}; "
                        "pass with no values to skip)")
    p.add_argument("--start", help="ISO date, e.g. 2023-01-01 (default: 30 days ago)")
    p.add_argument("--end", help="ISO date (default: now)")
    return p.parse_args(argv)


def _ingest(label: str, zones: list[str], fetch, token: str, db_path: str,
            start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Run one fetch kind (wind/solar or load) over zones × year-chunks."""
    total = 0
    for zone in zones:
        zone_total = 0
        for c_start, c_end in iter_year_chunks(start, end):
            try:
                frame = fetch(token, zone, c_start, c_end)
            except Exception as e:  # noqa: BLE001 - report and continue other chunks
                print(f"  {label} {zone} {c_start.date()}–{c_end.date()}: skip ({type(e).__name__})",
                      file=sys.stderr)
                continue
            written = upsert_zone_forecasts(db_path, to_records(zone, frame))
            zone_total += written
            print(f"  {label} {zone} {c_start.date()}–{c_end.date()}: {written} rows")
        print(f"{label} {zone}: {zone_total} rows total")
        total += zone_total
    return total


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    load_dotenv()
    token = os.getenv("ENTSOE_TOKEN")
    # Forecasts are large but reproducible, so they share the git-ignored zone DB.
    db_path = os.getenv("ZONE_DB_PATH", "market.db")

    if not token or token == "your-entsoe-token-here":
        print("ERROR: set ENTSOE_TOKEN in .env (see .env.example).", file=sys.stderr)
        return 1

    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ).normalize()
    start = (pd.Timestamp(args.start, tz=TZ) if args.start
             else end - pd.Timedelta(days=30))

    total = _ingest("wind/solar", args.wind_zones, fetch_wind_solar, token, db_path, start, end)
    total += _ingest("load", args.load_zones, fetch_load, token, db_path, start, end)

    print(f"Done: {total} rows upserted into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
