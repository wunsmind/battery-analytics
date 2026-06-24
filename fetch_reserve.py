#!/usr/bin/env python3
"""Backfill / update Svenska kraftnät (Mimer) FCR reserve prices into SQLite.

Examples:
    python fetch_reserve.py                       # last 30 days, all SE+DK2 zones
    python fetch_reserve.py --start 2024-01-01    # backfill to ~now
    python fetch_reserve.py --start 2023-01-01 --end 2024-01-01 --zones SE_3 SE_4

No API key required (Mimer is a public endpoint). Idempotent — re-running
refreshes existing rows. Chunks by 3-month windows so long backfills are
resumable after an interrupt. Writes to market.db (git-ignored internal data) by
default; override with RESERVE_DB_PATH / ZONE_DB_PATH or --db.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from mimer_ingest import (
    ZONE_COLUMN_SUFFIX,
    fetch_fcr_csv,
    iter_month_chunks,
    parse_fcr_csv,
)
from store.db import upsert_reserve_prices

TZ = "Europe/Stockholm"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Mimer FCR reserve prices.")
    p.add_argument("--zones", nargs="+", default=None,
                   help=f"Keep only these zones (default: all). Choices: "
                        f"{' '.join(ZONE_COLUMN_SUFFIX)}")
    p.add_argument("--start", help="ISO date, e.g. 2024-01-01 (default: 30 days ago)")
    p.add_argument("--end", help="ISO date (default: now)")
    p.add_argument("--db", help="SQLite path (default: $RESERVE_DB_PATH/$ZONE_DB_PATH/market.db)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    db_path = (args.db or os.getenv("RESERVE_DB_PATH")
               or os.getenv("ZONE_DB_PATH") or "market.db")
    zones = set(args.zones) if args.zones else None

    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ).normalize()
    start = (pd.Timestamp(args.start, tz=TZ) if args.start
             else end - pd.Timedelta(days=30))

    total = 0
    for c_start, c_end in iter_month_chunks(start, end):
        try:
            csv_text = fetch_fcr_csv(c_start, c_end)
            rows = parse_fcr_csv(csv_text)
        except Exception as e:  # noqa: BLE001 - report and continue other chunks
            print(f"  ! {c_start.date()}..{c_end.date()}: {e}", file=sys.stderr)
            continue
        if zones is not None:
            rows = [r for r in rows if r["zone"] in zones]
        written = upsert_reserve_prices(db_path, rows)
        total += written
        print(f"  {c_start.date()}..{c_end.date()}: {written} rows")

    print(f"Done: {total} reserve rows -> {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
