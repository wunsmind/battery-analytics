#!/usr/bin/env python3
"""Backtest dispatch optimizers on ENTSO-E zone history.

    python -m optimizer.backtest                       # SE_3 hourly, last 365 days
    python -m optimizer.backtest --zone SE_4 --start 2020-01-01 --end 2021-01-01
    python -m optimizer.backtest --resolution QUARTER_HOURLY --start 2026-01-01

Compares the naive threshold baseline vs the perfect-foresight LP. The LP is an
upper bound (it sees future prices); a real forecast-driven controller lands
between the two.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from optimizer import (
    BatteryAsset,
    MarketData,
    MILPDispatchOptimizer,
    ThresholdArbitrageOptimizer,
)


def _trim(data: MarketData, start: str | None, end: str | None) -> MarketData:
    s = data.spot_price
    if start:
        s = s[s.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        s = s[s.index <= pd.Timestamp(end, tz="UTC")]
    return MarketData(index=s.index, resolution_minutes=data.resolution_minutes,
                      spot_price=s, currency=data.currency)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backtest dispatch optimizers.")
    p.add_argument("--zone", default="SE_3")
    p.add_argument("--resolution", default="HOURLY")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--window-days", type=float, default=30.0)
    p.add_argument("--stack", action="store_true",
                   help="Also run spot + FCR capacity stacking and report the uplift")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    load_dotenv()
    zone_db = os.getenv("ZONE_DB_PATH", "market.db")

    data = MarketData.from_zone_prices(zone_db, zone=args.zone, resolution=args.resolution)
    # Default to the last 365 days if no range given (keeps the run quick).
    if not args.start and not args.end:
        end_ts = data.index.max()
        start_ts = end_ts - pd.Timedelta(days=365)
        data = _trim(data, str(start_ts), None)
    else:
        data = _trim(data, args.start, args.end)
    if len(data.index) == 0:
        print("No data in range.", file=sys.stderr)
        return 1

    asset = BatteryAsset.example_catl_lfp(currency=data.currency)
    days = (data.index[-1] - data.index[0]).total_seconds() / 86400 or 1

    base = ThresholdArbitrageOptimizer(25, 75).optimize(asset, data)
    milp = MILPDispatchOptimizer(window_days=args.window_days).optimize(asset, data)

    print(f"Zone {args.zone} | {args.resolution} | {len(data.index):,} steps over {days:.0f} days "
          f"({days/365:.2f} yr) | {asset.name}")
    print(f"{'':18}{'baseline':>14}{'LP (perfect)':>16}")
    for label, attr in [("net", None), ("arbitrage", "arbitrage"),
                        ("degradation", "degradation_cost")]:
        if label == "net":
            b, m = base.revenue.net, milp.revenue.net
        else:
            b, m = getattr(base.revenue, attr), getattr(milp.revenue, attr)
        print(f"{label:18}{b:>14,.0f}{m:>16,.0f}  {data.currency}")
    print(f"{'cycles':18}{base.equivalent_full_cycles:>14,.1f}{milp.equivalent_full_cycles:>16,.1f}")
    print(f"{'net /MW/yr':18}{base.revenue.net*365/days:>14,.0f}"
          f"{milp.revenue.net*365/days:>16,.0f}  {data.currency}/MW/yr")
    uplift = (milp.revenue.net / base.revenue.net - 1) * 100 if base.revenue.net > 0 else float("nan")
    print(f"\nLP uplift over baseline: {uplift:.0f}%")

    if args.stack:
        _print_stacking(args, zone_db, asset, data, milp, days)
    return 0


def _print_stacking(args, zone_db, asset, data, milp_spot, days):
    """Run the LP again with FCR capacity attached and report spot+FCR uplift."""
    stack_data = MarketData(index=data.index, resolution_minutes=data.resolution_minutes,
                            spot_price=data.spot_price, currency=data.currency)
    stack_data.attach_reserves(zone_db, zone=args.zone)
    products = [p.id for p in stack_data.available_products() if p.id != "spot"]
    if not products:
        print(f"\n[stack] No FCR price data for {args.zone} — run fetch_reserve.py first.")
        return
    milp_stack = MILPDispatchOptimizer(window_days=args.window_days).optimize(asset, stack_data)

    cur, ann = data.currency, 365 / days
    print(f"\n=== Spot vs Spot+FCR (LP, perfect foresight) — products: {', '.join(products)} ===")
    print(f"{'':20}{'spot-only':>14}{'spot+FCR':>16}")
    print(f"{'arbitrage':20}{milp_spot.revenue.arbitrage:>14,.0f}"
          f"{milp_stack.revenue.arbitrage:>16,.0f}  {cur}")
    for pid, v in milp_stack.revenue.capacity.items():
        print(f"{'capacity['+pid+']':20}{0:>14,.0f}{v:>16,.0f}  {cur}")
    print(f"{'degradation':20}{milp_spot.revenue.degradation_cost:>14,.0f}"
          f"{milp_stack.revenue.degradation_cost:>16,.0f}  {cur}")
    print(f"{'net':20}{milp_spot.revenue.net:>14,.0f}{milp_stack.revenue.net:>16,.0f}  {cur}")
    print(f"{'net /MW/yr':20}{milp_spot.revenue.net*ann:>14,.0f}"
          f"{milp_stack.revenue.net*ann:>16,.0f}  {cur}/MW/yr")
    print(f"{'cycles':20}{milp_spot.equivalent_full_cycles:>14,.1f}"
          f"{milp_stack.equivalent_full_cycles:>16,.1f}")
    if milp_spot.revenue.net > 0:
        up = (milp_stack.revenue.net / milp_spot.revenue.net - 1) * 100
        print(f"\nFCR stacking uplift over spot-only: {up:.0f}%")
    return


if __name__ == "__main__":
    raise SystemExit(main())
