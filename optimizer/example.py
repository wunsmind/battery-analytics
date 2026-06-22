#!/usr/bin/env python3
"""Run the baseline dispatch optimizer on real stored prices.

    python -m optimizer.example

Demonstrates the asset/markets/dispatch interfaces end-to-end. This is the naive
spot-only lower bound — the Phase-3 MILP with revenue stacking should beat it.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from optimizer import BatteryAsset, MarketData, ThresholdArbitrageOptimizer


def main() -> int:
    load_dotenv()
    db_path = os.getenv("DB_PATH", "prices.db")

    data = MarketData.from_prices_db(db_path, resolution="HOURLY", metric="energy")
    asset = BatteryAsset.example_catl_lfp(currency=data.currency)
    optimizer = ThresholdArbitrageOptimizer(charge_pct=25, discharge_pct=75)

    result = optimizer.optimize(asset, data)
    print(result.summary())
    print(
        f"\nMarginal degradation cost: "
        f"{asset.degradation.marginal_cost_per_mwh():.1f} {asset.currency}/MWh"
    )
    avail = ", ".join(p.id for p in data.available_products())
    print(f"Available products this run: {avail}")
    print("(reserve products inactive until ENTSO-E / SvK price feeds are wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
