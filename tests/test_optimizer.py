#!/usr/bin/env python3
"""Sanity tests for the dispatch optimizer scaffold.

Self-contained (synthetic prices, no DB). Runs under pytest or standalone:
    python -m tests.test_optimizer
"""

from __future__ import annotations

import pandas as pd

from optimizer import BatteryAsset, MarketData, ThresholdArbitrageOptimizer
from optimizer.degradation import ThroughputDegradationModel


def _synthetic_data(n: int = 24) -> MarketData:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    # Alternating cheap/expensive hours -> clear arbitrage signal.
    prices = pd.Series([100.0 if i % 2 == 0 else 900.0 for i in range(n)], index=idx)
    return MarketData(index=idx, resolution_minutes=60, spot_price=prices, currency="SEK")


def _asset() -> BatteryAsset:
    return BatteryAsset(
        name="test 1MW/2MWh",
        energy_capacity_mwh=2.0,
        power_max_mw=1.0,
        round_trip_efficiency=0.9,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        degradation=ThroughputDegradationModel(pack_cost=1000.0, lifetime_throughput_mwh=100.0),
    )


def test_soc_stays_within_bounds():
    asset, data = _asset(), _synthetic_data()
    res = ThresholdArbitrageOptimizer().optimize(asset, data)
    lo, hi = asset.soc_bounds_mwh()
    assert res.schedule.soc_mwh.min() >= lo - 1e-9
    assert res.schedule.soc_mwh.max() <= hi + 1e-9


def test_arbitrage_is_profitable_on_clear_spread():
    asset, data = _asset(), _synthetic_data()
    res = ThresholdArbitrageOptimizer(charge_pct=40, discharge_pct=60).optimize(asset, data)
    assert res.revenue.arbitrage > 0
    assert res.discharge_throughput_mwh > 0


def test_degradation_cost_matches_throughput():
    asset, data = _asset(), _synthetic_data()
    res = ThresholdArbitrageOptimizer().optimize(asset, data)
    expected = asset.degradation.marginal_cost_per_mwh() * res.discharge_throughput_mwh
    assert abs(res.revenue.degradation_cost - expected) < 1e-6
    assert res.revenue.net == res.revenue.gross - res.revenue.degradation_cost


def test_efficiency_reduces_delivered_energy():
    # With <100% efficiency, energy delivered to grid < energy drawn from grid.
    asset, data = _asset(), _synthetic_data()
    res = ThresholdArbitrageOptimizer().optimize(asset, data)
    grid_in = (res.schedule.charge_mw * data.dt_hours).sum()
    grid_out = (res.schedule.discharge_mw * data.dt_hours).sum() * asset.one_way_efficiency
    assert grid_out < grid_in  # losses exist


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
