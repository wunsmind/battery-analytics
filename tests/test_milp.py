#!/usr/bin/env python3
"""Tests for the LP dispatch optimizer (optimizer.milp).

    python -m tests.test_milp
"""

from __future__ import annotations

import pandas as pd

from optimizer import BatteryAsset, MarketData, MILPDispatchOptimizer, ThresholdArbitrageOptimizer
from optimizer.degradation import ThroughputDegradationModel


def _data(n: int = 48, lo: float = 50.0, hi: float = 500.0) -> MarketData:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    prices = pd.Series([lo if i % 2 == 0 else hi for i in range(n)], index=idx)
    return MarketData(index=idx, resolution_minutes=60, spot_price=prices, currency="EUR")


def _asset(marginal: float = 5.0) -> BatteryAsset:
    return BatteryAsset(
        name="test", energy_capacity_mwh=2.0, power_max_mw=1.0,
        round_trip_efficiency=0.9, soc_min_frac=0.0, soc_max_frac=1.0,
        currency="EUR",
        degradation=ThroughputDegradationModel(
            pack_cost=marginal * 100.0, lifetime_throughput_mwh=100.0),  # marginal EUR/MWh
    )


def test_soc_within_bounds():
    asset, data = _asset(), _data()
    res = MILPDispatchOptimizer(window_days=1).optimize(asset, data)
    lo, hi = asset.soc_bounds_mwh()
    assert res.schedule.soc_mwh.min() >= lo - 1e-6
    assert res.schedule.soc_mwh.max() <= hi + 1e-6


def test_milp_beats_or_matches_baseline():
    asset, data = _asset(), _data()
    milp = MILPDispatchOptimizer(window_days=2).optimize(asset, data)
    base = ThresholdArbitrageOptimizer(25, 75).optimize(asset, data)
    assert milp.revenue.net >= base.revenue.net - 1e-6
    assert milp.revenue.net > 0


def test_high_degradation_suppresses_cycling():
    # Marginal degradation far above any achievable spread -> don't cycle.
    asset, data = _asset(marginal=10_000.0), _data(lo=50.0, hi=60.0)
    res = MILPDispatchOptimizer(window_days=2).optimize(asset, data)
    assert res.discharge_throughput_mwh < 1e-6
    assert abs(res.revenue.net) < 1e-6


def test_power_cap_respected():
    asset, data = _asset(), _data()
    res = MILPDispatchOptimizer(window_days=1).optimize(asset, data)
    assert res.schedule.charge_mw.max() <= asset.charge_power_mw + 1e-6
    assert res.schedule.discharge_mw.max() <= asset.discharge_power_mw + 1e-6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
