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


def test_commits_fcr_capacity_when_paid():
    # Flat spot -> no arbitrage; a paid FCR-N price should pull out capacity.
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    md = MarketData(index=idx, resolution_minutes=60,
                    spot_price=pd.Series(50.0, index=idx), currency="EUR")
    md.capacity_prices["fcr_n"] = pd.Series(20.0, index=idx)   # 20 EUR/MW/h
    res = MILPDispatchOptimizer(window_days=1).optimize(_asset(), md)
    assert res.revenue.capacity.get("fcr_n", 0.0) > 0
    alloc = res.schedule.reserve_allocation["fcr_n"]
    # FCR-N is symmetric with 60-min endurance on a 1 MW / 2 MWh asset: the
    # both-sided reservation caps commit at min(soc, cap - soc)/1h = 1 MW at mid-SoC.
    assert 0.0 < alloc.max() <= 1.0 + 1e-6


def test_energy_reservation_caps_commitment():
    # FCR-D up endurance is 20 min (1/3 h), so committed MW <= usable_energy / (1/3).
    # A 1 MW / 0.2 MWh-usable asset is energy-bound at 0.6 MW, below the 1 MW power cap.
    idx = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    md = MarketData(index=idx, resolution_minutes=60,
                    spot_price=pd.Series(50.0, index=idx), currency="EUR")
    md.capacity_prices["fcr_d_up"] = pd.Series(100.0, index=idx)  # up-only, big price
    small = BatteryAsset(name="small", energy_capacity_mwh=0.2, power_max_mw=1.0,
                         round_trip_efficiency=0.9, soc_min_frac=0.0, soc_max_frac=1.0,
                         currency="EUR",
                         degradation=ThroughputDegradationModel(
                             pack_cost=500.0, lifetime_throughput_mwh=100.0))
    res = MILPDispatchOptimizer(window_days=1).optimize(small, md)
    commit = res.schedule.reserve_allocation["fcr_d_up"].max()
    assert commit <= 0.6 + 1e-6        # energy-bound...
    assert commit > 0.5                # ...and binding below the 1 MW power cap


def test_stacking_never_worse_than_spot_only():
    asset = _asset()
    base = _data()                                   # spot only
    stack = _data()
    stack.capacity_prices["fcr_d_up"] = pd.Series(8.0, index=stack.index)
    r_base = MILPDispatchOptimizer(window_days=2).optimize(asset, base)
    r_stack = MILPDispatchOptimizer(window_days=2).optimize(asset, stack)
    # Capacity is an extra revenue option; the LP can always decline it.
    assert r_stack.revenue.net >= r_base.revenue.net - 1e-6
    assert sum(r_stack.revenue.capacity.values()) > 0


def _qh_reserve_data(start: str, hours: int) -> MarketData:
    # 15-min spot with a spike in the last quarter of every hour + an hourly FCR-N
    # price. Unconstrained, the LP sells FCR for three quarters and dumps power
    # into the spike (commit swings 0<->1 MW mid-hour — an unbiddable schedule);
    # block-constrained it holds ~1 MW of FCR for the whole hour.
    idx = pd.date_range(start, periods=hours * 4, freq="15min", tz="UTC")
    spot = pd.Series([150.0 if i % 4 == 3 else 40.0 for i in range(len(idx))], index=idx)
    md = MarketData(index=idx, resolution_minutes=15, spot_price=spot, currency="EUR")
    md.capacity_prices["fcr_n"] = pd.Series(40.0, index=idx)
    return md


def _assert_constant_within_hours(alloc: pd.Series) -> None:
    g = alloc.groupby(alloc.index.floor("h"))
    spread = g.max() - g.min()
    assert spread.max() <= 1e-6, f"commit changes mid-hour: {spread[spread > 1e-6]}"


def test_hourly_reserve_commit_is_block_constant_on_15min_steps():
    md = _qh_reserve_data("2026-01-01", hours=24)
    res = MILPDispatchOptimizer(window_days=1).optimize(_asset(), md)
    alloc = res.schedule.reserve_allocation["fcr_n"]
    assert alloc.max() > 0                      # still sells FCR...
    _assert_constant_within_hours(alloc)        # ...but one bid per hour


def test_window_boundary_never_splits_an_hour():
    # Start mid-hour and use a window that isn't a whole number of hours, so a
    # naive chunker would cut an MTU in two.
    md = _qh_reserve_data("2026-01-01 00:15", hours=12)
    res = MILPDispatchOptimizer(window_days=0.1).optimize(_asset(), md)  # 9.6 steps
    _assert_constant_within_hours(res.schedule.reserve_allocation["fcr_n"])
    assert len(res.schedule.charge_mw) == len(md.index)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
