"""Rainflow counting + DoD-aware degradation costing."""

import math

import pytest

from optimizer.assets import BatteryAsset
from optimizer.degradation import (
    DoDCycleLifeCurve,
    RainflowDegradationModel,
    ThroughputDegradationModel,
)
from optimizer.rainflow import count_ranges, extract_cycles


# --- rainflow counting ----------------------------------------------------

def test_astm_canonical_example():
    """The textbook ASTM E1049-85 series and its documented cycle histogram."""
    series = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    assert count_ranges(series) == {3: 0.5, 4: 1.5, 6: 0.5, 8: 1.0, 9: 0.5}


def test_total_count_is_half_the_segments():
    """Every record conserves count: Σ count = (reversals - 1) / 2."""
    series = [-2, 1, -3, 5, -1, 3, -4, 4, -2]  # 9 reversals → 8 segments → 4.0
    assert sum(c for _, c in extract_cycles(series)) == pytest.approx(4.0)


def test_single_full_cycle():
    """Up then back to start = one closed full cycle of that range."""
    assert count_ranges([0.0, 1.0, 0.0]) == {1.0: 1.0}


def test_flat_and_monotonic_have_no_cycles():
    assert extract_cycles([0.5, 0.5, 0.5]) == []
    assert extract_cycles([0.0, 1.0, 2.0, 3.0]) == [(3.0, 0.5)]  # one residual half


def test_plateaus_collapse():
    """Repeated SoC holds don't invent turning points."""
    assert count_ranges([0.0, 1.0, 1.0, 1.0, 0.0]) == {1.0: 1.0}


# --- DoD cycle-life curve --------------------------------------------------

def test_curve_exact_at_anchors():
    curve = DoDCycleLifeCurve.illustrative_lfp()
    for dod, n in curve.points:
        assert curve.cycles_to_eol(dod) == pytest.approx(n)


def test_curve_monotonic_decreasing():
    curve = DoDCycleLifeCurve.illustrative_lfp()
    cycles = [curve.cycles_to_eol(d) for d in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    assert all(a > b for a, b in zip(cycles, cycles[1:]))


def test_curve_log_log_midpoint():
    """Halfway (in log space) between two anchors is their geometric mean."""
    curve = DoDCycleLifeCurve(points=((0.1, 1000.0), (1.0, 10.0)))
    mid_dod = math.sqrt(0.1 * 1.0)
    assert curve.cycles_to_eol(mid_dod) == pytest.approx(math.sqrt(1000.0 * 10.0))


def test_curve_validation():
    with pytest.raises(ValueError):
        DoDCycleLifeCurve(points=((0.5, 100.0),))               # too few
    with pytest.raises(ValueError):
        DoDCycleLifeCurve(points=((1.0, 10.0), (0.5, 100.0)))   # not ascending
    with pytest.raises(ValueError):
        DoDCycleLifeCurve(points=((0.0, 100.0), (1.0, 10.0)))   # DoD out of range
    with pytest.raises(ValueError):
        DoDCycleLifeCurve(points=((0.5, 0.0), (1.0, 10.0)))     # bad cycle life


# --- rainflow degradation model -------------------------------------------

def _rainflow_model(usable=1.9, pack_cost=205_200.0):
    return RainflowDegradationModel(
        pack_cost=pack_cost,
        usable_capacity_mwh=usable,
        curve=DoDCycleLifeCurve.illustrative_lfp(),
    )


def test_marginal_matches_throughput_model_at_full_dod():
    """LP-facing marginal cost agrees with the linear throughput model."""
    usable, cycle_life, marginal = 1.9, 6_000, 18.0
    pack_cost = marginal * usable * cycle_life
    rf = _rainflow_model(usable=usable, pack_cost=pack_cost)
    tp = ThroughputDegradationModel(
        pack_cost=pack_cost, lifetime_throughput_mwh=usable * cycle_life
    )
    assert rf.marginal_cost_per_mwh() == pytest.approx(tp.marginal_cost_per_mwh())
    assert rf.marginal_cost_per_mwh() == pytest.approx(18.0)


def test_one_full_cycle_costs_pack_over_full_life():
    """A single full-depth cycle consumes 1/N(1.0) of the pack."""
    rf = _rainflow_model()
    soc = [0.0, 1.9, 0.0]  # full usable swing
    n_full = rf.curve.cycles_to_eol(1.0)
    assert rf.cost_for_soc_trajectory(soc) == pytest.approx(rf.pack_cost / n_full)


def test_curve_steeper_than_inverse_dod():
    """Every segment slope < -1 ⇒ shallow cycling is genuinely gentler per MWh.

    Guards against a 1/DoD curve (slope -1), which would silently collapse this
    model into the linear throughput one.
    """
    curve = DoDCycleLifeCurve.illustrative_lfp()
    pts = curve.points
    for (d1, n1), (d2, n2) in zip(pts, pts[1:]):
        slope = math.log(n2 / n1) / math.log(d2 / d1)
        assert slope < -1.0


def test_shallow_cycling_cheaper_per_mwh_than_deep():
    """The whole point: equal throughput, gentler shallow cycling costs less."""
    rf = _rainflow_model()
    # One deep cycle: 1.9 MWh discharged. Range = full usable.
    deep = [0.0, 1.9, 0.0]
    # Same 1.9 MWh discharged as five shallow cycles of depth 1.9/5 (DoD 0.2).
    step = 1.9 / 5.0
    shallow = [0.0]
    for _ in range(5):
        shallow += [step, 0.0]
    deep_cost = rf.cost_for_soc_trajectory(deep)
    shallow_cost = rf.cost_for_soc_trajectory(shallow)
    # Equal discharge throughput...
    assert sum(max(a - b, 0) for a, b in zip(deep, deep[1:])) == pytest.approx(1.9)
    assert sum(max(a - b, 0) for a, b in zip(shallow, shallow[1:])) == pytest.approx(1.9)
    # ...but shallow wears far less: 5 cycles at N(0.2)=60k vs 1 at N(1.0)=6k.
    assert shallow_cost == pytest.approx(rf.pack_cost * 5 / 60_000)
    assert deep_cost == pytest.approx(rf.pack_cost / 6_000)
    assert shallow_cost < 0.6 * deep_cost


def test_base_trajectory_default_is_linear_throughput():
    """Non-DoD models price a trajectory as plain discharge throughput."""
    tp = ThroughputDegradationModel(pack_cost=100_000.0, lifetime_throughput_mwh=10_000.0)
    soc = [0.0, 1.0, 0.5, 1.5, 0.0]  # discharge throughput = 0.5 + 1.5 = 2.0
    assert tp.cost_for_soc_trajectory(soc) == pytest.approx(tp.cost_for_throughput(2.0))


def test_asset_factory_marginals_agree():
    """Both CATL factories present the same LP-facing marginal cost."""
    flat = BatteryAsset.example_catl_lfp("EUR")
    rf = BatteryAsset.example_catl_lfp_rainflow("EUR")
    assert rf.degradation.marginal_cost_per_mwh() == pytest.approx(
        flat.degradation.marginal_cost_per_mwh()
    )
