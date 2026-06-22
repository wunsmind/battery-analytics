"""LP-based dispatch optimizer (Phase 3) — proper revenue-stacking foundation.

Replaces the naive ThresholdArbitrageOptimizer with a linear program that
maximizes arbitrage profit net of degradation and efficiency losses, subject to
SoC, power, and (optional) warranty constraints. It only cycles when the spread
beats the round-trip + degradation cost — fixing the over-cycling the baseline
showed on fine-grained data.

Model (grid-side power, per step t, Δt hours):
    maximize  Σ price_t·Δt·(d_t − c_t)  −  c_deg · (d_t·Δt/η)
    s.t.      soc_{t+1} = soc_t + c_t·Δt·η − d_t·Δt/η
              0 ≤ c_t ≤ P_chg,   0 ≤ d_t ≤ P_dis
              soc_lo ≤ soc_t ≤ soc_hi
where η is one-way efficiency (√RTE) and c_deg the marginal degradation €/MWh on
cell-side discharge throughput (d_t·Δt/η). No binaries are needed: losses +
degradation make simultaneous charge/discharge strictly unprofitable, so it stays
a fast LP.

Long horizons are chunked into windows (carrying SoC across them). Each window is
solved with *perfect foresight* of its prices, so results are an upper bound on
what a real (forecast-driven) controller could achieve — the benchmark to beat.

Energy-only today; the Product/MarketData model is built to add reserve
(FCR/aFRR/mFRR) capacity + activation terms here later.
"""

from __future__ import annotations

import pandas as pd
import pulp

from .assets import BatteryAsset
from .dispatch import (
    DispatchOptimizer,
    DispatchResult,
    DispatchSchedule,
    RevenueBreakdown,
)
from .markets import MarketData


def _solve_window(
    prices: list[float],
    dt: float,
    p_chg: float,
    p_dis: float,
    eff: float,
    soc_lo: float,
    soc_hi: float,
    soc_start: float,
    c_deg: float,
) -> tuple[list[float], list[float], list[float], float]:
    """Solve one LP window. Returns (charge, discharge, soc_after, arbitrage_revenue).

    arbitrage_revenue excludes degradation (reported separately).
    """
    n = len(prices)
    prob = pulp.LpProblem("dispatch", pulp.LpMaximize)
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=p_chg) for t in range(n)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=p_dis) for t in range(n)]
    soc = [pulp.LpVariable(f"s{t}", lowBound=soc_lo, upBound=soc_hi) for t in range(n + 1)]

    prob += soc[0] == soc_start
    for t in range(n):
        prob += soc[t + 1] == soc[t] + c[t] * dt * eff - d[t] * dt / eff

    revenue = pulp.lpSum(prices[t] * dt * (d[t] - c[t]) for t in range(n))
    degr = pulp.lpSum(c_deg * d[t] * dt / eff for t in range(n))
    prob += revenue - degr

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")

    charge = [c[t].value() or 0.0 for t in range(n)]
    discharge = [d[t].value() or 0.0 for t in range(n)]
    soc_after = [soc[t].value() or 0.0 for t in range(n)]  # SoC at the start of each step
    arb = sum(prices[t] * dt * (discharge[t] - charge[t]) for t in range(n))
    return charge, discharge, soc_after, arb


def _solve_window_robust(
    scenarios: list[list[float]],
    weights: list[float],
    dt: float,
    p_chg: float,
    p_dis: float,
    eff: float,
    soc_lo: float,
    soc_hi: float,
    soc_start: float,
    c_deg: float,
    beta: float,
) -> tuple[list[float], list[float], list[float]]:
    """Risk-aware LP: pick ONE schedule good across price scenarios.

    Objective: maximize (1−β)·E[profit] + β·worst-case profit, where worst-case is
    a max-min term (z ≤ profit_s for every scenario s). β=0 → expected-value
    (≈ point forecast); β=1 → fully robust max-min. Scenarios are the per-hour
    quantile paths (a marginal-quantile approximation of joint price paths).
    """
    n = len(scenarios[0])
    S = len(scenarios)
    prob = pulp.LpProblem("robust_dispatch", pulp.LpMaximize)
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=p_chg) for t in range(n)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=p_dis) for t in range(n)]
    soc = [pulp.LpVariable(f"s{t}", lowBound=soc_lo, upBound=soc_hi) for t in range(n + 1)]
    z = pulp.LpVariable("worst")

    prob += soc[0] == soc_start
    for t in range(n):
        prob += soc[t + 1] == soc[t] + c[t] * dt * eff - d[t] * dt / eff

    deg = pulp.lpSum(c_deg * d[t] * dt / eff for t in range(n))
    profit = [pulp.lpSum(scenarios[s][t] * dt * (d[t] - c[t]) for t in range(n)) - deg
              for s in range(S)]
    for s in range(S):
        prob += z <= profit[s]
    expected = pulp.lpSum(weights[s] * profit[s] for s in range(S))
    prob += (1 - beta) * expected + beta * z

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"robust LP not optimal: {pulp.LpStatus[status]}")
    charge = [c[t].value() or 0.0 for t in range(n)]
    discharge = [d[t].value() or 0.0 for t in range(n)]
    soc_after = [soc[t].value() or 0.0 for t in range(n)]
    return charge, discharge, soc_after


class MILPDispatchOptimizer(DispatchOptimizer):
    """Perfect-foresight LP dispatch for energy arbitrage.

    window_days: horizon is solved in windows of this many days, carrying SoC
    across them (keeps each LP small/fast). Larger = fewer boundary artifacts but
    slower. 30 is a good default for backtests.
    """

    def __init__(self, window_days: float = 30.0):
        self.window_days = window_days

    def optimize(
        self,
        asset: BatteryAsset,
        data: MarketData,
        *,
        soc_uncertainty_frac: float = 0.0,
    ) -> DispatchResult:
        if asset.currency != data.currency:
            raise ValueError(
                f"currency mismatch: asset {asset.currency} vs data {data.currency}"
            )
        prices = data.spot_price
        dt = data.dt_hours
        eff = asset.one_way_efficiency
        soc_lo, soc_hi = asset.soc_bounds_mwh()
        c_deg = asset.degradation.marginal_cost_per_mwh()
        steps_per_window = max(1, int(round(self.window_days * 24 / dt)))

        soc_start = asset.energy_capacity_mwh * asset.initial_soc_frac
        all_c: list[float] = []
        all_d: list[float] = []
        all_soc: list[float] = []
        arbitrage = 0.0

        vals = prices.to_numpy(dtype=float)
        for i in range(0, len(vals), steps_per_window):
            window = vals[i:i + steps_per_window].tolist()
            c, d, soc_after, arb = _solve_window(
                window, dt, asset.charge_power_mw, asset.discharge_power_mw,
                eff, soc_lo, soc_hi, soc_start, c_deg,
            )
            all_c.extend(c)
            all_d.extend(d)
            all_soc.extend(soc_after)
            arbitrage += arb
            # carry SoC: end-of-window SoC = soc_after[0..n-1] + last step transition
            soc_start = soc_after[-1] + c[-1] * dt * eff - d[-1] * dt / eff

        idx = prices.index
        throughput = sum(d * dt / eff for d in all_d)  # cell-side discharge MWh
        deg_cost = asset.degradation.cost_for_throughput(throughput)
        schedule = DispatchSchedule(
            index=idx,
            charge_mw=pd.Series(all_c, index=idx),
            discharge_mw=pd.Series(all_d, index=idx),
            soc_mwh=pd.Series(all_soc, index=idx),
        )
        return DispatchResult(
            schedule=schedule,
            revenue=RevenueBreakdown(arbitrage=arbitrage, degradation_cost=deg_cost),
            asset=asset,
            discharge_throughput_mwh=throughput,
            equivalent_full_cycles=(
                throughput / asset.usable_energy_mwh if asset.usable_energy_mwh else 0.0
            ),
        )
