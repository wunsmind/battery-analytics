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

Revenue stacking (spot + FCR capacity): each available capacity-reserve product
gets a per-step commitment r_k,t (MW) that competes with arbitrage for the same
power and energy. Coupling (LP relaxation — continuous MW, no lot-sizing):
    power:   d_t + Σ_{up} r_k,t ≤ P_dis ,   c_t + Σ_{down} r_k,t ≤ P_chg
    energy:  reserved MW must be sustainable for the product's endurance E_k —
             soc − soc_lo ≥ Σ_{up} r·E_k  and  soc_hi − soc ≥ Σ_{down} r·E_k
             (enforced at both ends of the step). FCR-N is symmetric (both sides).
    revenue: + Σ price_k,t·r_k,t·Δt   (capacity price is currency/MW/h)
    wear:    − c_deg · Σ act_frac_k·r_k,t·Δt   (expected activated throughput)
Stays a fast LP. Activation *energy markets* (aFRR/mFRR) are not modelled yet —
this is FCR capacity stacking, which is what the Mimer feed supplies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pulp

from .assets import BatteryAsset
from .dispatch import (
    DispatchOptimizer,
    DispatchResult,
    DispatchSchedule,
    RevenueBreakdown,
)
from .markets import Direction, MarketData, Mechanism


@dataclass(frozen=True)
class _Reserve:
    """One capacity-reserve product resolved onto a solve window."""

    id: str
    up: bool                      # needs discharge-side power + energy headroom
    down: bool                    # needs charge-side power + energy headroom
    endurance_h: float            # sustain requirement -> energy reserved per MW
    act_frac: float               # E[activated energy]/committed MW (wear proxy)
    prices: tuple[float, ...]     # capacity price (cur/MW/h) per step; NaN = unavailable
    # Per-step market-time-unit id (e.g. the hour a 15-min step belongs to). Steps
    # sharing an id must carry the same commitment — an hourly FCR bid can't
    # change mid-hour. None = every step is its own MTU.
    blocks: tuple[int, ...] | None = None

    @staticmethod
    def committable(price: float) -> bool:
        return price is not None and not math.isnan(price)


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
    reserves: list[_Reserve] | None = None,
) -> tuple[list[float], list[float], list[float], float, dict[str, list[float]], dict[str, float], float]:
    """Solve one LP window with optional FCR capacity stacking.

    Returns (charge, discharge, soc_after, arbitrage_revenue, reserve_alloc,
    capacity_revenue, reserve_activation_mwh). arbitrage_revenue and
    capacity_revenue exclude degradation (reported separately); reserve_alloc maps
    product id -> per-step committed MW; reserve_activation_mwh is the expected
    activated throughput used for wear accounting.
    """
    reserves = reserves or []
    n = len(prices)
    prob = pulp.LpProblem("dispatch", pulp.LpMaximize)
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=p_chg) for t in range(n)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=p_dis) for t in range(n)]
    soc = [pulp.LpVariable(f"s{t}", lowBound=soc_lo, upBound=soc_hi) for t in range(n + 1)]

    # Reserve commitments: forced to 0 on steps with no published price.
    p_cap = max(p_chg, p_dis)
    r = {
        k.id: [pulp.LpVariable(f"r_{k.id}_{t}", lowBound=0,
                               upBound=p_cap if k.committable(k.prices[t]) else 0.0)
               for t in range(n)]
        for k in reserves
    }
    # Block commitments: one bid per product MTU (hourly FCR over 15-min steps).
    for k in reserves:
        if k.blocks is None:
            continue
        for t in range(1, n):
            if k.blocks[t] == k.blocks[t - 1]:
                prob += r[k.id][t] == r[k.id][t - 1]

    prob += soc[0] == soc_start
    for t in range(n):
        prob += soc[t + 1] == soc[t] + c[t] * dt * eff - d[t] * dt / eff
        up = [k for k in reserves if k.up]
        down = [k for k in reserves if k.down]
        # Power headroom: reserved MW is unavailable to arbitrage on its side.
        prob += d[t] + pulp.lpSum(r[k.id][t] for k in up) <= p_dis
        prob += c[t] + pulp.lpSum(r[k.id][t] for k in down) <= p_chg
        # Energy reservation: committed MW must be sustainable for endurance E_k,
        # enforced at both ends of the step so intra-step cycling can't void it.
        for s in (soc[t], soc[t + 1]):
            prob += s - soc_lo >= pulp.lpSum(r[k.id][t] * k.endurance_h for k in up)
            prob += soc_hi - s >= pulp.lpSum(r[k.id][t] * k.endurance_h for k in down)

    arb = pulp.lpSum(prices[t] * dt * (d[t] - c[t]) for t in range(n))
    cap = pulp.lpSum(k.prices[t] * r[k.id][t] * dt
                     for k in reserves for t in range(n) if k.committable(k.prices[t]))
    degr = pulp.lpSum(c_deg * d[t] * dt / eff for t in range(n))
    wear = pulp.lpSum(c_deg * k.act_frac * r[k.id][t] * dt
                      for k in reserves for t in range(n))
    prob += arb + cap - degr - wear

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")

    charge = [c[t].value() or 0.0 for t in range(n)]
    discharge = [d[t].value() or 0.0 for t in range(n)]
    soc_after = [soc[t].value() or 0.0 for t in range(n)]  # SoC at the start of each step
    reserve_alloc = {k.id: [r[k.id][t].value() or 0.0 for t in range(n)] for k in reserves}
    arb_val = sum(prices[t] * dt * (discharge[t] - charge[t]) for t in range(n))
    cap_val = {
        k.id: sum(k.prices[t] * reserve_alloc[k.id][t] * dt
                  for t in range(n) if k.committable(k.prices[t]))
        for k in reserves
    }
    reserve_act = sum(k.act_frac * reserve_alloc[k.id][t] * dt
                      for k in reserves for t in range(n))
    return charge, discharge, soc_after, arb_val, reserve_alloc, cap_val, reserve_act


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


def _capacity_reserves(data: MarketData) -> list:
    """Available CAPACITY_RESERVE products that have a price series attached."""
    return [p for p in data.available_products()
            if p.mechanism is Mechanism.CAPACITY_RESERVE and p.id in data.capacity_prices]


def _reserve_blocks(products: list, data: MarketData) -> dict[str, "pd.Index"]:
    """Per-step MTU id for each product: the step's timestamp floored to the
    product's market time unit (as int ns). Coarser-than-spot products get blocks;
    a product at or finer than the spot resolution maps every step to itself."""
    out = {}
    for p in products:
        mtu = max(int(p.resolution_minutes), data.resolution_minutes)
        out[p.id] = data.index.floor(f"{mtu}min").asi8
    return out


def _slice_reserves(products: list, data: MarketData, lo: int, hi: int,
                    blocks: dict | None = None) -> list[_Reserve]:
    """Resolve each reserve product onto the window [lo, hi) as a `_Reserve`."""
    out = []
    for p in products:
        vals = data.capacity_prices[p.id].to_numpy(dtype=float)[lo:hi]
        blk = tuple(int(b) for b in blocks[p.id][lo:hi]) if blocks else None
        out.append(_Reserve(
            blocks=blk,
            id=p.id,
            up=p.direction in (Direction.UP, Direction.SYMMETRIC),
            down=p.direction in (Direction.DOWN, Direction.SYMMETRIC),
            endurance_h=(p.endurance_minutes or 0.0) / 60.0,
            act_frac=p.typical_activation_fraction,
            prices=tuple(vals),
        ))
    return out


class MILPDispatchOptimizer(DispatchOptimizer):
    """Perfect-foresight LP dispatch for energy arbitrage + FCR capacity stacking.

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

        reserve_products = _capacity_reserves(data)

        soc_start = asset.energy_capacity_mwh * asset.initial_soc_frac
        all_c: list[float] = []
        all_d: list[float] = []
        all_soc: list[float] = []
        arbitrage = 0.0
        capacity_rev: dict[str, float] = {p.id: 0.0 for p in reserve_products}
        reserve_alloc: dict[str, list[float]] = {p.id: [] for p in reserve_products}
        reserve_throughput = 0.0

        blocks = _reserve_blocks(reserve_products, data)

        vals = prices.to_numpy(dtype=float)
        n_total = len(vals)
        i = 0
        while i < n_total:
            j = min(i + steps_per_window, n_total)
            # Never split a reserve MTU across windows (its block-equal commitment
            # would be unenforceable): extend the window to the MTU boundary.
            while j < n_total and any(b[j] == b[j - 1] for b in blocks.values()):
                j += 1
            window = vals[i:j].tolist()
            reserves = _slice_reserves(reserve_products, data, i, j, blocks)
            i = j
            c, d, soc_after, arb, alloc, cap, r_act = _solve_window(
                window, dt, asset.charge_power_mw, asset.discharge_power_mw,
                eff, soc_lo, soc_hi, soc_start, c_deg, reserves,
            )
            all_c.extend(c)
            all_d.extend(d)
            all_soc.extend(soc_after)
            arbitrage += arb
            reserve_throughput += r_act
            for pid, v in cap.items():
                capacity_rev[pid] += v
            for pid, series in alloc.items():
                reserve_alloc[pid].extend(series)
            # carry SoC: end-of-window SoC = soc_after[0..n-1] + last step transition
            soc_start = soc_after[-1] + c[-1] * dt * eff - d[-1] * dt / eff

        idx = prices.index
        # Cell-side throughput = arbitrage discharge + expected reserve activation.
        throughput = sum(d * dt / eff for d in all_d) + reserve_throughput
        deg_cost = asset.degradation.cost_for_throughput(throughput)
        schedule = DispatchSchedule(
            index=idx,
            charge_mw=pd.Series(all_c, index=idx),
            discharge_mw=pd.Series(all_d, index=idx),
            soc_mwh=pd.Series(all_soc, index=idx),
            reserve_allocation={pid: pd.Series(v, index=idx) for pid, v in reserve_alloc.items()},
        )
        return DispatchResult(
            schedule=schedule,
            revenue=RevenueBreakdown(
                arbitrage=arbitrage,
                capacity={k: v for k, v in capacity_rev.items() if v},
                degradation_cost=deg_cost,
            ),
            asset=asset,
            discharge_throughput_mwh=throughput,
            equivalent_full_cycles=(
                throughput / asset.usable_energy_mwh if asset.usable_energy_mwh else 0.0
            ),
        )
