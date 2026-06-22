"""Dispatch optimizer interface + revenue breakdown + a naive baseline.

The interface is the contract the real revenue-stacking MILP (Phase 3) will
implement. `ThresholdArbitrageOptimizer` is a deliberately simple, spot-only
baseline so the data model is exercised end-to-end and gives a lower bound to
beat — it ignores reserves entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from .assets import BatteryAsset
from .markets import MarketData


@dataclass
class RevenueBreakdown:
    """Per-stream P&L for the horizon, in the asset's currency."""

    arbitrage: float = 0.0
    capacity: dict[str, float] = field(default_factory=dict)    # product_id -> revenue
    activation: dict[str, float] = field(default_factory=dict)  # product_id -> revenue
    degradation_cost: float = 0.0

    @property
    def gross(self) -> float:
        return self.arbitrage + sum(self.capacity.values()) + sum(self.activation.values())

    @property
    def net(self) -> float:
        return self.gross - self.degradation_cost


@dataclass
class DispatchSchedule:
    """The decided per-step actions (MW per step; SoC in MWh, at step start)."""

    index: pd.DatetimeIndex
    charge_mw: pd.Series
    discharge_mw: pd.Series
    soc_mwh: pd.Series
    reserve_allocation: dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class DispatchResult:
    schedule: DispatchSchedule
    revenue: RevenueBreakdown
    asset: BatteryAsset
    discharge_throughput_mwh: float = 0.0
    equivalent_full_cycles: float = 0.0

    def summary(self) -> str:
        r, cur = self.revenue, self.asset.currency
        lines = [
            f"Asset: {self.asset.name}",
            f"Horizon steps: {len(self.schedule.index)} "
            f"({self.schedule.index[0]} → {self.schedule.index[-1]})",
            f"Throughput: {self.discharge_throughput_mwh:.2f} MWh "
            f"({self.equivalent_full_cycles:.1f} equivalent full cycles)",
            f"Arbitrage revenue:   {r.arbitrage:11,.0f} {cur}",
        ]
        for pid, v in r.capacity.items():
            lines.append(f"Capacity [{pid}]:   {v:11,.0f} {cur}")
        for pid, v in r.activation.items():
            lines.append(f"Activation [{pid}]: {v:11,.0f} {cur}")
        lines += [
            f"Degradation cost:  -{r.degradation_cost:11,.0f} {cur}",
            f"NET:                 {r.net:11,.0f} {cur}",
        ]
        return "\n".join(lines)


class DispatchOptimizer(ABC):
    """Contract for any dispatch strategy/optimizer."""

    @abstractmethod
    def optimize(
        self,
        asset: BatteryAsset,
        data: MarketData,
        *,
        soc_uncertainty_frac: float = 0.0,
    ) -> DispatchResult:
        """Return an optimized dispatch over data's horizon.

        soc_uncertainty_frac reserves SoC headroom for robustness (Phase 3); the
        baseline accepts but does not yet use it.
        """


class ThresholdArbitrageOptimizer(DispatchOptimizer):
    """Naive spot-only baseline: charge cheap hours, discharge expensive hours.

    Charges when the spot price is below `charge_pct` percentile and discharges
    above `discharge_pct`, respecting power, SoC window, and efficiency, and
    charging straight to degradation cost. A lower bound for the real optimizer.
    """

    def __init__(self, charge_pct: float = 25.0, discharge_pct: float = 75.0):
        if not 0 <= charge_pct < discharge_pct <= 100:
            raise ValueError("require 0 <= charge_pct < discharge_pct <= 100")
        self.charge_pct = charge_pct
        self.discharge_pct = discharge_pct

    def optimize(
        self,
        asset: BatteryAsset,
        data: MarketData,
        *,
        soc_uncertainty_frac: float = 0.0,
    ) -> DispatchResult:
        if asset.currency != data.currency:
            raise ValueError(
                f"currency mismatch: asset is {asset.currency} but market data is "
                f"{data.currency}. Use a matching asset (FX conversion is not yet modelled)."
            )
        prices = data.spot_price
        dt = data.dt_hours
        eff = asset.one_way_efficiency
        soc_lo, soc_hi = asset.soc_bounds_mwh()
        charge_thr = prices.quantile(self.charge_pct / 100.0)
        discharge_thr = prices.quantile(self.discharge_pct / 100.0)

        soc = asset.energy_capacity_mwh * asset.initial_soc_frac
        charge, discharge, soc_trace = [], [], []
        arbitrage = 0.0
        throughput = 0.0  # MWh discharged from the cells

        for price in prices:
            soc_trace.append(soc)
            c_mw = d_mw = 0.0
            if price <= charge_thr and soc < soc_hi:
                # Energy we can still store, referred to grid side (before losses).
                headroom_grid = (soc_hi - soc) / eff
                c_mw = min(asset.charge_power_mw, headroom_grid / dt)
                grid_in = c_mw * dt
                soc += grid_in * eff
                arbitrage -= grid_in * price
            elif price >= discharge_thr and soc > soc_lo:
                # Energy available from cells; grid receives after losses.
                cell_avail = soc - soc_lo
                d_mw = min(asset.discharge_power_mw, cell_avail / dt)
                cell_out = d_mw * dt
                soc -= cell_out
                arbitrage += cell_out * eff * price
                throughput += cell_out
            charge.append(c_mw)
            discharge.append(d_mw)

        deg_cost = asset.degradation.cost_for_throughput(throughput)
        schedule = DispatchSchedule(
            index=prices.index,
            charge_mw=pd.Series(charge, index=prices.index),
            discharge_mw=pd.Series(discharge, index=prices.index),
            soc_mwh=pd.Series(soc_trace, index=prices.index),
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
