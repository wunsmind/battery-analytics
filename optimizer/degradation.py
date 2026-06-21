"""Degradation as a marginal cost of throughput.

A degradation model's only job in dispatch is to price each MWh of cycling so the
optimizer stops over-cycling for tiny spreads (see ROADMAP.md). For LFP the cost
is small (~€0.01-0.03/kWh ≈ €10-30/MWh) and often smaller than round-trip-
efficiency losses — so start simple (throughput) and add fidelity only if it pays.

Staging (per roadmap):
  1. ThroughputDegradationModel   — flat €/MWh, linear in the optimizer  (here)
  2. RainflowDegradationModel     — DoD-dependent equivalent cycles      (stub)
  3. SemiEmpiricalDegradationModel— calendar + cycle f(T, SoC, DoD, C)   (stub)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class DegradationModel(ABC):
    """Maps battery usage to a monetary wear cost (same currency as the asset)."""

    @abstractmethod
    def marginal_cost_per_mwh(self) -> float:
        """Cost of one additional MWh of (discharge) throughput."""

    def cost_for_throughput(self, throughput_mwh: float) -> float:
        """Default: linear in throughput. Override for nonlinear (DoD) models."""
        return self.marginal_cost_per_mwh() * max(throughput_mwh, 0.0)


@dataclass(frozen=True)
class ThroughputDegradationModel(DegradationModel):
    """Flat cost per MWh of discharge throughput.

        marginal = pack_cost / lifetime_throughput_mwh

    where lifetime_throughput_mwh ≈ usable_capacity_mwh × cycle_life. Throughput is
    counted as energy *discharged* (delivered), the conventional cycle basis.
    """

    pack_cost: float                 # total replacement cost of the pack (asset currency)
    lifetime_throughput_mwh: float   # usable_capacity_mwh × cycle_life_to_eol

    def marginal_cost_per_mwh(self) -> float:
        if self.lifetime_throughput_mwh <= 0:
            return 0.0
        return self.pack_cost / self.lifetime_throughput_mwh


class RainflowDegradationModel(DegradationModel):
    """Planned: rainflow-counted equivalent cycles against a DoD–cycle-life curve.

    Values irregular SoC trajectories. Implement in Phase 2 once a DoD curve from
    the real CATL datasheet is available.
    """

    def marginal_cost_per_mwh(self) -> float:  # pragma: no cover - stub
        raise NotImplementedError("RainflowDegradationModel is a Phase-2 stub.")


class SemiEmpiricalDegradationModel(DegradationModel):
    """Planned: calendar + cycle aging as f(time, T, SoC, DoD, C-rate).

    Highest fidelity for dispatch; needs a thermal model + datasheet/lab fits.
    """

    def marginal_cost_per_mwh(self) -> float:  # pragma: no cover - stub
        raise NotImplementedError("SemiEmpiricalDegradationModel is a Phase-2 stub.")
