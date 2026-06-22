"""Battery asset model — physical limits + economics, parameterized per pack.

Any pack (CATL LFP today, anything later) is config, not code. The optimizer reads
power/energy limits, the SoC window, round-trip efficiency, the degradation cost
model, and (optional) warranty throughput limit as a hard constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .degradation import DegradationModel, ThroughputDegradationModel


def _default_degradation() -> DegradationModel:
    # Illustrative only — see example_catl_lfp() for the documented derivation.
    return ThroughputDegradationModel(pack_cost=2_280_000.0, lifetime_throughput_mwh=11_400.0)


@dataclass(frozen=True)
class BatteryAsset:
    name: str
    energy_capacity_mwh: float          # nominal energy at 0-100% SoC
    power_max_mw: float                 # default symmetric charge/discharge power cap
    round_trip_efficiency: float = 0.92  # LFP ~0.90-0.94 AC-AC
    soc_min_frac: float = 0.05          # usable window floor
    soc_max_frac: float = 1.0           # usable window ceiling
    initial_soc_frac: float = 0.5
    power_charge_mw: float | None = None     # overrides power_max_mw if asymmetric
    power_discharge_mw: float | None = None
    degradation: DegradationModel = field(default_factory=_default_degradation)
    warranty_throughput_mwh: float | None = None  # lifetime discharge cap (constraint)
    currency: str = "SEK"

    # ---- derived helpers -------------------------------------------------
    @property
    def usable_energy_mwh(self) -> float:
        return self.energy_capacity_mwh * (self.soc_max_frac - self.soc_min_frac)

    @property
    def one_way_efficiency(self) -> float:
        """Per-direction efficiency, split symmetrically from round-trip."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def charge_power_mw(self) -> float:
        return self.power_charge_mw if self.power_charge_mw is not None else self.power_max_mw

    @property
    def discharge_power_mw(self) -> float:
        return self.power_discharge_mw if self.power_discharge_mw is not None else self.power_max_mw

    def soc_bounds_mwh(self) -> tuple[float, float]:
        return (
            self.energy_capacity_mwh * self.soc_min_frac,
            self.energy_capacity_mwh * self.soc_max_frac,
        )

    # ---- factory ---------------------------------------------------------
    @classmethod
    def example_catl_lfp(cls, currency: str = "EUR") -> "BatteryAsset":
        """A 1 MW / 2 MWh CATL-LFP-style asset. NUMBERS ARE ILLUSTRATIVE.

        Verify against the real datasheet before any economic conclusion:
        cycle-life-vs-DoD, calendar curve, RTE, C-rate, warranty terms.

        Degradation derivation (illustrative):
          usable ≈ 2.0 MWh × (1.00-0.05) = 1.9 MWh
          cycle life to EoL ≈ 6,000 cycles
          lifetime discharge throughput ≈ 1.9 × 6,000 ≈ 11,400 MWh
          marginal ≈ 18 EUR/MWh (≈ 200 SEK/MWh) → pack_cost = marginal × lifetime

        `currency` must match the market data it's run against (EUR for ENTSO-E
        zone prices, SEK for Tibber home prices) — the optimizer enforces this.
        """
        usable = 2.0 * (1.0 - 0.05)
        cycle_life = 6_000
        lifetime_throughput = usable * cycle_life
        marginal_per_mwh = {"EUR": 18.0, "SEK": 200.0}.get(currency.upper(), 18.0)
        return cls(
            name=f"CATL LFP 1MW/2MWh (illustrative, {currency.upper()})",
            energy_capacity_mwh=2.0,
            power_max_mw=1.0,
            round_trip_efficiency=0.92,
            soc_min_frac=0.05,
            soc_max_frac=1.0,
            degradation=ThroughputDegradationModel(
                pack_cost=marginal_per_mwh * lifetime_throughput,
                lifetime_throughput_mwh=lifetime_throughput,
            ),
            warranty_throughput_mwh=lifetime_throughput,
            currency=currency.upper(),
        )
