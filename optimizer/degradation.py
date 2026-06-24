"""Degradation as a marginal cost of throughput.

A degradation model's only job in dispatch is to price each MWh of cycling so the
optimizer stops over-cycling for tiny spreads (see ROADMAP.md). For LFP the cost
is small (~€0.01-0.03/kWh ≈ €10-30/MWh) and often smaller than round-trip-
efficiency losses — so start simple (throughput) and add fidelity only if it pays.

Staging (per roadmap):
  1. ThroughputDegradationModel   — flat €/MWh, linear in the optimizer  (here)
  2. RainflowDegradationModel     — DoD-dependent equivalent cycles      (here)
  3. SemiEmpiricalDegradationModel— calendar + cycle f(T, SoC, DoD, C)   (stub)

The LP optimizes against the *linear* marginal (1); the rainflow model (2) is a
post-hoc / settlement re-pricing of the realized SoC path against a DoD curve —
it values irregular cycling the linear rate cannot see. Both agree on a diet of
full cycles by construction (shared full-DoD anchor).
"""

from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .rainflow import extract_cycles


class DegradationModel(ABC):
    """Maps battery usage to a monetary wear cost (same currency as the asset)."""

    @abstractmethod
    def marginal_cost_per_mwh(self) -> float:
        """Cost of one additional MWh of (discharge) throughput."""

    def cost_for_throughput(self, throughput_mwh: float) -> float:
        """Default: linear in throughput. Override for nonlinear (DoD) models."""
        return self.marginal_cost_per_mwh() * max(throughput_mwh, 0.0)

    def cost_for_soc_trajectory(self, soc_mwh: Sequence[float]) -> float:
        """Wear cost of a realized SoC path (MWh per step).

        Default: collapse the path to discharge throughput (sum of downward
        steps) and price it linearly — identical to what the LP optimizes
        against. DoD-aware models override this to re-price the *shape* of the
        path, which is the whole point of rainflow.
        """
        throughput = sum(
            max(a - b, 0.0) for a, b in zip(soc_mwh, soc_mwh[1:])
        )
        return self.cost_for_throughput(throughput)


@dataclass(frozen=True)
class DoDCycleLifeCurve:
    """Cycles-to-end-of-life as a function of depth of discharge (DoD).

    Datasheets publish this as a handful of points on a log-log chart (deeper
    cycles wear faster, so the curve falls steeply). We store the anchor points
    and interpolate in log-log space — a piecewise power law, which is how these
    curves behave physically. Outside the anchored range we extrapolate along the
    nearest segment's slope. DoD is a fraction of *usable* capacity in (0, 1].
    """

    points: tuple[tuple[float, float], ...]  # (dod_fraction, cycles_to_eol), ascending DoD

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("DoDCycleLifeCurve needs at least two anchor points.")
        dods = [d for d, _ in self.points]
        if dods != sorted(dods) or len(set(dods)) != len(dods):
            raise ValueError("Anchor points must have strictly ascending DoD.")
        if any(not (0.0 < d <= 1.0) for d in dods):
            raise ValueError("DoD anchors must lie in (0, 1].")
        if any(n <= 0 for _, n in self.points):
            raise ValueError("Cycle-life anchors must be positive.")

    def cycles_to_eol(self, dod: float) -> float:
        """Interpolated cycles-to-EoL at depth `dod` (clamped to (0, 1])."""
        dod = min(max(dod, 1e-9), 1.0)
        xs = [math.log(d) for d, _ in self.points]
        ys = [math.log(n) for _, n in self.points]
        lx = math.log(dod)
        if lx <= xs[0]:
            i, j = 0, 1
        elif lx >= xs[-1]:
            i, j = len(xs) - 2, len(xs) - 1
        else:
            j = bisect.bisect_left(xs, lx)
            i = j - 1
        t = (lx - xs[i]) / (xs[j] - xs[i])
        return math.exp(ys[i] + t * (ys[j] - ys[i]))

    @classmethod
    def illustrative_lfp(cls) -> "DoDCycleLifeCurve":
        """An LFP-shaped curve, anchored to 6,000 cycles at 100% DoD.

        ILLUSTRATIVE — replace with the real CATL datasheet curve before any
        economic conclusion. The 100%-DoD anchor matches the throughput model's
        6,000-cycle assumption, so the two models agree on a full-cycle diet and
        diverge only on shallow/irregular cycling.

        The curve is steeper than a pure 1/DoD law (every log-log slope < -1), so
        equivalent lifetime *throughput* d·N(d) rises as cycles get shallower
        (15,000 → 6,000 usable-capacity-equivalents from 10% to 100% DoD). That is
        the real LFP behaviour rainflow exists to value: shallow cycling is gentler
        per MWh, deep cycling disproportionately harsher. A flat 1/DoD curve (slope
        -1) would make this model identical to the linear throughput one.
        """
        return cls(points=(
            (0.10, 150_000.0),
            (0.20, 60_000.0),
            (0.50, 18_000.0),
            (0.80, 9_000.0),
            (1.00, 6_000.0),
        ))


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


@dataclass(frozen=True)
class RainflowDegradationModel(DegradationModel):
    """Rainflow-counted wear against a DoD–cycle-life curve.

    Prices the *shape* of a SoC trajectory, not just its total throughput: the
    path is rainflow-counted (ASTM E1049) into swings, and each swing of depth d
    consumes 1/N(d) of pack life, where N(d) is the cycle-life curve. Two shallow
    half-cycles and one deep full-cycle move the same energy but wear an LFP cell
    differently — that difference is what this model captures and the linear
    throughput model cannot.

        cost = pack_cost × Σ_cycles  count / N(depth)

    The LP still optimizes against `marginal_cost_per_mwh()` (a linear proxy
    pinned to the full-DoD point of the curve); this model re-prices the realized
    trajectory at settlement via `cost_for_soc_trajectory` — the same
    forecast-then-settle split used elsewhere in the pipeline.
    """

    pack_cost: float                 # total replacement cost of the pack (asset currency)
    usable_capacity_mwh: float       # energy across the usable SoC window (DoD = 1.0)
    curve: DoDCycleLifeCurve

    def marginal_cost_per_mwh(self) -> float:
        n_full = self.curve.cycles_to_eol(1.0)
        lifetime_throughput = n_full * self.usable_capacity_mwh
        if lifetime_throughput <= 0:
            return 0.0
        return self.pack_cost / lifetime_throughput

    def cost_for_soc_trajectory(self, soc_mwh: Sequence[float]) -> float:
        if self.usable_capacity_mwh <= 0:
            return 0.0
        damage = 0.0
        for rng, count in extract_cycles(soc_mwh):
            if rng <= 0.0:
                continue
            dod = rng / self.usable_capacity_mwh
            damage += count / self.curve.cycles_to_eol(dod)
        return self.pack_cost * damage


class SemiEmpiricalDegradationModel(DegradationModel):
    """Planned: calendar + cycle aging as f(time, T, SoC, DoD, C-rate).

    Highest fidelity for dispatch; needs a thermal model + datasheet/lab fits.
    """

    def marginal_cost_per_mwh(self) -> float:  # pragma: no cover - stub
        raise NotImplementedError("SemiEmpiricalDegradationModel is a Phase-2 stub.")
