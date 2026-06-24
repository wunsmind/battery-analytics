"""Illustrative figures for the rainflow degradation model.

Renders four panels that explain, end to end, why DoD-aware (rainflow) costing
differs from a flat €/MWh throughput rate:

  1. the DoD cycle-life curve (the physics input)               — degradation.DoDCycleLifeCurve
  2. that curve translated to wear cost per MWh of throughput   — the economic punchline
  3. a sample SoC trajectory and the swings rainflow extracts   — rainflow.extract_cycles
  4. linear vs rainflow cost across cycling patterns            — the difference, in money

Numbers are ILLUSTRATIVE (see optimizer/assets.py). Run:

    .venv/bin/python plot_degradation.py        # writes figures/rainflow_degradation.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt
import numpy as np

from optimizer.assets import BatteryAsset
from optimizer.degradation import DoDCycleLifeCurve
from optimizer.rainflow import count_ranges

OUT_DIR = "figures"
USABLE = 1.9  # MWh across the usable SoC window (matches example_catl_lfp)


def _sample_soc_path() -> np.ndarray:
    """An illustrative 3-day SoC path: one deep daily cycle + shallow intraday wiggles.

    The kind of trajectory a price-following optimizer actually produces — a big
    overnight-charge / daytime-discharge swing, plus a couple of shallow top-ups
    chasing intraday spread. Exactly the irregular shape rainflow is built to value.
    """
    lo, hi = 0.1, 2.0  # the real usable window (soc_min=0.1, soc_max=2.0 MWh)
    day = [
        lo, hi,                 # deep charge to full
        1.3, 1.6, 1.3,          # shallow wiggle near the top
        lo,                     # deep discharge
        0.5, 0.3, 0.5,          # shallow wiggle near the bottom
    ]
    return np.array(day * 3)


def panel_dod_curve(ax, curve: DoDCycleLifeCurve) -> None:
    dods = np.linspace(0.05, 1.0, 200)
    cycles = [curve.cycles_to_eol(d) for d in dods]
    ax.loglog(dods * 100, cycles, color="#1f77b4", lw=2, label="LFP curve (illustrative)")
    # 1/DoD reference anchored at the 100%-DoD point — the boundary where rainflow
    # would collapse into the linear throughput model.
    n_full = curve.cycles_to_eol(1.0)
    ax.loglog(dods * 100, n_full / dods, color="#999999", ls="--", lw=1.3,
              label="1/DoD reference (slope −1)")
    ax_dods = [d * 100 for d, _ in curve.points]
    ax_cyc = [n for _, n in curve.points]
    ax.scatter(ax_dods, ax_cyc, color="#1f77b4", zorder=5, s=35)
    ax.set_xlabel("Depth of discharge (%)")
    ax.set_ylabel("Cycles to end-of-life")
    ax.set_title("1. DoD cycle-life curve\n(steeper than 1/DoD ⇒ shallow cycling is gentler)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)


def panel_cost_per_mwh(ax, curve: DoDCycleLifeCurve, pack_cost: float, linear_rate: float) -> None:
    dods = np.linspace(0.05, 1.0, 200)
    # Wear cost per MWh of throughput for a cycle of depth d:
    #   one cycle discharges d·USABLE MWh and consumes 1/N(d) of the pack.
    rate = [pack_cost / (curve.cycles_to_eol(d) * d * USABLE) for d in dods]
    ax.plot(dods * 100, rate, color="#d62728", lw=2, label="rainflow (DoD-aware)")
    ax.axhline(linear_rate, color="#7f7f7f", ls="--", lw=1.5,
               label=f"linear throughput ({linear_rate:.0f} €/MWh)")
    ax.set_xlabel("Depth of discharge (%)")
    ax.set_ylabel("Wear cost (€/MWh throughput)")
    ax.set_title("2. Same curve as a marginal cost\n(shallow cycling priced below the flat rate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def panel_soc_and_cycles(ax, soc: np.ndarray, curve: DoDCycleLifeCurve) -> None:
    ax.plot(range(len(soc)), soc, color="#2ca02c", lw=1.6, marker="o", ms=3)
    ax.fill_between(range(len(soc)), 0.1, soc, color="#2ca02c", alpha=0.10)
    ax.set_xlabel("Step")
    ax.set_ylabel("SoC (MWh)")
    ax.set_title("3. Sample dispatch SoC path\n(deep daily cycle + shallow intraday wiggles)")
    ax.grid(True, alpha=0.25)
    # Inset: the rainflow swing histogram (depth vs counted cycles).
    hist = count_ranges(soc)
    ins = ax.inset_axes([0.58, 0.58, 0.39, 0.39])
    depths = sorted(hist)
    ins.bar([d / USABLE * 100 for d in depths], [hist[d] for d in depths],
            width=6, color="#9467bd")
    ins.set_title("rainflow swings", fontsize=7)
    ins.set_xlabel("DoD (%)", fontsize=6)
    ins.set_ylabel("cycles", fontsize=6)
    ins.tick_params(labelsize=6)


def panel_cost_comparison(ax, asset_rf, asset_lin, soc: np.ndarray) -> None:
    step = 1.9 / 5.0
    scenarios = {
        "1 deep\nfull cycle": [0.1, 2.0, 0.1],
        "5 shallow\nDoD-0.2": [0.0] + [v for _ in range(5) for v in (step, 0.0)],
        "3-day\nmixed path": list(soc),
    }
    labels = list(scenarios)
    lin = [asset_lin.degradation.cost_for_soc_trajectory(s) for s in scenarios.values()]
    rf = [asset_rf.degradation.cost_for_soc_trajectory(s) for s in scenarios.values()]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, lin, w, label="linear throughput", color="#7f7f7f")
    ax.bar(x + w / 2, rf, w, label="rainflow (DoD-aware)", color="#d62728")
    for i, (l, r) in enumerate(zip(lin, rf)):
        ax.text(i + w / 2, r, f"{r / l:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Degradation cost (€)")
    ax.set_title("4. Cost by cycling pattern\n(equal-ish throughput, very different wear)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    asset_rf = BatteryAsset.example_catl_lfp_rainflow("EUR")
    asset_lin = BatteryAsset.example_catl_lfp("EUR")
    curve = asset_rf.degradation.curve
    pack_cost = asset_rf.degradation.pack_cost
    linear_rate = asset_lin.degradation.marginal_cost_per_mwh()
    soc = _sample_soc_path()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_dod_curve(axes[0, 0], curve)
    panel_cost_per_mwh(axes[0, 1], curve, pack_cost, linear_rate)
    panel_soc_and_cycles(axes[1, 0], soc, curve)
    panel_cost_comparison(axes[1, 1], asset_rf, asset_lin, soc)
    fig.suptitle(
        "Rainflow degradation — illustrative LFP (1 MW / 2 MWh). "
        "Numbers are not datasheet-verified.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_DIR, "rainflow_degradation.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
