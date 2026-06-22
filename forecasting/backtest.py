"""Forecast evaluation + forecast-driven dispatch P&L.

The honest test of a price forecast for a trader isn't just MAE — it's the money
it makes. forecast_driven_dispatch decides each day's schedule on the *forecast*
prices, then settles that physical schedule against the *actual* prices. SoC
physics are always respected (they depend on power, not price), so the only thing
the forecast error changes is realized revenue — exactly as in live operation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.assets import BatteryAsset
from optimizer.milp import _solve_window


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    a, p = y_true.align(y_pred, join="inner")
    err = (a - p).dropna()
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "n": int(len(err)),
    }


def forecast_driven_dispatch(
    asset: BatteryAsset,
    forecast: pd.Series,
    actual: pd.Series,
    dt: float,
    window_hours: int = 24,
) -> dict[str, float]:
    """Decide dispatch on `forecast`, settle on `actual`. Returns P&L breakdown.

    Both series must share an index (the test horizon). Solves per day-window,
    carrying SoC across windows — mirroring day-ahead operation.
    """
    forecast, actual = forecast.align(actual, join="inner")
    eff = asset.one_way_efficiency
    soc_lo, soc_hi = asset.soc_bounds_mwh()
    c_deg = asset.degradation.marginal_cost_per_mwh()
    steps = max(1, int(round(window_hours / dt)))

    soc = asset.energy_capacity_mwh * asset.initial_soc_frac
    fvals = forecast.to_numpy(dtype=float)
    avals = actual.to_numpy(dtype=float)
    realized_arb = 0.0
    throughput = 0.0

    for i in range(0, len(fvals), steps):
        fw = fvals[i:i + steps].tolist()
        c, d, soc_after, _ = _solve_window(
            fw, dt, asset.charge_power_mw, asset.discharge_power_mw,
            eff, soc_lo, soc_hi, soc, c_deg,
        )
        aw = avals[i:i + steps]
        for t in range(len(c)):
            realized_arb += aw[t] * dt * (d[t] - c[t])   # settle on ACTUAL price
            throughput += d[t] * dt / eff
        soc = soc_after[-1] + c[-1] * dt * eff - d[-1] * dt / eff

    deg = asset.degradation.cost_for_throughput(throughput)
    return {
        "arbitrage": realized_arb,
        "degradation": deg,
        "net": realized_arb - deg,
        "throughput_mwh": throughput,
    }
