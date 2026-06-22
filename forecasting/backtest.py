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
from optimizer.milp import _solve_window, _solve_window_robust


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
    daily_pnl: list[float] = []

    for i in range(0, len(fvals), steps):
        fw = fvals[i:i + steps].tolist()
        c, d, soc_after, _ = _solve_window(
            fw, dt, asset.charge_power_mw, asset.discharge_power_mw,
            eff, soc_lo, soc_hi, soc, c_deg,
        )
        aw = avals[i:i + steps]
        win_arb = sum(aw[t] * dt * (d[t] - c[t]) for t in range(len(c)))  # settle on ACTUAL
        win_thru = sum(d[t] * dt / eff for t in range(len(c)))
        realized_arb += win_arb
        throughput += win_thru
        daily_pnl.append(win_arb - c_deg * win_thru)
        soc = soc_after[-1] + c[-1] * dt * eff - d[-1] * dt / eff

    deg = asset.degradation.cost_for_throughput(throughput)
    return {
        "arbitrage": realized_arb,
        "degradation": deg,
        "net": realized_arb - deg,
        "throughput_mwh": throughput,
        "daily_pnl": daily_pnl,
    }


def robust_dispatch(
    asset: BatteryAsset,
    quantiles: pd.DataFrame,
    actual: pd.Series,
    dt: float,
    beta: float = 0.5,
    weights: tuple[float, ...] = (0.25, 0.5, 0.25),
    window_hours: int = 24,
) -> dict[str, float]:
    """Risk-aware dispatch: decide on quantile *scenarios*, settle on actual.

    quantiles: DataFrame of per-hour quantile forecasts (columns sorted low→high,
    e.g. q10/q50/q90), aligned to `actual`. beta tunes risk aversion (0=expected,
    1=worst-case). Returns the same shape as forecast_driven_dispatch.
    """
    q = quantiles.reindex(actual.index).dropna()
    actual = actual.reindex(q.index)
    cols = list(q.columns)
    if len(cols) != len(weights):
        raise ValueError("weights length must match number of quantile columns")
    eff = asset.one_way_efficiency
    soc_lo, soc_hi = asset.soc_bounds_mwh()
    c_deg = asset.degradation.marginal_cost_per_mwh()
    steps = max(1, int(round(window_hours / dt)))

    soc = asset.energy_capacity_mwh * asset.initial_soc_frac
    avals = actual.to_numpy(dtype=float)
    qmat = {c: q[c].to_numpy(dtype=float) for c in cols}
    realized_arb = 0.0
    throughput = 0.0
    daily_pnl: list[float] = []

    for i in range(0, len(avals), steps):
        scenarios = [qmat[c][i:i + steps].tolist() for c in cols]
        cc, dd, soc_after = _solve_window_robust(
            scenarios, list(weights), dt, asset.charge_power_mw, asset.discharge_power_mw,
            eff, soc_lo, soc_hi, soc, c_deg, beta,
        )
        aw = avals[i:i + steps]
        win_arb = sum(aw[t] * dt * (dd[t] - cc[t]) for t in range(len(cc)))
        win_thru = sum(dd[t] * dt / eff for t in range(len(cc)))
        realized_arb += win_arb
        throughput += win_thru
        daily_pnl.append(win_arb - c_deg * win_thru)
        soc = soc_after[-1] + cc[-1] * dt * eff - dd[-1] * dt / eff

    deg = asset.degradation.cost_for_throughput(throughput)
    return {
        "arbitrage": realized_arb,
        "degradation": deg,
        "net": realized_arb - deg,
        "throughput_mwh": throughput,
        "daily_pnl": daily_pnl,
    }
