"""Rolling-horizon (receding-horizon / MPC) dispatch backtest — Phase 3.

`forecast_driven_dispatch` (see backtest.py) optimizes one fixed 24h window at a
time and commits the whole thing. That makes the optimizer blind past every
midnight: with no view of tomorrow it has no reason *not* to empty the battery at
each window boundary, and a half-charged pack going into a cheap morning is left
on the table. A real day-ahead controller looks further.

This module runs the honest version. At each daily gate it:
  1. forecasts a longer lookahead (default 48h = today + tomorrow),
  2. LP-optimizes the entire lookahead from the *realized* state of charge,
  3. commits only the first 24h to the auction, settles it on actual prices,
  4. rolls SoC forward on that executed block, and re-plans at the next gate.

The committed day is forecast from realized prices alone — honest, because at the
~12:00 CET gate the whole previous day is already cleared, so every lag (24/48/
168h back) is known. The lookahead *beyond* tomorrow is a **recursive** forecast:
tomorrow's same-hour lag is today's price, which isn't realized at the gate, so
the model's own day-D predictions feed day-(D+1)'s lag features. That far half is
necessarily lower quality (forecast error compounds) — and the open question this
backtest answers is whether the SoC-continuity benefit of seeing past midnight
outweighs the noisier far-horizon forecast. The settlement is always on actuals,
so the P&L is what the controller would really have banked.
"""

from __future__ import annotations

import pandas as pd

from optimizer.assets import BatteryAsset
from optimizer.milp import _solve_window
from .features import build_features
from .models import GBMForecaster


def recursive_forecast(
    model: GBMForecaster,
    history: pd.Series,
    horizon: pd.DatetimeIndex,
    block_steps: int,
    weather: pd.DataFrame | None = None,
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Forecast `horizon` at a gate, given realized prices in `history`.

    Predicts block by block (each `block_steps` long). A block's own features only
    reference lags ≥ 24h — i.e. earlier blocks — so each block is forecast, then
    written back into the working series to become the lag inputs for later blocks.
    This makes the second day onward a genuine recursive forecast (no leakage of
    unrealized future prices), matching what a controller actually has at the gate.

    `history` must already contain enough lead-in for the longest lag (≥ 7 days for
    lag168 / the rolling features); callers pass a trimmed tail of the full series.
    """
    work = history.astype(float).sort_index().copy()
    preds: list[pd.Series] = []
    for i in range(0, len(horizon), block_steps):
        blk = horizon[i:i + block_steps]
        # Placeholder values so build_features keeps these rows (it drops NaN y).
        # A block's features never depend on the block's own values (all lags are
        # ≥ 24h = one block back), so the placeholder is inert — it's overwritten
        # with the prediction below before any later block reads it as a lag.
        work = work.reindex(work.index.union(blk))
        work.loc[blk] = 0.0
        X, _ = build_features(work, weather, exog)
        Xb = X.reindex(blk).dropna()
        if Xb.empty:
            break
        pb = model.predict(Xb)
        work.loc[pb.index] = pb.to_numpy()
        preds.append(pb)
    return pd.concat(preds) if preds else pd.Series(dtype=float)


def rolling_horizon_dispatch(
    asset: BatteryAsset,
    model: GBMForecaster,
    series: pd.Series,
    test_index: pd.DatetimeIndex,
    dt: float,
    *,
    weather: pd.DataFrame | None = None,
    exog: pd.DataFrame | None = None,
    lookahead_hours: int = 48,
    commit_hours: int = 24,
    warmup_days: int = 14,
) -> dict[str, float]:
    """Receding-horizon dispatch over `test_index`. Returns forecast_driven_dispatch's
    dict shape (arbitrage / degradation / net / throughput_mwh / daily_pnl).

    `series` is the full realized price history (for lags *and* settlement);
    `test_index` is the out-of-sample horizon to dispatch over. At each gate the
    LP plans `lookahead_hours` but only the first `commit_hours` are executed and
    settled, then SoC rolls forward and the gate advances by `commit_hours`.
    """
    series = series.astype(float).sort_index()
    eff = asset.one_way_efficiency
    soc_lo, soc_hi = asset.soc_bounds_mwh()
    c_deg = asset.degradation.marginal_cost_per_mwh()
    look_steps = max(1, int(round(lookahead_hours / dt)))
    commit_steps = max(1, int(round(commit_hours / dt)))

    test_index = pd.DatetimeIndex(test_index).sort_values()
    warmup = pd.Timedelta(days=warmup_days)
    soc = asset.energy_capacity_mwh * asset.initial_soc_frac
    realized_arb = 0.0
    throughput = 0.0
    daily_pnl: list[float] = []

    i = 0
    n = len(test_index)
    while i < n:
        horizon = test_index[i:i + look_steps]
        gate = horizon[0]
        hist = series.loc[(series.index < gate) & (series.index >= gate - warmup)]
        fc = recursive_forecast(model, hist, horizon, commit_steps, weather, exog)
        fc = fc.reindex(horizon).ffill().bfill()  # guard rare unfilled tails

        c, d, soc_after, _ = _solve_window(
            fc.to_numpy(dtype=float).tolist(), dt,
            asset.charge_power_mw, asset.discharge_power_mw,
            eff, soc_lo, soc_hi, soc, c_deg,
        )
        # Commit & settle only the first commit_steps of the plan on ACTUAL prices.
        k = min(commit_steps, len(horizon))
        a = series.reindex(horizon).to_numpy(dtype=float)
        win_arb = sum(a[t] * dt * (d[t] - c[t]) for t in range(k))
        win_thru = sum(d[t] * dt / eff for t in range(k))
        realized_arb += win_arb
        throughput += win_thru
        daily_pnl.append(win_arb - c_deg * win_thru)
        # Roll SoC through the executed block only (mirrors _solve_window's carry).
        soc = soc_after[k - 1] + c[k - 1] * dt * eff - d[k - 1] * dt / eff
        i += k

    deg = asset.degradation.cost_for_throughput(throughput)
    return {
        "arbitrage": realized_arb,
        "degradation": deg,
        "net": realized_arb - deg,
        "throughput_mwh": throughput,
        "daily_pnl": daily_pnl,
    }
