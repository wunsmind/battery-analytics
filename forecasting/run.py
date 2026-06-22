#!/usr/bin/env python3
"""Train a price forecaster and run the three-way dispatch backtest.

    python -m forecasting.run                       # SE_3 hourly, last 365d as test
    python -m forecasting.run --zone SE_4 --test-days 365

Compares dispatch P&L under:
  baseline   — naive threshold strategy on actual prices
  forecast   — LP dispatch on GBM forecast, settled on actual  (realistic)
  perfect    — LP dispatch on actual prices                    (upper bound)
The 'forecast' row is the credible, deployable number.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from forecasting.backtest import (
    forecast_driven_dispatch,
    metrics,
    robust_dispatch,
    scenario_robust_dispatch,
)
from forecasting.features import build_features
from forecasting.models import GBMForecaster, QuantileForecaster, seasonal_naive
from forecasting.scenarios import daily_residual_blocks
from optimizer import (
    BatteryAsset,
    MarketData,
    MILPDispatchOptimizer,
    ThresholdArbitrageOptimizer,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Forecast + dispatch backtest.")
    p.add_argument("--zone", default="SE_3")
    p.add_argument("--resolution", default="HOURLY")
    p.add_argument("--test-days", type=int, default=365)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    load_dotenv()
    zone_db = os.getenv("ZONE_DB_PATH", "market.db")
    data = MarketData.from_zone_prices(zone_db, zone=args.zone, resolution=args.resolution)
    series = data.spot_price
    dt = data.dt_hours

    # Optional weather features (fetch_weather.py populates the `weather` table).
    from store.db import load_weather  # local import to avoid a hard dep at import time
    weather = load_weather(zone_db, zone=args.zone)
    weather = weather.set_index("starts_at") if not weather.empty else None

    # Train/test split: last N days are the out-of-sample test.
    cutoff = series.index.max() - pd.Timedelta(days=args.test_days)
    X, y = build_features(series)
    train = X.index < cutoff
    test = X.index >= cutoff
    if test.sum() == 0 or train.sum() == 0:
        print("Not enough data for the split.", file=sys.stderr)
        return 1

    model = GBMForecaster().fit(X[train], y[train])
    gbm_pred = model.predict(X[test])
    naive_pred = seasonal_naive(series, X.index[test])
    actual = y[test]

    m_gbm = metrics(actual, gbm_pred)
    m_naive = metrics(actual, naive_pred)
    print(f"Zone {args.zone} | {args.resolution} | test {args.test_days}d "
          f"({m_gbm['n']:,} pts) | {data.currency}/MWh")
    print(f"  forecast MAE  — GBM {m_gbm['mae']:.2f} | seasonal-naive {m_naive['mae']:.2f} "
          f"({100*(1-m_gbm['mae']/m_naive['mae']):.0f}% better)")
    print(f"  forecast RMSE — GBM {m_gbm['rmse']:.2f} | seasonal-naive {m_naive['rmse']:.2f}")

    # Three-way dispatch comparison over the test horizon.
    asset = BatteryAsset.example_catl_lfp(currency=data.currency)
    test_data = MarketData(index=actual.index, resolution_minutes=data.resolution_minutes,
                           spot_price=actual, currency=data.currency)
    days = (actual.index[-1] - actual.index[0]).total_seconds() / 86400 or 1

    ann = 365 / days
    base = ThresholdArbitrageOptimizer(25, 75).optimize(asset, test_data).revenue.net
    fc = forecast_driven_dispatch(asset, gbm_pred, actual, dt)
    perfect = MILPDispatchOptimizer(window_days=30).optimize(asset, test_data).revenue.net

    print(f"\n  dispatch net P&L ({data.currency}/MW/yr):")
    print(f"    baseline (threshold/actual) : {base*ann:>10,.0f}")
    print(f"    forecast (LP/forecast→actual): {fc['net']*ann:>10,.0f}   <- realistic")
    print(f"    perfect  (LP/actual)         : {perfect*ann:>10,.0f}   (ceiling)")
    capture = 100 * fc["net"] / perfect if perfect else float("nan")
    print(f"  forecast captures {capture:.0f}% of the perfect-foresight ceiling.")

    # Does weather pay? Train a weather-augmented GBM on the same split and report
    # the MAE and dispatch-P&L lift vs the weather-free model above.
    if weather is not None:
        Xw, yw = build_features(series, weather)
        tw, sw = Xw.index < cutoff, Xw.index >= cutoff
        added = [c for c in Xw.columns if c not in X.columns]
        if sw.sum() and tw.sum():
            wmodel = GBMForecaster().fit(Xw[tw], yw[tw])
            w_pred = wmodel.predict(Xw[sw])
            m_w = metrics(yw[sw], w_pred)
            w_fc = forecast_driven_dispatch(asset, w_pred, yw[sw], dt)
            mae_lift = 100 * (1 - m_w["mae"] / m_gbm["mae"])
            pnl_lift = 100 * (w_fc["net"] / fc["net"] - 1) if fc["net"] else float("nan")
            print(f"\n  + weather ({len(added)} features: {', '.join(added)}):")
            print(f"    forecast MAE  — weather {m_w['mae']:.2f} vs no-weather {m_gbm['mae']:.2f} "
                  f"({mae_lift:+.0f}%)")
            print(f"    dispatch net  — weather {w_fc['net']*ann:>10,.0f} vs "
                  f"no-weather {fc['net']*ann:>10,.0f} ({pnl_lift:+.1f}%)")
            # Which weather variables actually move price? Permutation importance =
            # the MAE rise on the test set when each feature alone is shuffled.
            from sklearn.inspection import permutation_importance
            imp = permutation_importance(
                wmodel.model, Xw[sw], yw[sw], n_repeats=5, random_state=0,
                scoring="neg_mean_absolute_error")
            ranked = sorted(((c, imp.importances_mean[i]) for i, c in enumerate(Xw.columns)
                             if c in added), key=lambda kv: kv[1], reverse=True)
            print(f"    weather feature importance (MAE rise when shuffled, {data.currency}/MWh):")
            for name, val in ranked:
                print(f"      {name:12}{val:6.2f}")
        else:
            print("\n  + weather: present but doesn't cover the test split — skipping.")
    else:
        print("\n  + weather: no rows in `weather` table — run fetch_weather.py to enable.")

    # Probabilistic forecast + robust (risk-aware) dispatch.
    qf = QuantileForecaster().fit(X[train], y[train])
    q_test = qf.predict(X[test])
    cov = qf.coverage(X[test], actual)
    print(f"\n  quantile calibration: actuals inside P10–P90 {100*cov:.0f}% of the time "
          f"(target 80%)")

    rob_rob = robust_dispatch(asset, q_test, actual, dt, beta=1.0)  # marginal-quantile (naive)

    def _risk(res):
        p = np.array(res["daily_pnl"])
        return res["net"] * ann, p.std(), p.min()

    # Joint scenarios: error day-shapes from an out-of-sample validation slice.
    val_cut = cutoff - pd.Timedelta(days=90)
    tr2 = X.index < val_cut
    val = (X.index >= val_cut) & (X.index < cutoff)
    val_pred = GBMForecaster().fit(X[tr2], y[tr2]).predict(X[val])
    blocks = daily_residual_blocks(y[val], val_pred, steps=int(round(24 / dt)))
    sc_exp = scenario_robust_dispatch(asset, gbm_pred, blocks, actual, dt, beta=0.0)
    sc_rob = scenario_robust_dispatch(asset, gbm_pred, blocks, actual, dt, beta=1.0)

    print(f"\n  robust dispatch — risk vs return ({data.currency}):")
    print(f"    {'strategy':30}{'net/MW/yr':>12}{'daily σ':>10}{'worst day':>11}")
    rows = [
        ("point forecast", fc),
        ("marginal-quantile  β=1", rob_rob),
        (f"joint-scenario ({blocks.shape[0]}d)  β=0", sc_exp),
        (f"joint-scenario ({blocks.shape[0]}d)  β=1", sc_rob),
    ]
    for label, res in rows:
        a, s, w = _risk(res)
        print(f"    {label:30}{a:>12,.0f}{s:>10,.0f}{w:>11,.0f}")
    print("\n  (marginal = independent per-hour quantiles glued together;")
    print("   joint = whole-day error shapes → realistic, correlated scenarios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
