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

from forecasting.backtest import forecast_driven_dispatch, metrics, robust_dispatch
from forecasting.features import build_features
from forecasting.models import GBMForecaster, QuantileForecaster, seasonal_naive
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

    # Probabilistic forecast + robust (risk-aware) dispatch.
    qf = QuantileForecaster().fit(X[train], y[train])
    q_test = qf.predict(X[test])
    cov = qf.coverage(X[test], actual)
    print(f"\n  quantile calibration: actuals inside P10–P90 {100*cov:.0f}% of the time "
          f"(target 80%)")

    rob_exp = robust_dispatch(asset, q_test, actual, dt, beta=0.0)
    rob_rob = robust_dispatch(asset, q_test, actual, dt, beta=1.0)

    def _risk(res):
        p = np.array(res["daily_pnl"])
        return res["net"] * ann, p.std(), p.min()

    print(f"\n  robust dispatch — risk vs return ({data.currency}):")
    print(f"    {'strategy':24}{'net/MW/yr':>12}{'daily σ':>10}{'worst day':>11}")
    for label, res in [("point forecast", fc),
                       ("robust β=0 (expected)", rob_exp),
                       ("robust β=1 (max-min)", rob_rob)]:
        a, s, w = _risk(res)
        print(f"    {label:24}{a:>12,.0f}{s:>10,.0f}{w:>11,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
