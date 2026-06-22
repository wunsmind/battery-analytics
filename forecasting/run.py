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

import pandas as pd
from dotenv import load_dotenv

from forecasting.backtest import forecast_driven_dispatch, metrics
from forecasting.features import build_features
from forecasting.models import GBMForecaster, seasonal_naive
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

    base = ThresholdArbitrageOptimizer(25, 75).optimize(asset, test_data).revenue.net
    fcast = forecast_driven_dispatch(asset, gbm_pred, actual, dt)["net"]
    perfect = MILPDispatchOptimizer(window_days=30).optimize(asset, test_data).revenue.net

    print(f"\n  dispatch net P&L ({data.currency}/MW/yr):")
    print(f"    baseline (threshold/actual) : {base*365/days:>10,.0f}")
    print(f"    forecast (LP/forecast→actual): {fcast*365/days:>10,.0f}   <- realistic")
    print(f"    perfect  (LP/actual)         : {perfect*365/days:>10,.0f}   (ceiling)")
    capture = 100 * fcast / perfect if perfect else float("nan")
    print(f"\n  forecast captures {capture:.0f}% of the perfect-foresight ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
