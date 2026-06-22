#!/usr/bin/env python3
"""Tests for the forecasting layer.

    python -m tests.test_forecasting
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.backtest import forecast_driven_dispatch, metrics
from forecasting.features import LAGS, build_features
from forecasting.models import GBMForecaster, seasonal_naive
from optimizer import BatteryAsset
from optimizer.degradation import ThroughputDegradationModel


def _series(n: int = 24 * 60) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    # daily + weekly cycle + noise -> learnable structure
    hours = np.arange(n)
    daily = 50 + 40 * np.sin(2 * np.pi * (hours % 24) / 24)
    weekly = 10 * np.sin(2 * np.pi * (hours % 168) / 168)
    rng = np.random.default_rng(0)
    return pd.Series(daily + weekly + rng.normal(0, 3, n), index=idx)


def test_features_no_leakage_and_shape():
    s = _series()
    X, y = build_features(s)
    assert list(X.columns) == ["hour", "dow", "month", "is_weekend",
                               *[f"lag{l}" for l in LAGS], "prevday_mean"]
    # first max(LAGS) rows dropped due to lags
    assert len(X) == len(s) - max(LAGS)
    assert not X.isna().any().any()


def test_gbm_beats_seasonal_naive():
    s = _series()
    X, y = build_features(s)
    cut = int(len(X) * 0.7)
    model = GBMForecaster(max_iter=100).fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    naive = seasonal_naive(s, X.index[cut:])
    assert metrics(y.iloc[cut:], pred)["mae"] < metrics(y.iloc[cut:], naive)["mae"]


def test_forecast_driven_never_exceeds_perfect():
    s = _series()
    asset = BatteryAsset(
        name="t", energy_capacity_mwh=2.0, power_max_mw=1.0, round_trip_efficiency=0.9,
        soc_min_frac=0.0, soc_max_frac=1.0, currency="EUR",
        degradation=ThroughputDegradationModel(pack_cost=500.0, lifetime_throughput_mwh=100.0),
    )
    # Perfect = dispatch on actual; forecast-driven on actual must equal it.
    perfect = forecast_driven_dispatch(asset, s, s, dt=1.0)["net"]
    noisy = s + np.random.default_rng(1).normal(0, 20, len(s))
    fcast = forecast_driven_dispatch(asset, noisy, s, dt=1.0)["net"]
    assert fcast <= perfect + 1e-6  # imperfect forecast can't beat perfect foresight


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
