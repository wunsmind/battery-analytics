#!/usr/bin/env python3
"""Tests for the forecasting layer.

    python -m tests.test_forecasting
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.backtest import forecast_driven_dispatch, metrics, robust_dispatch
from forecasting.features import LAGS, build_features
from forecasting.models import GBMForecaster, QuantileForecaster, seasonal_naive
from optimizer import BatteryAsset
from optimizer.degradation import ThroughputDegradationModel


def _eur_asset() -> BatteryAsset:
    return BatteryAsset(
        name="t", energy_capacity_mwh=2.0, power_max_mw=1.0, round_trip_efficiency=0.9,
        soc_min_frac=0.0, soc_max_frac=1.0, currency="EUR",
        degradation=ThroughputDegradationModel(pack_cost=500.0, lifetime_throughput_mwh=100.0),
    )


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


def _weather(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hours = np.arange(len(idx))
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"temp_c": 5 + 10 * np.sin(2 * np.pi * (hours % 24) / 24),
         "wind_100m": 20 + 15 * np.cos(2 * np.pi * hours / 168),
         "solar_rad": np.clip(400 * np.sin(2 * np.pi * (hours % 24) / 24), 0, None),
         "cloud_cover": rng.uniform(0, 100, len(idx)),
         "precip": rng.gamma(0.3, 1.0, len(idx))},
        index=idx,
    )


def test_weather_absent_leaves_columns_unchanged():
    s = _series()
    base_cols = list(build_features(s)[0].columns)
    # explicit None and empty frame must both fall back to the weather-free set
    assert list(build_features(s, None)[0].columns) == base_cols
    assert list(build_features(s, pd.DataFrame())[0].columns) == base_cols


def test_weather_features_added_and_no_nan():
    s = _series()
    X, y = build_features(s, _weather(s.index))
    # all five raw drivers plus the derived 7-day precip sum
    for col in ("temp_c", "wind_100m", "solar_rad", "cloud_cover", "precip", "precip_7d"):
        assert col in X.columns
    assert not X.isna().any().any()
    # weather doesn't change row count vs weather-free (it's fully backfilled)
    assert len(X) == len(build_features(s)[0])


def test_weather_partial_columns_only_add_present():
    # A frame with a subset of drivers adds only those (no precip -> no precip_7d).
    s = _series()
    w = _weather(s.index)[["temp_c", "solar_rad"]]
    X, _ = build_features(s, w)
    assert "solar_rad" in X.columns and "temp_c" in X.columns
    assert "wind_100m" not in X.columns and "precip_7d" not in X.columns


def test_weather_helps_when_price_depends_on_wind():
    # Price driven by wind that is NOT purely periodic (white noise), so the price
    # lags can't recover it — only the weather feature can. Mirrors reality: wind
    # is genuine exogenous information beyond yesterday/last-week's price.
    s = _series()
    rng = np.random.default_rng(11)
    wind = pd.Series(rng.normal(20, 8, len(s)), index=s.index)
    s = s - 1.5 * wind  # wind depresses price, as in SE3/SE4
    Xc, yc = build_features(s)
    Xw, yw = build_features(s, wind.to_frame("wind_100m"))
    cut = int(len(Xc) * 0.7)
    mae_c = metrics(yc.iloc[cut:],
                    GBMForecaster(max_iter=150).fit(Xc.iloc[:cut], yc.iloc[:cut]).predict(Xc.iloc[cut:]))["mae"]
    mae_w = metrics(yw.iloc[cut:],
                    GBMForecaster(max_iter=150).fit(Xw.iloc[:cut], yw.iloc[:cut]).predict(Xw.iloc[cut:]))["mae"]
    assert mae_w < mae_c


def test_openmeteo_to_records_shapes_rows():
    from weather.openmeteo import to_records
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"temp_c": [1.0, np.nan, 3.0], "wind_100m": [10.0, 20.0, np.nan]}, index=idx)
    rows = to_records("SE_3", df)
    assert len(rows) == 3 and rows[0]["zone"] == "SE_3"
    assert rows[0]["starts_at"].endswith("+00:00")
    assert rows[1]["temp_c"] is None and rows[1]["wind_100m"] == 20.0
    # a fully-empty row is dropped
    empty = pd.DataFrame({"temp_c": [np.nan], "wind_100m": [np.nan]}, index=idx[:1])
    assert to_records("SE_3", empty) == []


def _exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hours = np.arange(len(idx))
    return pd.DataFrame(
        {"wind_DE_LU": 8000 + 5000 * np.cos(2 * np.pi * hours / 168),
         "load_SE_4": 2000 + 800 * np.sin(2 * np.pi * (hours % 24) / 24)},
        index=idx,
    )


def test_exog_folds_all_columns_generically():
    s = _series()
    base = list(build_features(s)[0].columns)
    X, _ = build_features(s, exog=_exog(s.index))
    assert "wind_DE_LU" in X.columns and "load_SE_4" in X.columns
    assert not X.isna().any().any()
    # exog is fully backfilled, so it doesn't change the row count
    assert len(X) == len(build_features(s)[0])
    # absent/empty exog falls back to exactly the calendar+lag set
    assert list(build_features(s, exog=None)[0].columns) == base
    assert list(build_features(s, exog=pd.DataFrame())[0].columns) == base


def test_exog_all_nan_column_is_dropped():
    # A column with no data must not survive (it would NaN-drop every row).
    s = _series()
    e = _exog(s.index)
    e["empty"] = np.nan
    X, _ = build_features(s, exog=e)
    assert "empty" not in X.columns and len(X) == len(build_features(s)[0])


def test_weather_and_exog_compose():
    s = _series()
    X, _ = build_features(s, _weather(s.index), _exog(s.index))
    for col in ("temp_c", "precip_7d", "wind_DE_LU", "load_SE_4"):
        assert col in X.columns
    assert not X.isna().any().any()


def test_exog_helps_when_price_depends_on_cross_border_wind():
    # Price depresses with foreign wind (non-periodic) the price lags can't recover.
    s = _series()
    rng = np.random.default_rng(13)
    wind = pd.Series(rng.normal(8000, 2500, len(s)), index=s.index)
    s = s - 0.004 * wind  # ~10 EUR swing, as DE/DK wind drives SE_4
    Xc, yc = build_features(s)
    Xe, ye = build_features(s, exog=wind.to_frame("wind_DE_LU"))
    cut = int(len(Xc) * 0.7)
    mae_c = metrics(yc.iloc[cut:],
                    GBMForecaster(max_iter=150).fit(Xc.iloc[:cut], yc.iloc[:cut]).predict(Xc.iloc[cut:]))["mae"]
    mae_e = metrics(ye.iloc[cut:],
                    GBMForecaster(max_iter=150).fit(Xe.iloc[:cut], ye.iloc[:cut]).predict(Xe.iloc[cut:]))["mae"]
    assert mae_e < mae_c


def test_entsoe_forecasts_to_records():
    from entsoe_forecasts import to_records
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"wind": [100.0, np.nan, 300.0], "load": [2000.0, 2100.0, 2200.0]},
                         index=idx)
    rows = to_records("DE_LU", frame)
    # 3 load + 2 wind (the NaN wind hour is dropped); resolution tagged HOURLY
    assert len(rows) == 5
    assert all(r["resolution"] == "HOURLY" for r in rows)
    assert {r["kind"] for r in rows} == {"wind", "load"}
    assert sum(r["kind"] == "wind" for r in rows) == 2
    assert rows[0]["zone"] == "DE_LU" and rows[0]["starts_at"].endswith("+00:00")
    # quarter-hourly cadence is detected from the index step
    q = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    qrows = to_records("DK_2", pd.DataFrame({"wind": [1.0, 2.0, 3.0, 4.0]}, index=q))
    assert all(r["resolution"] == "QUARTER_HOURLY" for r in qrows)
    assert to_records("DE_LU", pd.DataFrame()) == []


def test_forecasts_wide_pivot_roundtrips():
    import tempfile

    from store.db import load_forecasts_wide, upsert_zone_forecasts
    tmp = tempfile.mkdtemp()
    db = f"{tmp}/t.db"
    rows = [
        {"zone": "DE_LU", "starts_at": "2024-01-01T00:00:00+00:00", "kind": "wind",
         "resolution": "HOURLY", "value": 9000.0, "source": "entsoe"},
        {"zone": "SE_4", "starts_at": "2024-01-01T00:00:00+00:00", "kind": "load",
         "resolution": "HOURLY", "value": 1800.0, "source": "entsoe"},
    ]
    assert upsert_zone_forecasts(db, rows) == 2
    wide = load_forecasts_wide(db, zones=["DE_LU", "SE_4"])
    assert set(wide.columns) == {"wind_DE_LU", "load_SE_4"}
    assert wide.loc[wide.index[0], "wind_DE_LU"] == 9000.0
    # idempotent upsert: re-writing the same keys doesn't duplicate
    upsert_zone_forecasts(db, rows)
    assert len(load_forecasts_wide(db)) == 1


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


def test_quantiles_are_monotone_and_calibrated():
    s = _series()
    X, y = build_features(s)
    cut = int(len(X) * 0.7)
    qf = QuantileForecaster(max_iter=100).fit(X.iloc[:cut], y.iloc[:cut])
    pred = qf.predict(X.iloc[cut:])
    # columns sorted low->high every row
    assert (pred["q10"] <= pred["q50"] + 1e-9).all()
    assert (pred["q50"] <= pred["q90"] + 1e-9).all()
    cov = qf.coverage(X.iloc[cut:], y.iloc[cut:])
    assert 0.5 < cov < 1.0  # roughly calibrated toward 0.8


def test_robust_equals_point_when_no_uncertainty():
    # If all scenarios are identical, robust dispatch == point-forecast dispatch.
    s = _series(24 * 14)
    asset = _eur_asset()
    fcast = s + np.random.default_rng(2).normal(0, 15, len(s))
    q = pd.DataFrame({"q10": fcast, "q50": fcast, "q90": fcast}, index=s.index)
    point = forecast_driven_dispatch(asset, fcast, s, dt=1.0)["net"]
    rob = robust_dispatch(asset, q, s, dt=1.0, beta=1.0)["net"]
    assert abs(point - rob) < 1e-6


def test_residual_blocks_shape():
    s = _series(24 * 10)
    pred = s + 5.0
    from forecasting.scenarios import daily_residual_blocks
    blocks = daily_residual_blocks(s, pred, steps=24)
    assert blocks.shape == (10, 24)
    assert np.allclose(blocks, -5.0)  # residual = actual - pred = -5 everywhere


def test_scenario_robust_equals_point_with_zero_error_blocks():
    # Zero-residual blocks -> every scenario equals the point forecast.
    from forecasting.backtest import scenario_robust_dispatch
    s = _series(24 * 14)
    asset = _eur_asset()
    fcast = s + np.random.default_rng(3).normal(0, 12, len(s))
    blocks = np.zeros((30, 24))
    point = forecast_driven_dispatch(asset, fcast, s, dt=1.0)["net"]
    rob = scenario_robust_dispatch(asset, fcast, blocks, s, dt=1.0, n_scenarios=10, beta=1.0)["net"]
    assert abs(point - rob) < 1e-6


class _PerfectModel:
    """A 'forecaster' holding the full truth series; predict() returns the actual
    price for the rows it's asked about. Used to test the MPC plumbing in
    isolation from forecast error."""

    def __init__(self, truth: pd.Series):
        self.truth = truth.astype(float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.truth.reindex(X.index)


def test_recursive_forecast_with_perfect_model_reproduces_actuals():
    # If the model returns truth, the recursive (block-by-block) forecast must
    # reproduce the actual prices over the whole multi-day horizon.
    from forecasting.mpc import recursive_forecast
    s = _series(24 * 30)
    split = 24 * 20
    history, horizon = s.iloc[:split], s.index[split:split + 48]
    fc = recursive_forecast(_PerfectModel(s), history, horizon, block_steps=24)
    assert len(fc) == 48
    pd.testing.assert_series_equal(fc, s.reindex(horizon), check_names=False)


def test_rolling_horizon_respects_soc_and_loses_to_perfect():
    from forecasting.mpc import rolling_horizon_dispatch
    s = _series(24 * 40)
    asset = _eur_asset()
    X, y = build_features(s)
    cut = int(len(X) * 0.6)
    model = GBMForecaster(max_iter=100).fit(X.iloc[:cut], y.iloc[:cut])
    test_idx = X.index[cut:]
    mpc = rolling_horizon_dispatch(asset, model, s, test_idx, dt=1.0,
                                   lookahead_hours=48, commit_hours=24)
    # Settles on actuals, so net is finite and degradation is non-negative.
    assert mpc["throughput_mwh"] >= 0 and mpc["degradation"] >= -1e-9
    # A real (imperfect) controller can't beat *global* perfect foresight (one LP
    # over the whole span — the true ceiling, not a windowed dispatch).
    actual = y.reindex(test_idx)
    perfect = forecast_driven_dispatch(asset, actual, actual, dt=1.0,
                                       window_hours=len(actual))["net"]
    assert mpc["net"] <= perfect + 1e-6


def test_rolling_horizon_perfect_model_nears_global_ceiling():
    # With a perfect model the receding-horizon controller should approach *global*
    # perfect foresight (its forecasts ARE the actuals), and beat the disjoint-48h
    # windowed dispatch — re-planning every 24h erases most boundary artifacts.
    from forecasting.mpc import rolling_horizon_dispatch
    s = _series(24 * 30)
    asset = _eur_asset()
    X, _ = build_features(s)
    test_idx = X.index[24 * 10:]
    mpc = rolling_horizon_dispatch(asset, _PerfectModel(s), s, test_idx, dt=1.0,
                                   lookahead_hours=48, commit_hours=24)
    actual = s.reindex(test_idx)
    ceiling = forecast_driven_dispatch(asset, actual, actual, dt=1.0,
                                       window_hours=len(actual))["net"]
    disjoint48 = forecast_driven_dispatch(asset, actual, actual, dt=1.0,
                                          window_hours=48)["net"]
    assert mpc["net"] <= ceiling + 1e-6           # can't beat the global LP
    assert mpc["net"] > 0.95 * ceiling            # but lands very close to it
    assert mpc["net"] >= disjoint48 - 1e-6        # and >= disjoint 48h windows


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
