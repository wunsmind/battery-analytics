"""Feature engineering for day-ahead price forecasting.

Features use only information available at gate closure (no future leakage):
calendar fields of the target time, and *same-hour lags* of 1/2/7 days plus the
previous day's mean. Calendar fields use local (Europe/Stockholm) time so the
model sees true daily/weekly human demand patterns.

Optionally folds in weather for the target hour — the largest exogenous day-ahead
drivers in the Nordics: temperature (demand), 100 m wind and solar radiation +
cloud cover (renewable supply), and precipitation (hydro inflow). When weather is
absent the feature set is byte-for-byte the weather-free one, so existing
backtests and tests are unaffected.

Caveats (all small and standard in electricity-price-forecasting baselines):
  - lag-24h: at the 12:00 CET day-ahead gate you don't yet have the full previous
    calendar day. lag-48/168 are always safe.
  - weather: a live forecast uses *forecast* weather for the delivery day; the
    backtest uses the archived actual as a proxy, a mild optimism. Modern weather
    forecasts are accurate a day out, so the gap is small.
  - precipitation: hourly rain at one point is a weak hydro signal — hydro
    responds to basin-wide reservoir levels over weeks. We add a 7-day rolling
    precip sum as a better inflow proxy, but reservoir levels (a slower, separate
    data source) would be stronger; see ROADMAP.
"""

from __future__ import annotations

import pandas as pd

LAGS = (24, 48, 168)  # hours = same hour 1, 2, 7 days back
LOCAL_TZ = "Europe/Stockholm"
# Raw weather columns folded in as-is (those present in the supplied frame).
WEATHER_COLS = ("temp_c", "wind_100m", "solar_rad", "cloud_cover", "precip")


def _fold_onto(idx: pd.DatetimeIndex, frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Reindex `frame[cols]` onto `idx`, interpolating sub-hourly gaps and
    forward/back-filling the edges. Returns only the columns that carry data."""
    f = frame[cols].sort_index()
    f = f[~f.index.duplicated(keep="last")]
    aligned = f.reindex(f.index.union(idx)).interpolate("time").reindex(idx)
    return aligned.loc[:, [c for c in cols if aligned[c].notna().any()]]


def build_features(
    series: pd.Series,
    weather: pd.DataFrame | None = None,
    exog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned and NaN-dropped for an hourly price series.

    weather: optional time-indexed (UTC) DataFrame with temp_c / wind_100m. It is
    reindexed to the price timestamps (interpolated across sub-hourly gaps), so it
    works for both HOURLY and QUARTER_HOURLY series.

    exog: optional time-indexed (UTC) DataFrame of *gate-aligned* exogenous
    drivers (ENTSO-E day-ahead wind/solar/load forecasts — see entsoe_forecasts).
    Every column is folded in generically by name (e.g. wind_DE_LU, load_SE_4),
    using the same reindex/interpolate as weather. Unlike the archived-actual
    weather proxy, these are genuinely knowable at the gate — no look-ahead.
    """
    s = series.astype(float).sort_index()
    idx = s.index
    local = idx.tz_convert(LOCAL_TZ) if idx.tz is not None else idx

    X = pd.DataFrame(index=idx)
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["month"] = local.month
    X["is_weekend"] = (local.dayofweek >= 5).astype(int)
    for lag in LAGS:
        X[f"lag{lag}"] = s.shift(lag)
    X["prevday_mean"] = s.shift(24).rolling(24).mean()

    if weather is not None and not weather.empty:
        present = [c for c in WEATHER_COLS if c in weather.columns]
        aligned = _fold_onto(idx, weather, present)
        for col in aligned.columns:
            X[col] = aligned[col].ffill().bfill()
        if "precip" in aligned.columns:
            # 7-day rolling precipitation — a better hydro-inflow proxy than the
            # mostly-zero hourly value. Time-based window handles HOURLY & 15-min.
            X["precip_7d"] = X["precip"].rolling("7D").sum()

    if exog is not None and not exog.empty:
        aligned = _fold_onto(idx, exog, list(exog.columns))
        for col in aligned.columns:
            X[col] = aligned[col].ffill().bfill()

    df = X.join(s.rename("y")).dropna()
    return df.drop(columns="y"), df["y"]
