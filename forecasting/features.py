"""Feature engineering for day-ahead price forecasting.

Features use only information available at gate closure (no future leakage):
calendar fields of the target time, and *same-hour lags* of 1/2/7 days plus the
previous day's mean. Calendar fields use local (Europe/Stockholm) time so the
model sees true daily/weekly human demand patterns.

Caveat: lag-24h is a mild simplification — at the 12:00 CET day-ahead gate you
don't yet have the full previous calendar day. The bias is small and standard in
electricity-price-forecasting baselines; lag-48/168 are always safe.
"""

from __future__ import annotations

import pandas as pd

LAGS = (24, 48, 168)  # hours = same hour 1, 2, 7 days back
LOCAL_TZ = "Europe/Stockholm"


def build_features(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned and NaN-dropped for an hourly price series."""
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

    df = X.join(s.rename("y")).dropna()
    return df.drop(columns="y"), df["y"]
