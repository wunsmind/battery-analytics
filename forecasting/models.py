"""Forecast models: a seasonal-naive baseline and a gradient-boosting forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def seasonal_naive(series: pd.Series, index: pd.DatetimeIndex, lag: int = 168) -> pd.Series:
    """Predict each timestamp as the price `lag` hours earlier (default 1 week).

    The classic electricity baseline — captures strong weekly seasonality. A
    good forecaster must beat this to earn its keep.
    """
    pred = series.astype(float).shift(lag).reindex(index)
    return pred


class GBMForecaster:
    """Gradient-boosting regressor over calendar + lag features.

    Fit on a train slice, predict any feature matrix. Uses actual lagged prices
    as inputs (realized history available at forecast time), so there's no
    future leakage as long as features come from build_features.
    """

    def __init__(self, **kwargs):
        params = dict(max_depth=6, learning_rate=0.05, max_iter=400,
                      l2_regularization=1.0, random_state=0)
        params.update(kwargs)
        self.model = HistGradientBoostingRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMForecaster":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)


class QuantileForecaster:
    """Probabilistic forecaster: one gradient-boosting model per quantile.

    Produces a price distribution (e.g. P10/P50/P90) per timestamp — the input a
    robust/risk-aware dispatch needs to know *how uncertain* each hour is. Quantile
    crossings (q-low above q-high) are corrected by sorting per row.
    """

    def __init__(self, quantiles=(0.1, 0.5, 0.9), **kwargs):
        self.quantiles = tuple(quantiles)
        base = dict(max_depth=6, learning_rate=0.05, max_iter=400,
                    l2_regularization=1.0, random_state=0)
        base.update(kwargs)
        self.models = {
            q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **base)
            for q in self.quantiles
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster":
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {f"q{int(q * 100)}": self.models[q].predict(X) for q in self.quantiles}
        df = pd.DataFrame(cols, index=X.index)
        # enforce monotone quantiles row-wise (guard against quantile crossing)
        df[:] = np.sort(df.to_numpy(), axis=1)
        return df

    def coverage(self, X: pd.DataFrame, y: pd.Series, lo: float = 0.1, hi: float = 0.9) -> float:
        """Fraction of actuals inside the [lo, hi] quantile band — should ≈ hi-lo."""
        pred = self.predict(X)
        lo_col, hi_col = f"q{int(lo * 100)}", f"q{int(hi * 100)}"
        inside = (y >= pred[lo_col]) & (y <= pred[hi_col])
        return float(inside.mean())
