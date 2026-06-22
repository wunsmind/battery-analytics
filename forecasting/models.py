"""Forecast models: a seasonal-naive baseline and a gradient-boosting forecaster."""

from __future__ import annotations

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
