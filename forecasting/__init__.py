"""Day-ahead price forecasting (Phase 1).

Turns the dispatch optimizer from a perfect-foresight upper bound into a
deployable strategy: forecast tomorrow's prices, dispatch on the forecast, settle
on the actual. Modules:
- features.py  calendar + same-hour lag features (no future leakage)
- models.py    seasonal-naive baseline + gradient-boosting forecaster
- backtest.py  forecast accuracy metrics + forecast-driven dispatch P&L
"""

from .features import build_features
from .models import GBMForecaster, QuantileForecaster, seasonal_naive
from .backtest import (
    forecast_driven_dispatch,
    metrics,
    robust_dispatch,
    scenario_robust_dispatch,
)
from .scenarios import daily_residual_blocks, sample_scenarios

__all__ = [
    "build_features",
    "GBMForecaster",
    "QuantileForecaster",
    "seasonal_naive",
    "forecast_driven_dispatch",
    "robust_dispatch",
    "scenario_robust_dispatch",
    "daily_residual_blocks",
    "sample_scenarios",
    "metrics",
]
