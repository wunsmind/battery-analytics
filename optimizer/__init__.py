"""Battery dispatch optimization — interfaces and a baseline.

This package defines the *shape* of the optimizer (Phase 2-3 of ROADMAP.md):
- assets.py        battery physical/economic model
- degradation.py   degradation as a marginal cost of throughput (the cost term)
- markets.py       energy + reserve products (FCR/FFR/aFRR/mFRR), aFRR-ready
- dispatch.py      the optimizer interface + revenue breakdown + a naive baseline

The real revenue-stacking MILP lands later; everything here is designed so it
slots in without reshaping the data model. New markets (e.g. SE aFRR when SvK
joins PICASSO) are added by registering a Product and supplying its price series.
"""

from .assets import BatteryAsset
from .degradation import (
    DegradationModel,
    ThroughputDegradationModel,
)
from .dispatch import (
    DispatchOptimizer,
    DispatchResult,
    DispatchSchedule,
    RevenueBreakdown,
    ThresholdArbitrageOptimizer,
)
from .markets import (
    Direction,
    MarketData,
    Mechanism,
    NORDIC_PRODUCTS,
    Product,
)

__all__ = [
    "BatteryAsset",
    "DegradationModel",
    "ThroughputDegradationModel",
    "Direction",
    "MarketData",
    "Mechanism",
    "NORDIC_PRODUCTS",
    "Product",
    "DispatchOptimizer",
    "DispatchResult",
    "DispatchSchedule",
    "RevenueBreakdown",
    "ThresholdArbitrageOptimizer",
]
