"""Markets & products — energy + Nordic reserves, uniform so aFRR slots in.

Every revenue stream is a `Product` distinguished by its *mechanism* (how you get
paid), direction, and physical requirements. Adding SE aFRR when Svenska kraftnät
joins PICASSO is just flipping `available=True` and supplying its price series —
no schema change. The (future) MILP allocates the battery's power + energy
headroom across whichever products are available, maximizing total payment minus
degradation and efficiency losses.

Units convention everywhere downstream:
  energy price      -> currency / MWh
  capacity price    -> currency / MW / hour
  activation price  -> currency / MWh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Mechanism(Enum):
    ENERGY = "energy"               # spot / intraday: buy low, sell high
    CAPACITY_RESERVE = "capacity"   # paid per MW/h to stand ready
    ACTIVATION_ENERGY = "activation"  # paid per MWh when actually activated


class Direction(Enum):
    UP = "up"            # discharge / reduce load on activation
    DOWN = "down"        # charge / increase load on activation
    SYMMETRIC = "symmetric"


@dataclass(frozen=True)
class Product:
    """A market product the battery can be allocated to."""

    id: str
    name: str
    mechanism: Mechanism
    direction: Direction
    resolution_minutes: int                  # market time unit
    min_bid_mw: float = 0.1                   # NOTE: per-service minimums are unsettled;
    bid_granularity_mw: float = 0.1          #       1 MW is confirmed only for mFRR/aFRR
    endurance_minutes: float | None = None    # sustain requirement -> energy headroom needed
    has_activation_market: bool = False       # reserve also paid for activated energy (aFRR/mFRR)
    typical_activation_fraction: float = 0.0  # E[activated energy]/committed capacity (throughput/wear)
    available: bool = True                     # False until prices/market exist (e.g. SE aFRR)
    notes: str = ""


# Nordic product catalogue. Reserve specifics are approximate and must be checked
# against current Svenska kraftnät technical requirements before trading.
NORDIC_PRODUCTS: dict[str, Product] = {
    p.id: p
    for p in [
        Product("spot", "Day-ahead spot", Mechanism.ENERGY, Direction.SYMMETRIC,
                resolution_minutes=60, notes="15-min MTU since 2025-10-01"),
        Product("intraday", "Intraday continuous", Mechanism.ENERGY, Direction.SYMMETRIC,
                resolution_minutes=15),
        Product("fcr_n", "FCR-N", Mechanism.CAPACITY_RESERVE, Direction.SYMMETRIC,
                resolution_minutes=60, endurance_minutes=60,
                typical_activation_fraction=0.10,
                notes="Symmetric; continuous regulation around 50 Hz"),
        Product("fcr_d_up", "FCR-D up", Mechanism.CAPACITY_RESERVE, Direction.UP,
                resolution_minutes=60, endurance_minutes=20,
                notes="Saturated/price-collapsed as of 2025"),
        Product("fcr_d_down", "FCR-D down", Mechanism.CAPACITY_RESERVE, Direction.DOWN,
                resolution_minutes=60, endurance_minutes=20),
        Product("ffr", "FFR", Mechanism.CAPACITY_RESERVE, Direction.UP,
                resolution_minutes=60, endurance_minutes=5,
                notes="Fast frequency reserve; seasonal (low-inertia)"),
        # aFRR — near-greenfield for SE batteries; OFF until SvK joins PICASSO (~2026).
        Product("afrr_up", "aFRR up", Mechanism.CAPACITY_RESERVE, Direction.UP,
                resolution_minutes=15, min_bid_mw=1.0, bid_granularity_mw=1.0,
                has_activation_market=True, typical_activation_fraction=0.15,
                available=False, notes="SE 0 MW battery prequalified Jan-2025; PICASSO pending"),
        Product("afrr_down", "aFRR down", Mechanism.CAPACITY_RESERVE, Direction.DOWN,
                resolution_minutes=15, min_bid_mw=1.0, bid_granularity_mw=1.0,
                has_activation_market=True, typical_activation_fraction=0.15,
                available=False, notes="SE aFRR opening = primary early-mover wedge"),
        Product("mfrr_up", "mFRR up", Mechanism.CAPACITY_RESERVE, Direction.UP,
                resolution_minutes=15, min_bid_mw=1.0, bid_granularity_mw=1.0,
                has_activation_market=True, typical_activation_fraction=0.05),
        Product("mfrr_down", "mFRR down", Mechanism.CAPACITY_RESERVE, Direction.DOWN,
                resolution_minutes=15, min_bid_mw=1.0, bid_granularity_mw=1.0,
                has_activation_market=True, typical_activation_fraction=0.05),
    ]
}


@dataclass
class MarketData:
    """Time-aligned price inputs for the optimization horizon.

    Only `spot_price` is populated today (from prices.db). Reserve price series stay
    empty until ENTSO-E / Svenska kraftnät feeds are wired — the optimizer simply
    sees no reserve revenue available, which is correct.
    """

    index: pd.DatetimeIndex
    resolution_minutes: int
    spot_price: pd.Series                                    # currency/MWh ("spot" product)
    energy_prices: dict[str, pd.Series] = field(default_factory=dict)      # other energy mkts (intraday)
    capacity_prices: dict[str, pd.Series] = field(default_factory=dict)    # id -> currency/MW/h
    activation_prices: dict[str, pd.Series] = field(default_factory=dict)  # id -> currency/MWh
    currency: str = "SEK"

    @property
    def dt_hours(self) -> float:
        return self.resolution_minutes / 60.0

    def available_products(self) -> list[Product]:
        """Products that are both globally available and have a price series here."""
        out = []
        for p in NORDIC_PRODUCTS.values():
            if not p.available:
                continue
            if p.mechanism is Mechanism.ENERGY:
                if p.id == "spot" or p.id in self.energy_prices:
                    out.append(p)
            elif p.id in self.capacity_prices:
                out.append(p)
        return out

    @classmethod
    def from_prices_db(
        cls,
        db_path: str,
        resolution: str = "HOURLY",
        home_id: str | None = None,
        metric: str = "energy",
    ) -> "MarketData":
        """Build spot-only MarketData from the prices table.

        Tibber prices are currency/kWh; converted to currency/MWh (×1000). The
        `energy` column is the spot/wholesale component (closest to Nord Pool).
        """
        from store.db import load_prices  # local import to avoid cycles

        df = load_prices(db_path, home_id=home_id, resolution=resolution)
        if df.empty:
            raise ValueError("No price data — run fetch.py / backfill.py first.")
        df = df.set_index("starts_at")
        spot = df[metric].astype(float) * 1000.0  # /kWh -> /MWh
        currency = str(df["currency"].dropna().iloc[0]) if df["currency"].notna().any() else "SEK"
        res_minutes = 15 if resolution.upper() == "QUARTER_HOURLY" else 60
        return cls(
            index=spot.index,
            resolution_minutes=res_minutes,
            spot_price=spot,
            currency=currency,
        )

    @classmethod
    def from_zone_prices(
        cls,
        db_path: str,
        zone: str,
        resolution: str = "HOURLY",
    ) -> "MarketData":
        """Build spot MarketData from ENTSO-E zone day-ahead prices.

        Already currency/MWh wholesale (no kWh conversion) — the right input for
        SE3/SE4 arbitrage backtests, vs Tibber's home consumer price.
        """
        from store.db import load_zone_prices  # local import to avoid cycles

        df = load_zone_prices(db_path, zone=zone, resolution=resolution)
        if df.empty:
            raise ValueError(f"No zone prices for {zone}/{resolution} — run fetch_entsoe.py.")
        df = df.set_index("starts_at")
        spot = df["price"].astype(float)
        currency = str(df["currency"].dropna().iloc[0]) if df["currency"].notna().any() else "EUR"
        res_minutes = 15 if resolution.upper() == "QUARTER_HOURLY" else 60
        return cls(
            index=spot.index,
            resolution_minutes=res_minutes,
            spot_price=spot,
            currency=currency,
        )
