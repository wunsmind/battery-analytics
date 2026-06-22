"""Data-derived EUR price breakdown: Tibber consumer price vs ENTSO-E wholesale.

Neither API itemizes the Swedish tax stack, so this stays strictly to what the
data supports. It joins Tibber's home price (energy/tax/total, SEK/kWh) with the
ENTSO-E wholesale spot (EUR/MWh) for a chosen bidding zone, converts everything
to EUR/MWh via a configurable SEK→EUR rate, and decomposes the total:

    total = spot (wholesale) + markup (Tibber premium) + tax (Tibber taxes+fees)

where  markup = tibber_energy − spot, and the three components sum to total.
'markup' can be negative if the compared zone doesn't match the home's zone or
on timing/rounding differences — pick the zone matching your home.
"""

from __future__ import annotations

import pandas as pd

from store.db import load_prices, load_zone_prices

DEFAULT_SEK_PER_EUR = 11.3  # configurable (env SEK_PER_EUR); verify the live rate


def compose_breakdown(
    tibber: pd.DataFrame, zone: pd.DataFrame, sek_per_eur: float = DEFAULT_SEK_PER_EUR
) -> pd.DataFrame:
    """Pure core: join Tibber + zone frames, return EUR/MWh component breakdown.

    tibber: columns starts_at, energy, tax, total (SEK/kWh).
    zone:   columns starts_at, price (EUR/MWh wholesale).
    """
    if tibber.empty or zone.empty:
        return pd.DataFrame()
    f = 1000.0 / sek_per_eur  # SEK/kWh -> EUR/MWh
    t = tibber.set_index("starts_at")
    z = zone.set_index("starts_at")
    df = pd.DataFrame(index=t.index)
    df["tibber_energy"] = t["energy"].astype(float) * f
    df["tax"] = t["tax"].astype(float) * f
    df["total"] = t["total"].astype(float) * f
    df = df.join(z["price"].astype(float).rename("spot"), how="inner")
    df["markup"] = df["tibber_energy"] - df["spot"]
    df = df.reset_index()
    return df[["starts_at", "spot", "markup", "tax", "total", "tibber_energy"]] \
        .sort_values("starts_at").reset_index(drop=True)


def build_breakdown(
    prices_db: str,
    zone_db: str,
    *,
    home_id: str | None = None,
    zone: str = "SE_3",
    resolution: str = "QUARTER_HOURLY",
    sek_per_eur: float = DEFAULT_SEK_PER_EUR,
) -> pd.DataFrame:
    """Load Tibber + ENTSO-E and return the EUR/MWh breakdown (see compose_breakdown)."""
    tib = load_prices(prices_db, home_id=home_id, resolution=resolution)
    zon = load_zone_prices(zone_db, zone=zone, resolution=resolution)
    return compose_breakdown(tib, zon, sek_per_eur=sek_per_eur)
