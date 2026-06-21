"""Minimal Tibber GraphQL client for fetching spot/energy prices.

Tibber exposes hourly prices for *today* and *tomorrow* only (tomorrow's are
published around 13:00 CET once Nord Pool day-ahead clears). Each hour is split
into:
    total  = energy + tax        (what you actually pay per kWh)
    energy = the spot/wholesale component (closest to the Nord Pool spot price)
    tax    = taxes, grid fees, VAT
We persist all three so the analytics layer can separate market price from levies.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

API_URL = "https://api.tibber.com/v1-beta/gql"

# Pulls every home on the account and both price windows in a single round-trip.
PRICE_QUERY = """
{
  viewer {
    homes {
      id
      appNickname
      address { address1 city }
      currentSubscription {
        priceInfo {
          today    { startsAt total energy tax level currency }
          tomorrow { startsAt total energy tax level currency }
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class PriceHour:
    home_id: str
    starts_at: str  # ISO-8601 with timezone offset, e.g. 2026-06-21T00:00:00+02:00
    total: float | None
    energy: float | None
    tax: float | None
    level: str | None  # Tibber price level, e.g. CHEAP / NORMAL / EXPENSIVE
    currency: str | None


class TibberError(RuntimeError):
    pass


def _post(token: str, query: str, timeout: int = 30) -> dict:
    resp = requests.post(
        API_URL,
        json={"query": query},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code == 401:
        raise TibberError("401 Unauthorized — check TIBBER_TOKEN.")
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise TibberError(f"GraphQL errors: {payload['errors']}")
    return payload["data"]


def fetch_prices(token: str) -> list[PriceHour]:
    """Return all available hourly prices (today + tomorrow) across all homes."""
    data = _post(token, PRICE_QUERY)
    homes = data.get("viewer", {}).get("homes") or []
    out: list[PriceHour] = []
    for home in homes:
        home_id = home["id"]
        sub = home.get("currentSubscription") or {}
        info = sub.get("priceInfo") or {}
        for window in ("today", "tomorrow"):
            for h in info.get(window) or []:
                out.append(
                    PriceHour(
                        home_id=home_id,
                        starts_at=h["startsAt"],
                        total=h.get("total"),
                        energy=h.get("energy"),
                        tax=h.get("tax"),
                        level=h.get("level"),
                        currency=h.get("currency"),
                    )
                )
    return out
