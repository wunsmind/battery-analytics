#!/usr/bin/env python3
"""Tests for the EUR price breakdown (pricing.compose_breakdown).

    python -m tests.test_pricing
"""

from __future__ import annotations

import pandas as pd

from pricing import compose_breakdown


def _frames():
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    tibber = pd.DataFrame({
        "starts_at": idx,
        "energy": [1.0, 1.2, 0.8, 1.0],   # SEK/kWh
        "tax": [0.5, 0.5, 0.5, 0.5],
        "total": [1.5, 1.7, 1.3, 1.5],
    })
    zone = pd.DataFrame({
        "starts_at": idx,
        "price": [80.0, 100.0, 60.0, 90.0],  # EUR/MWh wholesale
    })
    return tibber, zone


def test_components_sum_to_total():
    tib, zon = _frames()
    df = compose_breakdown(tib, zon, sek_per_eur=10.0)
    summed = df["spot"] + df["markup"] + df["tax"]
    assert (abs(summed - df["total"]) < 1e-9).all()


def test_sek_to_eur_conversion():
    tib, zon = _frames()
    df = compose_breakdown(tib, zon, sek_per_eur=10.0)
    # 1.0 SEK/kWh at 10 SEK/EUR -> 100 EUR/MWh
    assert abs(df.loc[0, "tibber_energy"] - 100.0) < 1e-9
    assert abs(df.loc[0, "tax"] - 50.0) < 1e-9
    # markup = tibber_energy - spot = 100 - 80 = 20
    assert abs(df.loc[0, "markup"] - 20.0) < 1e-9


def test_inner_join_drops_unmatched():
    tib, zon = _frames()
    zon = zon.iloc[:2]  # only first two timestamps have wholesale data
    df = compose_breakdown(tib, zon, sek_per_eur=10.0)
    assert len(df) == 2


def test_empty_inputs():
    assert compose_breakdown(pd.DataFrame(), pd.DataFrame()).empty


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
