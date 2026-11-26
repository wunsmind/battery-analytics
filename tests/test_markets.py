#!/usr/bin/env python3
"""Tests for Mimer reserve ingestion + markets wiring.

    python -m pytest tests/test_markets.py

No network: parsing is tested against a captured Mimer CSV fixture, and the DB /
MarketData paths use a temp SQLite file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from mimer_ingest import parse_fcr_csv
from optimizer.markets import MarketData
from store.db import load_reserve_prices, upsert_reserve_prices

# Captured live header (BOM + ';' + decimal comma + trailing ';') and one hour.
FIXTURE_CSV = (
    "﻿"
    "Datum;FCR-N Pris (EUR/MW);Total;SE1 FCRN;SE2 FCRN;SE3 FCRN;SE4 FCRN;DK2 FCRN;"
    "FCR-D upp Pris (EUR/MW);Total FCRD upp;SE1 FCRD upp;SE2 FCRD upp;SE3 FCRD upp;"
    "SE4 FCRD upp;DK2 FCRD upp;FCR-D ned Pris (EUR/MW);Total FCRD ned;SE1 FCRD ned;"
    "SE2 FCRD ned;SE3 FCRD ned;SE4 FCRD ned;DK2 FCRD ned;\n"
    "2026-06-20 00:00:00;18,04;212;11,6;51,7;130,3;12,2;6,2;"
    "2,37;565,3;44,6;61,5;301,7;110,2;47,3;"
    "8,56;574,1;40,6;60,5;326,3;86,2;60,5\n"
    # Mimer appends a "Summa" total footer that must be dropped, not parsed.
    "Summa;;5088;;;;;;;13776;;;;;;;14094;;;;;\n"
)


def test_parse_fcr_csv_shape_and_values():
    rows = parse_fcr_csv(FIXTURE_CSV)
    # 3 products x 5 zones x 1 hour
    assert len(rows) == 15
    by_key = {(r["product_id"], r["zone"]): r for r in rows}

    se3_n = by_key[("fcr_n", "SE_3")]
    assert se3_n["price"] == 18.04            # decimal comma parsed
    assert se3_n["volume_mw"] == 130.3        # per-zone column is a volume
    assert se3_n["price_type"] == "capacity"
    assert se3_n["currency"] == "EUR"
    assert se3_n["source"] == "svk_mimer"

    # Price is the shared Nordic marginal price -> identical across zones.
    assert by_key[("fcr_n", "SE_1")]["price"] == by_key[("fcr_n", "SE_4")]["price"]
    assert by_key[("fcr_d_up", "SE_3")]["price"] == 2.37
    assert by_key[("fcr_d_down", "SE_4")]["volume_mw"] == 86.2

    # Stockholm 00:00 in June (CEST, +02:00) -> 22:00 UTC the previous day.
    ts = pd.Timestamp(se3_n["starts_at"])
    assert ts.tz_convert("UTC") == pd.Timestamp("2026-06-19 22:00", tz="UTC")


def test_parse_empty_returns_no_rows():
    assert parse_fcr_csv("") == []
    assert parse_fcr_csv("﻿\n") == []


def test_db_roundtrip():
    rows = parse_fcr_csv(FIXTURE_CSV)
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "t.db")
        assert upsert_reserve_prices(db, rows) == 15
        # Idempotent: re-upsert overwrites, no duplicate rows.
        assert upsert_reserve_prices(db, rows) == 15
        df = load_reserve_prices(db, zone="SE_3", product_id="fcr_n")
        assert len(df) == 1
        assert df.iloc[0]["price"] == 18.04
        assert df.iloc[0]["volume_mw"] == 130.3


def test_attach_reserves_lights_up_products():
    idx = pd.date_range("2026-06-19 22:00", periods=4, freq="h", tz="UTC")
    md = MarketData(
        index=idx,
        resolution_minutes=60,
        spot_price=pd.Series([50.0, 40.0, 30.0, 45.0], index=idx),
        currency="EUR",
    )
    before = {p.id for p in md.available_products()}
    assert before == {"spot"}             # no reserves yet

    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "t.db")
        upsert_reserve_prices(db, parse_fcr_csv(FIXTURE_CSV))
        md.attach_reserves(db, zone="SE_3")

    assert "fcr_n" in md.capacity_prices
    # Fixture holds one hour (2026-06-20 00:00 Stockholm = 22:00 UTC): that step
    # gets the price; later hours have no published price and must stay NaN —
    # the last price is NOT carried forward past the data.
    fcr_n = md.capacity_prices["fcr_n"]
    assert fcr_n.iloc[0] == 18.04
    assert fcr_n.iloc[1:].isna().all()
    after = {p.id for p in md.available_products()}
    assert {"fcr_n", "fcr_d_up", "fcr_d_down"} <= after
    assert "afrr_up" not in after          # globally available=False until PICASSO


def test_attach_reserves_fills_quarter_hours_within_hour_only():
    # 15-min spot over 2 h; reserve price exists for the first hour only.
    idx = pd.date_range("2026-06-19 22:00", periods=8, freq="15min", tz="UTC")
    md = MarketData(index=idx, resolution_minutes=15,
                    spot_price=pd.Series(50.0, index=idx), currency="EUR")
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "t.db")
        upsert_reserve_prices(db, parse_fcr_csv(FIXTURE_CSV))
        md.attach_reserves(db, zone="SE_3")
    fcr_n = md.capacity_prices["fcr_n"]
    assert (fcr_n.iloc[:4] == 18.04).all()     # all four quarters of the hour
    assert fcr_n.iloc[4:].isna().all()         # next hour: no price, no fill
