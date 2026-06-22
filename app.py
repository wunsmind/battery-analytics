#!/usr/bin/env python3
"""Plotly Dash dashboard for electricity spot prices.

    python app.py   ->   http://127.0.0.1:8050

EUR-first: defaults to ENTSO-E bidding-zone day-ahead prices (EUR/MWh, the
wholesale market price). Tibber home prices (the local consumer price) are
available as a secondary source. Shows the price curve and the daily min/max
spread — the arbitrage potential a battery could capture.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dotenv import load_dotenv

from pricing import DEFAULT_SEK_PER_EUR, build_breakdown
from store.db import (
    list_homes,
    list_resolutions,
    list_zone_resolutions,
    list_zones,
    load_prices,
    load_zone_prices,
)

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "prices.db")            # Tibber home prices (tracked)
ZONE_DB_PATH = os.getenv("ZONE_DB_PATH", "market.db")  # ENTSO-E zones (reproducible, ignored)
SEK_PER_EUR = float(os.getenv("SEK_PER_EUR", DEFAULT_SEK_PER_EUR))

# Source-specific config: how to load, which column holds the price, and the unit.
ZONE = "zone"
TIBBER = "tibber"
BREAKDOWN = "breakdown"
TIBBER_METRICS = {
    "energy": "Spot / energy",
    "total": "Total (energy + tax)",
    "tax": "Tax & fees",
}

app = Dash(__name__, title="Battery Analytics — Spot Prices")


def _daily_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-day min/max/mean/spread for `col`, grouped by local calendar day."""
    if df.empty:
        return df
    local = df.copy()
    tz = local["starts_at"].dt.tz
    local["day"] = (local["starts_at"].dt.tz_convert(tz).dt.date if tz is not None
                    else local["starts_at"].dt.date)
    stats = local.groupby("day")[col].agg(["min", "max", "mean"]).reset_index()
    stats["spread"] = stats["max"] - stats["min"]
    return stats.sort_values("day")


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H1("⚡ Battery Analytics — Spot Prices"),
        html.P(
            "ENTSO-E day-ahead by bidding zone (EUR/MWh) by default; Tibber home price "
            "as a secondary source; or the EUR Breakdown decomposing the consumer total "
            "into wholesale + Tibber markup + tax & fees.",
            style={"color": "#666"},
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div([
                    html.Label("Source"),
                    dcc.Dropdown(
                        id="source",
                        options=[
                            {"label": "ENTSO-E zone (EUR)", "value": ZONE},
                            {"label": "Tibber home (local)", "value": TIBBER},
                            {"label": "Breakdown (EUR): wholesale + markup + tax", "value": BREAKDOWN},
                        ],
                        value=ZONE,
                        clearable=False,
                        style={"width": "220px"},
                    ),
                ]),
                html.Div([
                    html.Label("Zone / Home"),
                    dcc.Dropdown(id="series", clearable=False, style={"width": "240px"}),
                ]),
                html.Div([
                    html.Label("Metric (Tibber)"),
                    dcc.Dropdown(
                        id="metric",
                        options=[{"label": v, "value": k} for k, v in TIBBER_METRICS.items()],
                        value="energy",
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ]),
                html.Div([
                    html.Label("Resolution"),
                    dcc.Dropdown(id="resolution", clearable=False, style={"width": "180px"}),
                ]),
            ],
        ),
        html.Div(id="kpis", style={"display": "flex", "gap": "24px", "margin": "20px 0"}),
        dcc.Graph(id="price-curve"),
        html.H3("Daily min / max spread (battery arbitrage potential)"),
        dcc.Graph(id="spread-chart"),
        dcc.Interval(id="tick", interval=60_000, n_intervals=0),
    ],
)


@app.callback(
    Output("series", "options"),
    Output("series", "value"),
    Output("metric", "disabled"),
    Input("source", "value"),
    Input("tick", "n_intervals"),
    Input("series", "value"),
)
def _populate_series(source, _n, current):
    if source in (ZONE, BREAKDOWN):
        items = list_zones(ZONE_DB_PATH)
        # Prefer SE_3 (Stockholm) as a sensible default if present.
        default = "SE_3" if "SE_3" in items else (items[0] if items else None)
    else:
        items = list_homes(DB_PATH)
        default = items[0] if items else None
    options = [{"label": i, "value": i} for i in items]
    value = current if current in items else default
    return options, value, (source != TIBBER)


@app.callback(
    Output("resolution", "options"),
    Output("resolution", "value"),
    Input("source", "value"),
    Input("tick", "n_intervals"),
    Input("resolution", "value"),
)
def _populate_resolutions(source, _n, current):
    res = (list_zone_resolutions(ZONE_DB_PATH) if source in (ZONE, BREAKDOWN)
           else list_resolutions(DB_PATH))
    res = res or ["HOURLY"]
    options = [{"label": r.replace("_", "-").title(), "value": r} for r in res]
    value = current if current in res else res[0]
    return options, value


def _kpi_card(label: str, value: str, color: str = "#111"):
    return html.Div(
        style={"padding": "12px 16px", "background": "#f5f5f7", "borderRadius": "10px",
               "minWidth": "140px"},
        children=[
            html.Div(label, style={"fontSize": "12px", "color": "#888"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": 600, "color": color}),
        ],
    )


@app.callback(
    Output("price-curve", "figure"),
    Output("spread-chart", "figure"),
    Output("kpis", "children"),
    Input("source", "value"),
    Input("series", "value"),
    Input("metric", "value"),
    Input("resolution", "value"),
    Input("tick", "n_intervals"),
)
def _update(source, series, metric, resolution, _n):
    empty = go.Figure().update_layout(
        annotations=[dict(text="No data yet — run fetch_entsoe.py / fetch.py",
                          showarrow=False, font=dict(size=16))]
    )
    if not series:
        return empty, empty, [_kpi_card("Status", "no data")]

    if source == BREAKDOWN:
        df = build_breakdown(DB_PATH, ZONE_DB_PATH, zone=series,
                             resolution=resolution, sek_per_eur=SEK_PER_EUR)
        if df.empty:
            return empty, empty, [_kpi_card("Status",
                                            f"no Tibber↔{series} overlap at {resolution}")]
        labels = {"spot": "Wholesale spot", "markup": "Tibber markup", "tax": "Tax & fees"}
        long = df.melt(id_vars="starts_at", value_vars=list(labels),
                       var_name="component", value_name="val")
        long["component"] = long["component"].map(labels)
        area = px.area(long, x="starts_at", y="val", color="component",
                       labels={"starts_at": "", "val": "EUR/MWh"})
        area.add_scatter(x=df["starts_at"], y=df["total"], name="Total",
                         mode="lines", line=dict(color="black", dash="dash"))
        area.update_layout(title=f"Consumer price breakdown vs {series} wholesale (EUR/MWh)",
                           hovermode="x unified", margin=dict(t=50, b=20))

        means = df[["spot", "markup", "tax", "total"]].mean()
        def _share(c):
            return 100 * means[c] / means["total"] if means["total"] else 0.0
        stats = _daily_stats(df, "total")
        spread = go.Figure()
        spread.add_bar(x=stats["day"], y=stats["spread"], name="total spread (max-min)")
        spread.add_scatter(x=stats["day"], y=stats["mean"], name="daily mean total",
                           mode="lines+markers")
        spread.update_layout(title="Daily total spread & mean (EUR/MWh)", margin=dict(t=50, b=20))
        kpis = [
            _kpi_card("Avg total", f"{means['total']:.1f} EUR/MWh"),
            _kpi_card("Wholesale", f"{_share('spot'):.0f}%", "#1f77b4"),
            _kpi_card("Markup", f"{_share('markup'):.0f}%", "#ff7f0e"),
            _kpi_card("Tax & fees", f"{_share('tax'):.0f}%", "#2ca02c"),
            _kpi_card("FX SEK/EUR", f"{SEK_PER_EUR:.2f}"),
        ]
        return area, spread, kpis

    if source == ZONE:
        df = load_zone_prices(ZONE_DB_PATH, zone=series, resolution=resolution)
        col, unit, title = "price", "EUR/MWh", f"{series} day-ahead"
        fmt = "{:,.1f}"
    else:
        df = load_prices(DB_PATH, home_id=series, resolution=resolution)
        col = metric
        cur = (df["currency"].dropna().iloc[0] if not df.empty and df["currency"].notna().any()
               else "")
        unit, title = f"{cur}/kWh", f"{TIBBER_METRICS[metric]} ({series})"
        fmt = "{:,.3f}"
    if df.empty:
        return empty, empty, [_kpi_card("Status", "no data for selection")]

    curve = px.line(df, x="starts_at", y=col, labels={"starts_at": "", col: unit})
    curve.update_layout(title=f"{title} ({unit})", hovermode="x unified",
                        margin=dict(t=50, b=20))

    stats = _daily_stats(df, col)
    spread = go.Figure()
    spread.add_bar(x=stats["day"], y=stats["spread"], name="spread (max-min)")
    spread.add_scatter(x=stats["day"], y=stats["mean"], name="daily mean", mode="lines+markers")
    spread.update_layout(title=f"Daily spread & mean ({unit})", barmode="overlay",
                         margin=dict(t=50, b=20))

    latest = stats.iloc[-1]
    kpis = [
        _kpi_card("Intervals stored", f"{len(df):,}"),
        _kpi_card("Days tracked", f"{stats.shape[0]:,}"),
        _kpi_card("Latest spread", fmt.format(latest["spread"]), "#0a7d33"),
        _kpi_card("Latest max", fmt.format(latest["max"])),
        _kpi_card("Latest min", fmt.format(latest["min"])),
    ]
    return curve, spread, kpis


if __name__ == "__main__":
    app.run(debug=True)
