#!/usr/bin/env python3
"""Plotly Dash dashboard for Tibber / Nord Pool spot prices.

    python app.py   ->   http://127.0.0.1:8050

Reads from the SQLite DB populated by fetch.py. Shows the hourly price curve,
and daily stats including the min/max spread — the daily arbitrage potential
that a battery could capture (charge cheap, discharge expensive).
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dotenv import load_dotenv

from store.db import list_homes, list_resolutions, load_prices

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "prices.db")

METRICS = {
    "energy": "Spot / energy price",
    "total": "Total price (energy + tax)",
    "tax": "Tax & fees",
}

app = Dash(__name__, title="Battery Analytics — Spot Prices")


def _daily_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-day min/max/mean/spread for the chosen metric, in local (account) time."""
    if df.empty:
        return df
    local = df.copy()
    # Tibber timestamps carry the home's tz offset; group by the local calendar day.
    local["day"] = local["starts_at"].dt.tz_convert(local["starts_at"].dt.tz).dt.date \
        if local["starts_at"].dt.tz is not None else local["starts_at"].dt.date
    g = local.groupby("day")[metric]
    stats = g.agg(["min", "max", "mean"]).reset_index()
    stats["spread"] = stats["max"] - stats["min"]
    return stats.sort_values("day")


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H1("⚡ Battery Analytics — Spot Prices"),
        html.P(
            "Hourly prices from Tibber (Nord Pool day-ahead). History accumulates "
            "every time fetch.py runs.",
            style={"color": "#666"},
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div([
                    html.Label("Home"),
                    dcc.Dropdown(id="home", clearable=False, style={"width": "260px"}),
                ]),
                html.Div([
                    html.Label("Metric"),
                    dcc.Dropdown(
                        id="metric",
                        options=[{"label": v, "value": k} for k, v in METRICS.items()],
                        value="energy",
                        clearable=False,
                        style={"width": "260px"},
                    ),
                ]),
                html.Div([
                    html.Label("Resolution"),
                    dcc.Dropdown(id="resolution", clearable=False, style={"width": "200px"}),
                ]),
            ],
        ),
        html.Div(id="kpis", style={"display": "flex", "gap": "24px", "margin": "20px 0"}),
        dcc.Graph(id="price-curve"),
        html.H3("Daily min / max spread (battery arbitrage potential)"),
        dcc.Graph(id="spread-chart"),
        # Re-read the DB periodically so a background cron fetch shows up live.
        dcc.Interval(id="tick", interval=60_000, n_intervals=0),
    ],
)


@app.callback(
    Output("home", "options"),
    Output("home", "value"),
    Input("tick", "n_intervals"),
    Input("home", "value"),
)
def _populate_homes(_n, current):
    homes = list_homes(DB_PATH)
    options = [{"label": h, "value": h} for h in homes]
    value = current if current in homes else (homes[0] if homes else None)
    return options, value


@app.callback(
    Output("resolution", "options"),
    Output("resolution", "value"),
    Input("tick", "n_intervals"),
    Input("resolution", "value"),
)
def _populate_resolutions(_n, current):
    res = list_resolutions(DB_PATH) or ["HOURLY"]
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
    Input("home", "value"),
    Input("metric", "value"),
    Input("resolution", "value"),
    Input("tick", "n_intervals"),
)
def _update(home, metric, resolution, _n):
    df = load_prices(DB_PATH, home_id=home, resolution=resolution)
    empty = go.Figure().update_layout(
        annotations=[dict(text="No data yet — run: python fetch.py",
                          showarrow=False, font=dict(size=16))]
    )
    if df.empty:
        return empty, empty, [_kpi_card("Status", "no data")]

    cur = (df["currency"].dropna().iloc[0] if df["currency"].notna().any() else "")
    unit = f"{cur}/kWh" if cur else "/kWh"

    curve = px.line(df, x="starts_at", y=metric, markers=False,
                    labels={"starts_at": "", metric: unit})
    curve.update_layout(title=f"{METRICS[metric]} ({unit})", hovermode="x unified",
                        margin=dict(t=50, b=20))

    stats = _daily_stats(df, metric)
    spread = go.Figure()
    spread.add_bar(x=stats["day"], y=stats["spread"], name="spread (max-min)")
    spread.add_scatter(x=stats["day"], y=stats["mean"], name="daily mean", mode="lines+markers")
    spread.update_layout(title=f"Daily spread & mean ({unit})", barmode="overlay",
                         margin=dict(t=50, b=20))

    latest_day = stats.iloc[-1]
    kpis = [
        _kpi_card("Intervals stored", f"{len(df)}"),
        _kpi_card("Days tracked", f"{stats.shape[0]}"),
        _kpi_card(f"Latest spread", f"{latest_day['spread']:.3f}", "#0a7d33"),
        _kpi_card("Latest max", f"{latest_day['max']:.3f}"),
        _kpi_card("Latest min", f"{latest_day['min']:.3f}"),
    ]
    return curve, spread, kpis


if __name__ == "__main__":
    app.run(debug=True)
