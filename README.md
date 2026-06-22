# Battery Analytics — Spot Price Tracker

Tracks Nord Pool spot / energy prices via the **Tibber Developer API** and builds
up a historical time series in SQLite, visualised in a Plotly Dash dashboard.

Tibber only exposes **today + tomorrow** hourly prices, so history is accumulated
by running the fetcher on a schedule (idempotent — safe to re-run).

## Architecture

```
tibber/client.py   GraphQL client -> priceInfo (today + tomorrow)
store/db.py         SQLite store, idempotent upsert on (home_id, starts_at)
fetch.py            run-anytime: fetch -> store   (cron this daily)
app.py              Plotly Dash dashboard: price curve + daily arbitrage spread
```

Each hour is stored as `total`, `energy`, `tax`:
- **energy** — the spot/wholesale component (≈ Nord Pool day-ahead price)
- **total** — what you pay (energy + tax)
- **tax** — taxes, grid fees, VAT

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your token from
# https://developer.tibber.com/settings/access-token
```

No Tibber account? Use the public demo token (already noted in `.env.example`).

## Usage

```bash
python fetch.py     # pull today+tomorrow (HOURLY + QUARTER_HOURLY) — run daily
python backfill.py  # one-off: backfill recent history from Tibber (~31 days)
python app.py       # dashboard at http://127.0.0.1:8050

python fetch_entsoe.py           # ENTSO-E zone day-ahead prices -> market.db
python -m optimizer.example      # baseline dispatch optimizer on stored prices
python -m optimizer.backtest     # baseline vs LP optimizer on zone history
python -m forecasting.run        # forecast + 3-way dispatch backtest
python -m tests.test_optimizer   # tests (also test_milp, test_pricing, test_forecasting)
```

### Forecasting (Phase 1)

`forecasting/` turns the perfect-foresight optimizer into a deployable strategy:
a gradient-boosting day-ahead forecaster (calendar + same-hour lag features, no
leakage) and a backtest that dispatches on the *forecast* and settles on the
*actual*. `forecasting.run` prints the three-way P&L — baseline (lower bound),
forecast-driven (realistic), perfect foresight (ceiling). On SE_3/SE_4 the GBM is
~31% better than seasonal-naive and captures ~67–70% of the perfect-foresight
arbitrage value.

There's also a **probabilistic** path: `QuantileForecaster` (P10/P50/P90) plus
risk-aware dispatch over marginal (`robust_dispatch`) and joint error-shape
(`scenario_robust_dispatch`) scenarios, with a β knob from expected-value to
max-min. Backtests give a clean, theory-consistent result: **β=0 (expected) ≈ the
point forecast, and β=1 (max-min) only sacrifices return** — i.e. risk-averse
dispatch has no value for *pure arbitrage* (re-decided daily, no delivery penalty).
This machinery is parked for the **reserve phase** (FCR/aFRR/mFRR), where
under-delivery is penalized. The real arbitrage lever is forecast *accuracy*
(weather features, next).

### ENTSO-E deep history (wholesale zone prices)

`fetch_entsoe.py` pulls day-ahead prices per bidding zone (EUR/MWh — the "pure"
market price, vs Tibber's home consumer price) into **`market.db`** (`ZONE_DB_PATH`).
Needs `ENTSOE_TOKEN` in `.env`.

Storage split: `prices.db` (Tibber home, small, git-tracked) vs `market.db`
(ENTSO-E zones, large but **reproducible** via this script, so git-ignored). This
keeps the repo small — re-run `fetch_entsoe.py` to rebuild zone history anytime.

```bash
python fetch_entsoe.py                                  # last 30 days, SE_3 + SE_4
python fetch_entsoe.py --start 2015-01-05 --zones SE_1 SE_2 SE_3 SE_4   # deep backfill
```

15-minute since 2025-10-01, hourly before (tagged per row). Backtest on it via
`MarketData.from_zone_prices(db, "SE_4", "QUARTER_HOURLY")`.

The `optimizer/` package implements the dispatch layer (see [ROADMAP.md](ROADMAP.md)
Phase 2–3): `BatteryAsset` + degradation cost model, a `Product`/`MarketData`
catalogue (energy + FCR/FFR/aFRR/mFRR, aFRR gated off until SvK joins PICASSO),
and two optimizers behind one `DispatchOptimizer` interface:
- `ThresholdArbitrageOptimizer` — naive spot-only baseline (lower bound)
- `MILPDispatchOptimizer` — perfect-foresight LP maximizing
  `arbitrage − degradation − efficiency losses` under SoC/power constraints,
  chunked into windows for long horizons (`optimizer/milp.py`)

`optimizer/backtest.py` compares them on real zone history. Reserve
(FCR/aFRR/mFRR) revenue stacking is the next extension to the LP.

`fetch.py` captures **both** resolutions each run. Quarter-hourly is the native
market unit (15-min since 2025-10-01) and Tibber only serves it for ~7 days, so
fetching daily is what builds gap-free 15-min history going forward.

### Backfilling history

`fetch.py` only captures today + tomorrow, so history grows forward from your
first run. `backfill.py` uses Tibber's `priceInfoRange` for an immediate head
start, but Tibber caps the lookback:

| Resolution | Max lookback |
|------------|--------------|
| `HOURLY` (default) | ~31 days (744 intervals) |
| `QUARTER_HOURLY`   | ~7 days (672 intervals) |

```bash
python backfill.py                 # ~31 days hourly
python backfill.py QUARTER_HOURLY  # ~7 days quarter-hourly
```

For **deep history** (back to 2015, every Nordic/Baltic bidding zone) use
ENTSO-E — see the roadmap. Note: the EU day-ahead market moved to **15-minute
resolution on 1 Oct 2025**, so quarter-hourly is now the native market unit.
The `prices` table carries a `resolution` column so HOURLY and QUARTER_HOURLY
data coexist (PK is `(home_id, starts_at, resolution)`); the dashboard has a
resolution selector.

## Scheduling (build history automatically)

Tomorrow's prices publish ~13:00 CET. Fetch once daily after that. Add to `crontab -e`:

```cron
5 14 * * *  cd /Users/andym/Desktop/Github/battery-analytics/battery-analytics && .venv/bin/python fetch.py >> fetch.log 2>&1
```

## Roadmap (toward BESS / flexibility)

This Python core is designed to grow into a battery-optimisation and trading
backend: **ingest → forecast → optimise dispatch → trade**, with a realistic
battery model (degradation + round-trip efficiency + SoC) gating every decision.

See **[ROADMAP.md](ROADMAP.md)** for the full phased plan, including the
degradation-cost model and the (critical) SoC-estimation workstream.

Immediate next step: the **ENTSO-E ingester** (deep history back to 2015, all
Nordic/Baltic zones) — pending API access (email `transparency@entsoe.eu`,
"Restful API access"; client lib `entsoe-py`).
