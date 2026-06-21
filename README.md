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
```

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

This Python core is designed to grow into a battery-optimisation backend:
1. **Ingestion** — add an ENTSO-E ingester (free API; day-ahead prices back to
   2015 for all SE/NO/DK/FI/Baltic zones). Request API access by emailing
   `transparency@entsoe.eu` ("Restful API access"); client lib: `entsoe-py`.
   Then add balancing/imbalance market feeds.
2. **Forecasting** — predict next-day prices (statsmodels / ML)
3. **Optimisation** — battery dispatch via MILP (PuLP / Pyomo / cvxpy) to maximise
   arbitrage + grid-service revenue
4. **API + product UI** — expose results over an API; add a Next.js frontend for
   customers when needed
