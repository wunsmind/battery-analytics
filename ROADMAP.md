# Roadmap — battery-analytics → BESS optimization & trading

From a spot-price tracker to a battery energy storage (BESS) optimization and
flexibility-trading platform (think Flower / Capture Energy). The defensible core
is the Python pipeline: **ingest → forecast → optimize dispatch → trade**, with a
realistic **battery model** (degradation + efficiency + SoC) gating every decision.

Status legend: ✅ done · 🚧 in progress · ⏳ blocked/waiting · ⬜ planned

---

## Guiding principles

- **The battery model is the constraint, not an afterthought.** Dispatch is only
  honest if every cycle is costed (degradation), every conversion is lossy
  (round-trip efficiency), and available energy is *known* (SoC). Design the
  optimizer with these hooks from day one.
- **Data-licensing boundary** (see `data-licensing.md`): ENTSO-E/Tibber data is
  for **internal** R&D/modelling only. Anything customer-facing or traded-on needs
  a commercial license (Nord Pool or a vendor) — *derived signals count*. Keep a
  hard architectural line between internal data and external product.
- **Backtest before live.** Every strategy proves out on historical data (with
  realistic costs) before touching a real asset or a real market.

---

## Phase 0 — Market-data foundation ✅ (mostly)

- ✅ Tibber ingestion (today+tomorrow, HOURLY + QUARTER_HOURLY), idempotent SQLite
- ✅ Tibber ~31-day backfill via `priceInfoRange`
- ✅ Daily cloud cron (GitHub Actions) committing data back
- ✅ Dual-resolution schema; Dash dashboard (price curve + daily arbitrage spread)
- ✅ **ENTSO-E ingester** (`entsoe_ingest.py`, `fetch_entsoe.py`) — day-ahead
  per bidding zone (EUR/MWh) into `zone_prices`; 15-min since Oct-2025 / hourly
  before, tagged per row; backfillable to 2015; `MarketData.from_zone_prices`
  feeds the optimizer. Backtested on SE_3/SE_4.

## Phase 1 — Price intelligence ⬜

- ⬜ Migrate storage to **TimescaleDB/Postgres** when scale/queries demand it
- ⬜ Exploratory analytics: spread distributions, volatility, seasonality,
  intraday shape, zone-vs-zone differentials
- 🚧 **Price forecasting** — day-ahead (`forecasting/`):
  - ✅ Seasonal-naive baseline + gradient-boosting forecaster (calendar +
    same-hour lag features, no leakage). ~31% better MAE than naive on SE_3/SE_4.
  - ✅ **Probabilistic** forecasts: `QuantileForecaster` (P10/P50/P90) + risk-aware
    `robust_dispatch` (marginal) and `scenario_robust_dispatch` (joint error-shape
    scenarios). **Finding:** β=0 (expected) ≈ point forecast (as theory predicts);
    β=1 (max-min) costs return without improving realized worst-case. Conclusion:
    **risk-averse dispatch has no value for pure arbitrage** (re-decided daily, no
    delivery penalty) — the point forecast is the correct objective. This robust
    machinery is **parked for the reserve phase**, where under-delivery is penalized.
  - ✅ **Weather features** (Open-Meteo hourly archive → `weather/openmeteo.py` →
    `fetch_weather.py` → `weather` table) folded into `build_features` with a
    clean weather-free fallback. Six features: temp, 100 m wind, solar radiation,
    cloud cover, precipitation, and a 7-day rolling-precip hydro proxy. **Finding:**
    lowers forecast MAE in both zones (SE_3 −12%, SE_4 −9%); dispatch-P&L lift is
    flat in SE_3 (−0.4%, noise) and **+4.4% in SE_4** — accuracy only pays when it
    sharpens the daily high/low ordering, which it does most where wind swings
    prices hardest. **Permutation importance** (printed by `forecasting.run`)
    ranks the drivers: wind ≫ temp > solar/cloud, and **precipitation is inert**
    (~0 EUR/MWh) — point rainfall is not a hydro signal; hydro responds to
    basin-wide reservoir levels over weeks. *Refinements:* (1) reservoir-level /
    snowpack data (a slower, separate source) for the real hydro lever; (2) one
    representative point per zone today — a load/wind-weighted multi-point average
    would capture more; (3) cross-border weather + ENTSO-E gate-aligned forecasts
    — ✅ **done**, see next bullet.
  - ✅ **Gate-aligned ENTSO-E forecasts** (cross-border wind/solar + load):
    `entsoe_forecasts.py` → `fetch_forecasts.py` → `zone_forecasts` table →
    `build_features(exog=…)`. Pulls the day-ahead *forecasts* a trader actually has
    at the gate — total zonal wind+solar generation and load (MW) — for the target
    zone **and its interconnector neighbours** — both generation *and* load (SE_4 ←
    DE_LU / DK_2 / DK_1; SE_3 ← DK_1). Two reasons this beats the Open-Meteo point
    weather: (a) it's *honest* — the genuine day-ahead vintage, not archived
    **actual** weather (the weather block's mild look-ahead); (b) system-level
    generation/load MW is the quantity that clears the market, and it carries the
    cross-border signal that point weather can't. **Finding** (test on the
    forecast-coverage window, *on top of* weather, so the comparison is
    apples-to-apples): MAE **SE_4 −11% / SE_3 −5%**, dispatch P&L **SE_4 +10.7% /
    SE_3 +6.4%** — and the SE_4 gain is dominated by **German** drivers
    (importance: `solar_DE_LU` 5.2, `wind_SE_4` 4.4, `load_DE_LU` 2.1, `load_SE_4`
    1.5, `wind_DE_LU` 1.4): German demand pulls power out of SE_4 across the cables
    just as German wind floods it in — the interconnector thesis, confirmed twice
    over. SE_3, weakly coupled to Germany, leans on its own load+wind (`load_SE_3`
    6.5, `wind_SE_3` 4.6) with a small Danish (`wind_DK_1` 1.1, `load_DK_1` 1.0)
    contribution. The headline weather→+forecast lift is *additive* and
    *non-optimistic*: the honest signal still pays. (Sweden stopped publishing SE_4
    wind/solar mid-2025; the ingester skips-and-continues, and the deep history
    still trains the feature.)
- ✅ **Forecast-driven backtest** (`forecasting/run.py`): dispatch on forecast,
  settle on actual → realistic P&L (captures ~67–70% of the perfect-foresight
  ceiling on SE_3/SE_4), bracketed by baseline and ceiling.

## Phase 2 — Battery asset model & economics ⬜

The realism layer. Parameterized so any pack (CATL LFP today, anything later) is config.

- 🚧 **Battery config module**: usable capacity, SoC window, max C-rate (charge/
  discharge), round-trip efficiency, cycle-life-vs-DoD curve, calendar curve,
  warranty throughput/cycle limits. *Verify against real CATL datasheet.*
  → scaffolded in `optimizer/assets.py` (`BatteryAsset`, illustrative CATL LFP).
- 🚧 **Round-trip efficiency** in the economic model (for LFP ~90–94%, this often
  dominates the dispatch decision more than degradation) → in `BatteryAsset`.
- 🚧 **Degradation cost model**, staged:
  1. Throughput marginal cost (`€/kWh = pack_cost / lifetime_throughput`) — ✅ done
     (`optimizer/degradation.py`); semi-empirical is a stub
  2. ✅ **Rainflow cycle counting + DoD curve** (`optimizer/rainflow.py`,
     `RainflowDegradationModel`): ASTM E1049 counting of a realized SoC path into
     swings, each priced against a log-log DoD–cycle-life curve
     (`DoDCycleLifeCurve`, illustrative LFP). The LP still optimizes against the
     linear marginal (pinned to the full-DoD anchor, so the two models agree on a
     full-cycle diet); rainflow **re-prices the realized trajectory at settlement**
     — the same forecast-then-settle split used elsewhere. On the illustrative LFP
     curve, five shallow DoD-0.2 cycles cost ~50% of the flat throughput rate while
     deep full cycles stay at 100% — *valuing irregular cycling*, the point of the
     stage. **Datasheet-agnostic:** real CATL cycle-life-vs-DoD numbers drop into
     `DoDCycleLifeCurve` as config. ⏳ *Calibration still gated on the datasheet.*
  3. Semi-empirical: `loss = f_cal(t,T,SoC) + f_cyc(throughput,DoD,C-rate,T)`
- ⬜ **Warranty limits as optimizer constraints** (CATL warranties are throughput/
  cycle-bound — staying inside them is a hard limit, not just a cost)
- ⬜ **SoC estimation strategy** — see dedicated section below (critical)

## Phase 3 — Dispatch optimization ⬜

- ✅ **Optimizer interface + naive baseline** (`optimizer/dispatch.py`):
  `DispatchOptimizer` contract, `RevenueBreakdown`, spot-only
  `ThresholdArbitrageOptimizer` lower bound.
- ✅ **LP arbitrage optimizer** (`optimizer/milp.py`, PuLP/CBC): maximizes
  `arbitrage − degradation − efficiency losses` s.t. SoC/power constraints;
  window-chunked for long horizons. Beats the baseline ~200–350% on SE_3/SE_4.
- ✅ **Perfect-foresight backtest** (`optimizer/backtest.py`) over zone history
  (11 yr in ~11 s). LP = upper bound; real forecast-driven dispatch lands lower.
- ✅ **Rolling-horizon / MPC** dispatch (`forecasting/mpc.py`): at each daily gate,
  forecast a longer lookahead (48h), LP-optimize from realized SoC, **commit only
  the first 24h**, settle on actual, roll SoC, re-plan. The committed day forecasts
  from realized lags (honest — the prior day is cleared at the gate); the lookahead
  *beyond* tomorrow is a **recursive** forecast (the model's day-D predictions feed
  day-(D+1)'s same-hour lags). **Finding:** rolling-horizon lookahead **does not pay
  for daily arbitrage** — SE_4 −0.5%, SE_3 −0.3% vs the fixed 24h-window dispatch.
  Reason: a 2 MWh/1 MW (2-hour) battery runs ~one cycle/day and the optimal pattern
  returns to a similar SoC each midnight, so the 24h boundary was never binding;
  seeing past it adds nothing while the noisier recursive far-horizon forecast costs
  a hair. Same shape as the parked robust-dispatch result — the controller is correct
  architecture, the lookahead just has no value *at 2-hour duration*. **Verified
  crossover by duration** (SE_4, same model, varying capacity at 1 MW): 2h −1.6%,
  4h −1.4%, **8h +6.7%** — lookahead flips positive once storage is long enough that
  multi-day spreads bind and seeing past midnight outweighs the recursive far-horizon
  forecast noise. So the value of MPC is a function of duration; it also returns under
  (a) intraday re-optimization with fresh forecasts and (b) reserve commitments that
  couple SoC across days. Machinery built + tested; **use it for 8h+ assets**, keep
  the simpler 24h-window dispatch for 2h arbitrage.
- ⬜ **Robust/stochastic** optimization for forecast *and* SoC uncertainty
  (SoC buffers so commitments are deliverable even when the estimate is off)
- ⬜ **Backtesting engine**: simulate strategies on history → P&L, cycles,
  degradation, capacity trajectory

## Phase 4 — Real-time operations & control ⬜

- ⬜ Real-time data feeds (live prices, asset telemetry)
- ⬜ **BMS / EMS / inverter integration** (Modbus TCP / SunSpec / REST): live SoC,
  voltage, temperature, power; command interface
- ⬜ **SoC fusion/validation layer** (see below)
- ⬜ Closed-loop dispatch execution with safety constraints + fallback
- ⬜ Monitoring, alerting, audit log of every dispatch decision

## Phase 5 — Market participation (the revenue engine) ⬜

- ⬜ **Commercial data + market access**: Nord Pool license (or vendor); BRP
  (Balance Responsible Party) partnership or membership
- ⬜ Day-ahead trading → **intraday continuous** trading
- ⬜ **Ancillary / balancing services** (FCR, aFRR, mFRR) — typically *higher
  value* than pure arbitrage; the real money for BESS
- ⬜ **Revenue stacking**: co-optimize arbitrage + ancillary across one battery
  (capacity allocation between markets) — value is migrating from saturated
  FCR-D toward aFRR/mFRR/intraday, so multi-market is the point

### Go-to-market wedge (decided — see research below)

**Lead with software, not assets.** Asset-light optimization/trading SaaS beats
building an aggregator VPP for a capital-light entrant, because:
- Aggregation faces a regulatory wall: a Swedish BSP must *also* hold a BRP
  agreement at every delivery point; full **independent-aggregator status only
  arrives ~2028**. Selling software to existing BSP/BRP holders sidesteps it.
- FCR-D is **saturated and price-collapsed** (~€5/MW by Jan 2025; BESS ~73% of
  FCR-D-up vs sub-550 MW demand). A single-market aggregation bet rides a
  collapsing market; multi-market software rides the growing one.
- Incumbents exist in both, but barriers are lower in software.

**Minimum first product:** the Tibber + ENTSO-E pipeline + Python optimizer,
packaged as a **revenue-stacking dispatch/arbitrage API** for SE3/SE4 battery
owners who lack in-house quant teams. Hybrid path: graduate *into* aggregation
as independent-aggregator rules mature toward 2028.

**Market-access build-vs-partner ladder** (climb it; don't start at the top):
1. Pure software ("the brain") — sell optimization to a BSP/BRP/owner
2. Sub-aggregator / route-to-market partner — deliver into an existing portfolio
3. BSP-only, partner a BRP
4. Full stack — become BRP + BSP (collateral, eSett, Ediel, 24/7 ops)

**Known competitors:** Capalo AI (Zeus VPP — direct SaaS competitor: 300+ MW,
50+ assets, 4 countries, €11M Series A Feb 2026); CheckWatt (aggregation: ~15k
customers, ~100 MW, ~1/5 of Swedish FCR-D 2024). Differentiate on forecasting/
optimization quality, segment focus (C&I / self-operating owners), or SE3/SE4 depth.

**Verified unit economics (2024–25, SE3/SE4):**
- Battery revenue **~€90–140k/MW/yr** (~€100k central, 1 MW/2 MWh) — *and falling*
- Aggregator (CheckWatt) take: **20% performance fee** (10/10 split) + €5/system/mo
  + Swedish BRP 5–10% admin → owner keeps **~70–80%**
- Asset-light optimizer take: **~10% of trading profit** on 1–3 yr deals
  → only **~€3–6k/MW/yr** per MW. **Thin → this is a scale game** (need hundreds
  of MW under management). Tolling (~€110–150k/MW/yr fixed) is capital-heavy — *not us*.
- Category is VC-fundable: Entrix €43M + GridBeyond €12M (2026), Capalo €11M.

**Sharpest opening — aFRR early-mover:** SE has **0 MW prequalified battery aFRR**
(Jan 2025; vs DK 5.3, FI 40). The aFRR energy-activation market opens only when
SvK joins **PICASSO (~2026, date unconfirmed)**. FCR-D is saturated/collapsed;
aFRR is the empty next product. **Build the optimizer to be first/best at SE aFRR
bidding before it saturates like FCR-D did.**

**Open questions before committing capital:** cost/team/time-to-revenue to build
this (no data — partly internal); exact grid-scale SaaS pricing; confirmed PICASSO
date; full competitive map beyond Capalo/Entrix/GridBeyond/CheckWatt (Flower,
Sympower, Sourceful Energy, gridX, Enspired unverified).

## Phase 6 — Product, scale & aggregation ⬜

- ⬜ Multi-asset / multi-tenant platform
- ⬜ **VPP aggregation**: pool many small batteries, bid as one virtual asset
- ⬜ Customer-facing product (Next.js over the Python API): accounts, dashboards,
  reporting, billing — enforcing the data-licensing boundary
- ⬜ Compliance, SLAs, observability at fleet scale

---

## SoC estimation — critical workstream (Phase 2 → 4)

**Why it gates everything:** dispatch and market commitments assume you know the
available energy. A 5% SoC error means mis-sized bids → **under-delivery penalties**
(severe in FCR/aFRR where delivery is mandatory) and bad arbitrage timing. SoC
accuracy is a hard requirement for revenue, not a nicety.

**Why LFP makes it *harder*:** LFP's **flat OCV–SoC curve** (~20–90% SoC) means
voltage barely moves across most of the range, so voltage-based SoC is unreliable
in the middle band; pure coulomb counting **drifts** over time; and LFP has notable
**voltage hysteresis** (charge vs discharge OCV differ). This is the well-known LFP
SoC problem.

**Methods (state of the art):**
- Coulomb counting (integrate current) — accurate short-term, drifts long-term
- OCV correction at the curve's steep ends (near full/empty) to re-anchor
- **Model-based filters: Extended/Unscented Kalman Filter (EKF/UKF)** over an
  equivalent-circuit model — the industry standard; fuses current + voltage + model
- Data-driven / ML estimators once fleet data exists
- Periodic **full-cycle recalibration** to reset drift

**Build vs consume — key decision (flag, don't block):**
- **Software-only player** (most BESS optimizers, incl. Flower/Capture-style):
  *consume* SoC from the pack's BMS/EMS over Modbus/REST, and build a **validation
  + uncertainty + fusion layer** on top (sanity-check BMS SoC against your own
  coulomb-count + energy throughput; carry an uncertainty band into the optimizer).
- **Vertically integrated (own hardware/BMS):** SoC estimation becomes core IP
  (EKF/UKF on raw cell telemetry). Much larger scope.
- **Default recommendation:** consume BMS SoC + own a validation/uncertainty layer,
  and make the optimizer **SoC-uncertainty-aware** (Phase 3 robust optimization).
  Revisit owning estimation only if you go vertical.

---

## Cross-cutting

- **Data licensing** — enforce internal-vs-external boundary (`data-licensing.md`)
- **Infra/DevOps** — storage, scheduling, secrets, deployment, observability
- **Regulatory** — market membership, BRP, metering/settlement compliance
- **Validation discipline** — backtest with realistic costs before any live action

---

*Representative battery figures here are illustrative and must be verified against
the actual CATL pack datasheet (cycle-life-vs-DoD, calendar curve, RTE, C-rate,
warranty terms) before they enter the economic model.*
