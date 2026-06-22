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
- ⬜ **Price forecasting** — day-ahead first, then intraday:
  - Baselines (naive, seasonal) → statistical (SARIMAX, gradient boosting) → ML
  - Output **probabilistic** forecasts (quantiles), not point estimates — the
    optimizer needs uncertainty
- ⬜ Forecast backtesting + accuracy tracking (pinball loss, MAE by horizon)

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
     (`optimizer/degradation.py`); rainflow/semi-empirical are stubs
  2. Rainflow cycle counting + DoD curve (value irregular cycling)
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
- ⬜ **Rolling-horizon / MPC** dispatch driven by Phase-1 forecasts
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
