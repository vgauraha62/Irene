# Plan — from direction to a volatility edge

Decisions taken: **implied volatility first, bars second, Fibonacci deferred.**
This is the order of work, not a wish list. Each phase has an acceptance test that
can fail.

## Why this order

The agent currently forecasts *direction* and sells *volatility*. Direction is the
third most important input for a premium seller; the price of volatility relative
to what it delivers is the first. Today the trade arithmetic is computed from a
`synthetic` Black-Scholes stub (`analysis.SyntheticChain`), so the
`credit >= 20% of width` gate in `spread_plan` is a gate on an invented number.

Phase 1 makes that number real. Phase 2 makes range, structure and freshness real.
Phase 3 supplies the base rate without which no predictor can be judged. Phase 4
spends that measurement budget on structure signals.

## Non-goals

- No iron condors, no debit structures, no naked legs. Unchanged (invariant 5).
- No order routing. Output stays a candidate.
- No machine learning before Phase 3 produces a base rate. There is no sample to
  fit, and a model with parameters spends sample that does not exist yet.
- No Fibonacci. See *Deferred* at the bottom for the reasoning, so it is not
  re-litigated every month.

---

## Phase 0 — start the clock

**Status: code done.** `journal.SCHEMA_VERSION = 2`; every line now carries
`drift_split` (front/median/back per timeframe) and `chain` (what priced it).
Scheduling the writer is still yours to install — the line is in `AGENTS.md`.

**Goal:** nothing below can be measured without history, and one writer is already
built.

**Do**
1. Schedule the collector (`*/15 * * * *`, documented in `AGENTS.md`).
2. Extend `journal.record_from_payload` to keep, per timeframe, the ladder split:
   `drift_front` (`-ladder_slope["EMA10"]`), `drift_median`, `drift_back`
   (`-ladder_slope["EMA200"]`). These are already computed in
   `analysis.rung_slopes` and currently discarded.
3. Bump `journal.SCHEMA_VERSION` to 2; `journal.load` already skips wrong-schema
   lines, so old records age out instead of being misread. **Note:** this discards
   the handful of existing records. Acceptable at `n=2`, never acceptable again —
   after this, extend the record additively and keep one schema version.

**Acceptance:** a run appends a line containing `drift_front`; `score.py` still
reports the old records as skipped, not as errors.

**Why now:** Phase 4's H1 (front-rung lead/lag) needs nothing but this history. It
starts accumulating today and costs one small change.

---

## Phase 1 — real implied volatility

**Status: done.** `src/iv.py` — `fetch_chain`, `build_snapshot`, `DeribitChain`;
`--chain deribit|synthetic`; `payload["chain"]`, the journal and the dashboard header
all report the source and ATM IV. 24 offline tests against
`tests/fixtures/deribit_eth_chain.json`.

What the first live run showed: the stub quoted 61% where the market marked 50%, so
the 0.30-delta strike sat 20 points away from where it actually sits and the credit
was overstated (32.4% of width synthetic vs 28.6% live). The gate passed either way;
the arithmetic did not.

Two facts found while implementing, both now asserted in tests:

- Deribit publishes `mark_iv` in **percent**. 61.6 is 61.6%. Converted to a decimal
  once, at the parse boundary, because 61.6 as a sigma is not a price.
- Put and call `mark_iv` at the same strike are **identical** — one surface per
  expiry. The plan expected a side-to-side skew; there is none on the mark. The skew
  is across strikes. The `kind` parameter still earns its place (instrument
  selection, and traded bid/ask IVs do differ), but the earlier rationale was wrong.

**Goal:** replace the synthetic chain with live ETH option IV, so credit,
breakeven, POP and the credit gate are computed from market prices.

**Data source:** Deribit public API — no key, no per-call quota of the kind the
TradingView scraper has.

- `GET /api/v2/public/get_instruments?currency=ETH&kind=option` — strikes, expiries, types.
- `GET /api/v2/public/get_book_summary_by_currency?currency=ETH&kind=option`
  — `mark_iv`, `bid_iv`, `ask_iv`, `underlying_price`, `open_interest`,
  `volume`, `mid_price` per instrument. One request returns the whole chain.
- `GET /api/v2/public/get_volatility_index_data?currency=ETH&...` — DVOL history
  for a vol-percentile gate later.

**Deliverables**
1. `iv.py` (new): `DeribitChain` implementing `analysis.OptionsChain`.
   - pick the expiry nearest the requested DTE;
   - interpolate in log-moneyness across the live strike grid;
   - fall back to `SyntheticChain` when the chain is empty or stale, and **say so**
     in the payload;
   - refuse chains older than a configured age rather than silently pricing off
     yesterday.
2. **Interface change:** `OptionsChain.iv(dte_days, strike, spot)` cannot say
   which side to read, and a chain quotes puts and calls as separate instruments
   (separate bid/ask, and either side can be the thin one). Change it to
   `iv(kind, dte_days, strike, spot)`; update `SyntheticChain`, `spread_plan` and
   the tests. This is a real defect found by reading the seam, not a nicety.
3. Payload: `payload["chain"] = {"source": "deribit"|"synthetic", "as_of": ...,
   "iv_atm": ..., "skew_25d": ...}`.
4. `plot.py` / dashboard: one line showing chain source and ATM IV, so a synthetic
   credit is never mistaken for a market one.

**Acceptance**
- A live run reports `source: deribit`; the credit differs from the synthetic run
  and the difference is explained by the IV difference.
- Tests stay offline: capture one response to `tests/fixtures/deribit_chain.json`
  and test the parser, the log-moneyness interpolation, the stale rejection and
  the fallback.
- The 20% credit gate is now computed from market IV. Record the gate's decision
  in the journal so its effect can be measured later.

**Does not prove:** that the spread is profitable. It proves the arithmetic is
about the real market instead of a constant `base_iv = 0.60`.

---

## Phase 2 — bars, and with them range and freshness

**Status: code done, with one deliberate deferral.** `src/bars.py` fetches, stores
append-only, resamples and derives; `main.bars_block` wires it in fail-soft;
ATR reaches the payload, the table, the header, the console and the journal.

**Measured on the first live run** (2026-10-28, spot 2,351.56):

- Cross-check against the scanner, 13 EMAs across 3 timeframes: **max relative
  error 0.17 %**, and the sign is systematic — ours is *higher* every time,
  because the scanner's EMA includes the part-formed candle and ours ends at the
  last closed bar. That is a lag, not a different instrument.
- ATR(14): 15m 0.24 %, 30m 0.38 %, 4h 1.31 %.
- Compression: **4th percentile** — the tightest 4.5 % of the last 200 windows.
- **The published band is 1.6–3.2x too narrow.** 30m band ±0.17 % against a
  realised 1-sigma of 0.28 %; 4h band ±0.29 % against 0.92 %. A band narrower
  than the random walk claims knowledge it does not have.

**Deliverable 5 is deferred on purpose.** Replacing `BAND_FLOOR_PCT` with an
ATR-scaled form is now an arithmetic question with a known answer (multiply by
~3), but *which* band is better is a coverage question, and coverage is measured
by `score.py`'s `in-band %`, which needs scored runs that do not exist yet. So the
band stays as specified in `AGENTS.md` and the measured 1-sigma is shown beside it
on the console, the header and in every journal line. Switching is one line in
`analysis.project_price` once ~30 scored projections say the wider band covers.

**Follow-up, after the first user run: the measures were technically present and
practically invisible.** Three fixes, all small:

- `main.BARS_ROOT` was a *relative* path, so a dashboard started from `src/`
  created `src/out/bars/` with ten days of bars while the backfilled year sat in
  `out/bars/` unread. Anchored to the module now, with a test on the signature
  default (a `chdir` test would pass even with the bug, because the path is
  captured at import).
- The range line became a **panel**: per-timeframe ATR / Keltner / 1σ, the
  compression percentile with a rank bar, and `band ±x% vs measured 1σ y% (ratio)`
  per horizon. The band-is-too-narrow finding was a grey sentence; it is now the
  first thing under the header.
- `plot.py` gained a **third panel**: Keltner width over the last 120 bars against
  its own 10th–90th percentile band, with the current point marked. It appears
  only when a width series was measured — an unmeasured range draws two panels,
  never a flat line at zero.

The lesson worth keeping: a measurement that exists in the payload and in a test
has not been delivered. It has been delivered when it is on the screen, in the
place the eye already goes, next to the thing it qualifies.

**Goal:** get real OHLCV with exchange timestamps.

**Data source:** one public klines endpoint, e.g.
`GET api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=15m&limit=1000` —
1,000 completed bars, exchange timestamps, no key, weight 1. `requests` is already
a dependency; `ccxt` is not needed.

**Deliverables**
1. `bars.py` (new): fetch, append-only store under `out/bars/<symbol>_<tf>.csv`,
   dedupe on `open_time`, no whole-file rewrite.
2. Resample 15m → 30m / 4h locally so all timeframes come from one series.
3. **Cross-check:** recompute `EMA10..EMA200` from bars and diff against the 91
   scalars from the TradingView feed. Report max relative error. This is a free
   validation of the current input that has never been possible.
4. Derived, computed on read (never stored as the source of truth): ATR(14),
   Bollinger width, Keltner width, width percentile over N bars, volume z-score,
   realized volatility.
5. Volatility-adaptive band: replace the flat `BAND_FLOOR_PCT = 0.15` with a form
   driven by ATR and horizon, keeping the existing floor as a lower bound.
6. Freshness: anything derived from bars carries the exchange timestamp, so
   `as_of_source` becomes `"exchange"` for that side. Invariant 6 stops being
   unsatisfiable for bar-derived numbers.

**Acceptance**
- 1,000 bars stored, zero duplicate `open_time`, store survives an interrupted write.
- The EMA cross-check is printed on a run; a discrepancy above a set threshold is
  an error, not a warning.
- ATR and Keltner width match an independent hand calculation in a test.
- ~~The new band is wider than the flat one in compression and narrower in quiet
  expansion, demonstrated in a test.~~ Changed: the measured 1-sigma is published
  beside the band and recorded in the journal, and the band is switched only when
  `in-band %` justifies it. Replacing it blind would have been the same mistake as
  the invented volatility in `iv.py`, in the opposite direction.

**Where each item landed**

| deliverable | where | state |
|---|---|---|
| fetch + append-only store | `bars.fetch_klines`, `bars.append_bars` | done |
| dedupe on `open_time` | `bars.append_bars` (filter, never rewrite) | done |
| resample 15m → 30m / 4h | `bars.resample`, clock-aligned, incomplete tails dropped | done |
| EMA cross-check | `bars.cross_check`, 5 % tolerance, surfaced on screen | done |
| ATR / Bollinger / Keltner / percentile / realised vol | `bars.py`, derived on read | done |
| volume z-score | not done | deferred: nothing reads it yet |
| volatility-adaptive band | published beside the band, journaled | deliberately deferred (above) |
| exchange-timestamped freshness | `range.last_closed_bar`, `fetched_at` | done for the bar side |

**Store raw, derive on read.** Storing derived quantities freezes today's formulas
and makes every future fix throw away the history.

---

## Phase 3 — labels and the base rate

**Status: done, with one deliberate change of target.** The base rate is measured
and it is not flattering.

**The question was reframed, and this is the important part.** The plan proposed
labelling "a trend starts" — a hand-drawn threshold on a drift that would have to
be replayed from the feed, which cannot be done. What the agent actually does is
claim a *direction* and sell a spread against it, so the honest label is: **at
time t, did price move the way the claim said, by at least k trailing sigmas, over
the next horizon?** That label can be computed from bars, it survives a change of
threshold definition, and it is the thing the trade depends on.

**What was built** (`src/events.py`):

- `build_ladder` resamples once and precomputes every derived series, so a state
  can be read at any index without recomputing history.
- `state_at(i)` rebuilds the *live* `analysis.regime` from the bars that had
  closed by `i`. Same ladder, same flat band, same ADX floor. A backtest that
  asks a different question than production is a different backtest.
- `sample()` walks the history at a stride that defaults to the horizon, so
  consecutive labels are **non-overlapping by construction**. 96 samples a day
  are not 96 trials.
- `sigma_pct` is **ex-ante**: the threshold scales with trailing realised vol, so
  an event never gets easier because the future turned out calm.
- `wilson_interval` on every rate. 8.3% of 12 is reported as an interval, not a
  number.
- `purged_split` + `assert_disjoint`: chronological split with the seam removed.
- CLI: `python src/events.py --backfill 40 --json out/base_rate.json`.

**Measured** (Binance ETHUSDT 15m, 39,999 bars, 2025-09-06 → 2026-10-28, horizon
4h, event = |move| ≥ 0.5σ, 1,499 non-overlapping windows):

| | rate | 95% interval |
|---|---|---|
| up ≥ 0.5σ | 24.6% | 22.4 – 26.8 |
| down ≥ 0.5σ | 23.0% | 21.0 – 25.2 |
| flat | 52.4% | 49.9 – 55.0 |
| **signal fired** (3/3, out of chop), direction correct | **44.9%** | 39.0 – 50.8 |
| signal, claimed direction by ≥ 0.5σ | 21.7% | 17.2 – 27.0 |
| **lift vs same-direction base rate** | **0.91x** | 0.72 – 1.14 |

Sweeping the horizon does not rescue it: 1h 0.97x, 4h 0.91x, 12h 0.93x, 24h
1.19x (n=33), 48h 0.74x (n=15). The large-n horizons cluster below 1.00x.

**What this means, precisely.** The trend-state signal has *no measurable
directional edge* at a 4h horizon, and the point estimate is mildly anti-predictive
(mean reversion). The interval contains 1.00x, so the correct claim is "no edge
demonstrated", not "proven harmful". And it does **not** mean the spread loses
money: a bull put profits if price stays above the short strike, not if price
rises. Directional accuracy is the wrong scoreboard for a credit structure, and
replacing it with the right one is Phase 4's first job, not a detail.

**Two deliverables moved and one is not done:**

- *Precision/recall in `score.py`*: deferred, deliberately. `score.py` scores the
  journal, and the journal has no labelled runs yet. The base rate from bars is
  the substitute the plan actually needed, and it does not depend on the
  collector running. `in-band %` remains the calibration metric on the Score tab.
- *The base rate is now visible*, which the plan did not ask for and should have:
  `events.py --json out/base_rate.json` puts it on the Score tab, with its
  interval, above the scorecard.
- *Not done*: the spread-payoff label (P(close beyond the short strike | price
  and delta)). It needs either a 30-day horizon over more than one year of bars
  or a history of option prices. Both are absent right now, and inventing a
  pseudo-label would repeat the mistake `iv.py` fixed.

**Still true:** this phase adds no predictor. The numbers are the ruler.

---

## Phase 4 — structure signals, one at a time

**First item is not in the table and outranks everything in it:** the label Phase 3
could not build. A credit spread is a bet that price *stays on one side of a
strike*, not that it goes up, so directional accuracy is the wrong scoreboard.
Before any of H1–H4 is worth testing, `events.py` needs a second label:

    survived(bars, i, horizon_bars, short_strike, kind)
        = price never closed beyond `short_strike` before expiry

with `short_strike` placed the way the planner places it (~0.30 delta ≈ 1σ of the
30-day implied move). The blocker is honest: a 30-day horizon over one year of bars
gives ~12 non-overlapping windows, and the alternative — a history of option
prices — does not exist here yet. Options: backfill 120 pages (~3.3 years, ~40
windows), or accept overlapping windows with a stated effective n. Decide before
building it; do not quietly pick the flattering one.

Each hypothesis is pre-registered here, tested alone, and reported with its base
rate. Adding twenty of them and keeping the winner is how a backtest lies.

| id | hypothesis | metric | needs |
|---|---|---|---|
| H1 | front-rung flip precedes the median flip | lead time in bars; precision vs base rate | Phase 0 only |
| H2 | compression (KC width < 20th pct) precedes expansion | timing only; direction expected ≈ 50% | Phase 2, 3 |
| H3 | compression **and** drift sign call the expansion side | precision, lift | Phase 2, 3 |
| H4 | entry in a 0.5–0.618 retracement beats a random entry | lift, one parameter | Phase 2, 3 |

**Compression is a risk gate, not an alpha source.** We are short volatility: a
squeeze is the moment premium is cheapest and expansion is closest. It belongs in
the planner (demand more credit) and in the band (widen before the resolution),
not in the direction logic.

**Acceptance:** for each hypothesis, a report with `n`, base rate, lift, and a
confidence interval, or an explicit "no lift detected". Negative results get
written down, not deleted.

---

## Deferred — Fibonacci

Not now, and probably not ever, for three reasons:

1. **Two free parameters per timeframe** (swing anchor, level choice) against a
   sample that is still zero. Any level that "held" is one of twenty-four tried.
2. **No mechanism.** Nothing enforces a 0.618 retracement. At best it is a
   coordinate system where other participants place orders.
3. **A cheaper equivalent exists:** pullback depth normalized by ATR has one
   parameter, the same intent, and is scale-free across ETH at 2,300 and 4,000.

Revisit only after Phase 3 exists, and only as H4: one anchor definition, one
zone, tested against the base rate.

---

## Measurement discipline

- **Wrong metric for the question.** MAE scores a price estimate. A yes/no claim
  scores precision, recall and lift. Do not judge a regime signal by MAE.
- **The base rate is the baseline.** Every hit rate is quoted beside it.
- **Purged, embargoed splits.** Overlap makes effective `n` smaller than `n`.
- **Few hypotheses.** Each extra signal is a multiple-testing liability; at 5%
  significance, twenty signals produce one winner by chance.
- **Negative results are recorded.** A hypothesis that failed is knowledge.
- **No claim from `n` below the threshold** — currently 20, banner-enforced on the
  dashboard.

## Invariants that still apply

| invariant | how this plan respects it |
|---|---|
| 1. slopes are %/h | any new series normalizes before comparison; ATR and widths are stored as % of spot |
| 2. lookback ≠ clock | H1 is measured on the clock axis; rung slopes keep their own sign |
| 3. no network at import | Deribit and klines calls go through explicit-timeout, retrying fetchers |
| 4. trade needs 3/3, ADX ≥ 20, credit ≥ 20% | unchanged, but the credit term becomes market-derived |
| 5. candidate, never an order | unchanged |
| 6. freshness unverified | bars carry exchange timestamps; the feed side stays `UNVERIFIED` |
| 7. no projection without validation | Phase 3 is the gate; nothing in Phase 4 is reportable before it |
| 8. the sign is part of the unit | `drift_front` is defined as the negation of the ladder slope, explicitly |
| 9. score only within a symbol | `bars.py` and `journal.py` both key on symbol |

## What would make this wrong

- Deribit's `mark_iv` can be thin on illiquid strikes; interpolating across a
  sparse grid invents a surface. Check open interest before trusting a node.
- Bars from one exchange are not "the price". ETHUSD on Coinbase and ETHUSDT on
  Binance diverge in stressed moments, and the option chain is Deribit's.
- Compression predicts expansion *timing* with a wide and variable delay. A
  squeeze can last weeks.
- Everything here still assumes the past distribution resembles the next one; a
  regime break (a listing, a hack, a policy shock) invalidates the calibration,
  and no indicator in this plan detects that.
