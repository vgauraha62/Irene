# Trading agent

This agent takes in every data source this folder has to serve. Its job is not to
describe the market, but to answer three questions, in this order:

1. **Gradients** — how fast is each part of the term structure moving, and is that
   speed rising or falling (velocity, acceleration)?
2. **Trend** — which direction is the structure actually pointing, and how much should
   that conclusion be trusted?
3. **Expected price** — given 1 and 2, where is price most expected to be at a named
   horizon, and what range contains it?

Only then does it guide trades, and only credit spreads:
- bullish regime -> bullish put spread (bull put)
- bearish regime -> bearish call spread (bear call)

No debit spreads, no naked options, no iron condors, no single-leg trades.

**The signal has not yet earned the right to be traded.** Measured over one year
of bars (39,999 × 15m, 2025-09 → 2026-10, 272 non-overlapping 4h windows where the
gate fired), the trend read called the direction correctly 44.9% of the time
(95% interval 39.0–50.8) against a same-direction base rate, for a **lift of
0.91x**. That is not an edge. `src/events.py` is the ruler, `out/base_rate.json`
is where it is written down, and the dashboard shows it above the scorecard. Do
not add a predictor, a threshold or a filter until that number moves and its
interval clears 1.00x.

The spread is priced off the live Deribit ETH chain (`src/iv.py`). When that chain is
unreachable the run continues on a synthetic Black-Scholes stub and says so in
`payload["chain"]["source"]` — on the console, in the JSON, and on the dashboard header,
because a synthetic credit and a market credit must never look alike.

## What is measured

| quantity | definition | units |
|---|---|---|
| ladder slope | `d(% of spot) / d(hours of EMA lookback)`, per rung and summarised | %/hour |
| forward drift | that slope negated, i.e. on the clock axis (`dP/dt`) | %/hour |
| velocity | first time-derivative of one price series, per bar | %/hour |
| acceleration | second time-derivative of the same series | %/hour^2 |
| agreement | how many of 15m / 30m / 4h point the same way | 0..3 |
| trend strength | ADX per timeframe, with the chop floor at 20 | index |
| skill | `1 - MAE(projection) / MAE(predict no change)` | ratio |
| base rate | `P(\|move\| >= k trailing sigma over H)`, unconditionally | % |
| lift | a bucket's hit rate over the base rate for the **same direction** | ratio |
| ATR | Wilder true range over 14 bars, as % of spot | % |
| compression | today's Keltner width, ranked against the last 200 windows | percentile |
| expected move | realised 1-sigma over a horizon, from log returns | % |
| edge | `credit - E[payoff | trailing realised vol]`, per spread | USD, % of width |
| breakeven rate | `1 - credit / width`, the full-win rate a spread needs | % |
| keep / partial / max loss | `N(-d2)`-derived outcome split across the two strikes | % |

The ladder slope and the forward drift are **different axes**. The slope differentiates
with respect to lookback; the drift differentiates with respect to time. Never substitute
one for the other — that substitution was the original bug.

## How "most expected price" is defined

Not a point prediction. A conditional expectation with a band, at a named horizon:

    E[log P(t+H)] ~= log spot + drift_H * H + 0.5 * accel_H * H^2

- `drift_H` comes from the ladder whose timeframe suits the horizon (15m up to 1h, 30m
  up to 3h, 4h beyond); `accel_H` corrects for curvature.
- The band widens with cross-timeframe drift dispersion and in chop, and never goes below
  a fixed floor.
- Every projection ships with its **horizon**, the **band**, and the **condition that
  would invalidate it**. A bare number is not an output.

## Invariants — do not regress these

1. **Slopes are % of spot per hour.** A 4h bar is 16x a 15m bar; comparing per-bar
   gradients across timeframes made the first version report a permanently bullish shift.
2. **Lookback axis != clock axis.** See above. Velocity and acceleration come from a
   time series of one series — `velocity_and_acceleration`.
3. **No network at import.** Fetch with an explicit timeout, retries and backoff.
4. **A trade needs all three:** 3/3 agreement, ADX >= 20, credit >= 20% of width.
   Otherwise the answer is NEUTRAL / NO TRADE, naming the failing condition.
   The credit in that test is the **executable** credit, not the exchange mark
   (invariant 13). The pricing gate in `pricing.py` runs beside this one and is
   **shadow only** (invariant 12); it never satisfies or bypasses this rule.
5. **Output is a candidate, never an order.** No broker integration. The option chain
   is Deribit's live mark surface; the synthetic Black-Scholes stub is a *declared*
   fallback, recorded in `payload["chain"]["source"]`, never a silent default.
6. **Freshness cannot be verified** from this feed — the scraper stamps its own clock.
   Report `UNVERIFIED`; never invent an exchange timestamp. (Bars, when they arrive,
   will carry their own.)
7. **No projection without validation.** A price estimate is only reportable once it has
   been compared against realised outcomes out of sample. `score.py` is that comparison;
   until `n` is large enough to mean anything, every projection stays labelled a
   diagnostic. Never quote an error without the baseline beside it — a 0.4% error is worse
   than useless if predicting "no change" scores 0.1%.
8. **The sign is part of the unit.** Before the sign fix the agent read a calm +0.30 %/h
   rise as "BEARISH REGIME (GRAVITY)", because it used the ladder slope as a direction.
   `test_rising_market_is_bullish` and `test_drift_agrees_with_the_spectral_spread` are
   the guard; do not delete them.
9. **A projection is scored only against its own symbol.** Pairing is by time, so a single
   time array shared by an ETHUSD forecast and a BTCUSD spot "scores" the wrong asset and
   reports the result as skill. `score_records` filters by symbol and raises on a mixed
   journal with no filter; `test_an_eth_projection_is_never_scored_against_a_btc_price`
   is the guard.
10. **Measured and assumed never look alike, and raw bars are never rewritten.** An
    unmeasured range prints `UNMEASURED`, not a zero; a synthetic credit prints
    `SYNTHETIC`, not a credit. The bar store is append-only and deduped on `open_time`,
    because the only thing that makes 1,000 bars worth more than 1 is that the history
    was not silently edited. `bars.append_bars` and the `range` block's `status` are the
    guards.
11. **A backtest asks the question production asks, and never sees the future.** The
    state is rebuilt with `analysis.regime` from bars that had closed by that moment;
    the "real move" threshold scales with *trailing* sigma; consecutive samples do not
    overlap; train and test are separated by a purged gap. Every rate carries a Wilson
    interval and no rate is quoted without its base rate beside it.
    `test_state_at_cannot_see_the_future` and `test_a_split_that_leaks_fails_loudly`
    are the guards — the first grafts a spike onto the future and asserts the past does
    not move.
12. **The pricing gate is measured, never obeyed.** `pricing.py` asks a different
    question from the trend read — "is the credit larger than the payoff that trailing
    realised vol expects" — and answers it with `edge = credit - E[payoff | sigma_rv]`.
    It is recorded in the payload, the journal and the dashboard, and it emits nothing:
    with lift 0.91x on the direction signal there is no basis for letting a second,
    equally unproven gate override it or trade on its own. Promotion requires the
    calibration in `score.py` to exclude a lift of 1.00x.
    `test_the_shadow_does_not_change_the_decision_it_is_recorded_beside` and
    `test_the_block_calls_itself_a_shadow` are the guards.
13. **A credit is the price you can transact, not the price the venue marks.** The
    planner sells at `bid(short)` and buys at `ask(long)` and reports `credit_mark`
    beside `credit_source` (`book` / `mark` / `model`). Observed on the live chain: the
    mark overstates the executable credit by 1.0-2.7 USD per 50-wide spread, which is
    larger than the modelled edge and enough to move a 20%-of-width gate from pass to
    fail. `test_the_credit_is_not_the_mark` is the guard.
14. **A probability is N(-d2), not 1 - delta.** Delta is N(-d1); for an out-of-the-money
    put it understates `P(S_T < K)` by ~5 points at 0.30 delta and 30 days, which is the
    same order as the credit ratio the gate compares it with — so the approximation can
    flip a decision. The spread has **three** outcomes (keep all / partial / max loss)
    and all three are published. `test_the_probability_is_n_d2_and_not_one_minus_delta`
    is the guard.
15. **Strikes are instruments, and a contract is 1 ETH.** A 2211 put cannot be sold, so
    the planner snaps to the listed ladder, prices in the **chain's** spot rather than
    the scanner's, and reports `expiry` and `dte_actual` beside the requested DTE —
    `chain.iv("put", 7, ...)` silently returns the 9-day expiry, and "7d" that is really
    9d is a lie about the instrument. Sizing uses `contract_size = 1` on Deribit,
    not the 100-share equity convention that made every position 100x too small.
16. **A virtual observation is not a fill.** `journal.py` records analysis
    observations; `trade_journal.py` records user-started and user-closed binary
    outcomes in a separate append-only file. The Trades tab never sends orders
    and asks for no prices or trade details. Multiple observations per side can
    be open; each start is separate and Exit closes the oldest one. It picks the
    closest listed ATM option and the cheapest two-sided OTM hedge in the nearest
    30-day expiry. The virtual outcome is directional: bull put succeeds above
    entry spot; bear call succeeds below entry spot. Entry spot is captured
    automatically as a hidden comparison reference, and exit spot is not persisted.
17. **EMA weights are a shadow experiment.** Keep all six rung gradients, the
    versioned settings and the frozen listed ATM target with each eligible run.
    Score only non-overlapping four-hour outcomes against later bars. Weight
    changes apply to future records; they never rewrite history or alter the
    production regime, decision or trade buttons. Keep scheduled predictions
    separate from user-selected virtual entries because the latter are selected
    observations and can be biased.

## Current gap

- **The measurement exists and the signal is on the wrong side of it.** Lift
  0.91x [0.72, 1.14] at 4h; 0.97x at 1h; 0.93x at 12h. The interval contains
  1.00x, so the honest claim is "no edge demonstrated", not "proven harmful".
  But it does **not** follow that the spread loses money: a bull put pays if price
  stays above the short strike, not if price rises. Directional accuracy is the
  wrong scoreboard for a credit structure, and the right label — `survived(...)`,
  price never closing beyond the short strike before expiry — is Phase 4's first
  job. The blocker is sample size: a 30-day horizon over one year is ~12 windows.
- ~~Nothing measured whether the signal was better than chance.~~ Fixed:
  `events.py` replays the live `analysis.regime` from stored bars, labels the
  forward move against **trailing** sigma, samples non-overlapping windows and
  reports Wilson intervals, lift over the same-direction base rate, and a purged
  chronological split. `test_state_at_cannot_see_the_future` is the guard: graft a
  spike onto the future, the past must not move.
- **The direction question may simply be the wrong one for a credit spread.** A bull
  put is paid if price stays above the short strike, not if price rises. So
  `pricing.py` now measures the quantity a seller is actually paid on —
  `edge = credit - E[payoff | trailing realised vol]` — non-directionally, on both
  sides, at three deltas, and journals it beside the trend read. First live reading
  (2026-10-28, spot 2358, 30d, 21d realised 45.0%): bull put 2200/2150 credit 12.97
  (26% of width), edge **-1.84**; bear call 2550/2600 credit 9.43 (19% of width,
  fails the credit floor), edge **-1.57**. Verdict: NO TRADE on pricing alone, while
  a bullish trend read would still have emitted the put. That disagreement is the
  measurement, and it is now recorded on every run.
  The edge is deliberately computed at a **flat** sigma from bars, not at the
  market's own surface — at the surface it is zero by construction, which would
  answer nothing. It is also optimistic: the bar store has 21 fifteen-minute moves
  over 3% in a year where a lognormal at 45% predicts essentially none, and fat
  tails hurt a short-premium position.

- ~~The payload collapses the per-rung slopes to a summary.~~ Fixed: every rung is in
  `market.timeframes.<tf>.ladder_slope_pct_per_hour`, with `drift_pct_per_hour` beside it.
- ~~There is no projection function.~~ Fixed: `analysis.project_price` /
  `project_all`, published at 30m and 4h in `payload["projection"]`.
- ~~Nothing scores a projection.~~ Fixed: `journal.py` records every run, `score.py`
  reports error **and** the random-walk baseline, with `skill = 1 - MAE/baseline`.
- **The real gap now is sample size, not plumbing.** Skill needs hundreds of runs. One
  writer, one line per run; schedule it, do not press Refresh:
  `*/15 * * * * cd /path && ./venv/bin/python src/main.py --no-chart --journal out/projections.jsonl`
- ~~Nothing showed the score.~~ Fixed: the dashboard has Live / Score / About tabs. Score
  reads the journal and nothing else, so a rate-limited feed cannot blank it. Page load
  runs no scrape at all — market data is behind an explicit Refresh, because opening the
  dashboard used to cost three TradingView requests per view.
- Auto-journaling is deliberately **off** by default: manual runs are selection-biased
  (you look when the market moves), and a biased sample is worse than a small one.
- Consecutive projections overlap, so the effective sample is smaller than `n`. Purged,
  embargoed splits are required before any claim of predictive skill.
- Acceleration cannot be computed from one snapshot — `velocity_and_acceleration` needs a
  time series, so `project_price` is first-order unless a caller passes acceleration in.
- ~~The spread was priced off an invented vol.~~ Fixed: `iv.py` reads Deribit's live
  chain (one unauthenticated request, 736 instruments, ~11 expiries) and
  `DeribitChain` implements `OptionsChain`. Observed on the first live run: the stub
  quoted 61% where the market marked 50%, so every credit, strike, breakeven and
  contract count was wrong by 10-15%. The chain source is now in the payload, the
  journal and the dashboard header.
- Two unit facts worth keeping: Deribit publishes `mark_iv` in **percent** (61.6), so
  `_quote_iv` converts to a decimal at the parse boundary; and its put and call
  `mark_iv` at the same strike are **identical** (one surface per expiry), so the skew
  lives across strikes, not across sides.
- ~~The feed returns 91 scalars: no OHLC, no ATR.~~ Fixed for *measurement*, not yet
  for the band. `bars.py` fetches 1,000 completed klines from Binance (one request,
  no key), stores them append-only under `out/bars/`, resamples 15m -> 30m/4h locally
  and derives ATR, Bollinger and Keltner width, a compression percentile and the
  realised 1-sigma move at each horizon. The cross-check against the scanner agrees
  to **0.17 % max**, with ours systematically *higher* because the scanner's EMA
  includes the part-formed candle. **The band itself is still the old one, and it is
  now known to be 1.6-3.2x too narrow**: the 4h band is +-0.29 % against a realised
  1-sigma of 0.92 %. The measured sigma is printed beside the band and recorded in
  every journal line; swapping the band is gated on `score.py`'s `in-band %`.
- Bars are `ETHUSDT` on Binance; the trade and the chain are `ETH`. That basis is not
  modelled and the payload says so (`range.basis_note`). The bar store is the only
  input with genuine exchange timestamps, so invariant 6 is satisfiable for the bar
  side only.
- ~~The bar store path depended on the caller's working directory.~~ Fixed: `main.BARS_ROOT`
  is anchored to the module, not the CWD. It was relative, so launching the dashboard
  from `src/` silently built `src/out/bars/` with ten days of bars while the backfilled
  year sat in `out/bars/` untouched — a second store nobody was reading.
  `test_the_bar_store_never_resolves_against_the_working_directory` checks the
  signature default, because the path is captured at import and a `chdir` test would
  pass regardless.

## Layout

| file | role |
|---|---|
| `src/indicators.py` | fetch + validate snapshots, TTL cache, CSV loader |
| `src/analysis.py`   | pure maths: term structure, drift, velocity, regime, projection, planner |
| `src/main.py`       | orchestrator; `analyse(**kwargs)` is the API, `run(args)` the CLI |
| `src/journal.py`    | append-only JSONL record of every run (`out/projections.jsonl`) |
| `src/iv.py`         | live Deribit option chain -> `analysis.OptionsChain`; no key, no deps |
| `src/pricing.py`    | the shadow gate: `edge = credit - E[payoff | realised vol]`, never obeyed |
| `src/bars.py`       | Binance klines -> append-only CSV store, resample, ATR / ADX / width / sigma |
| `src/events.py`     | labels, base rates, lift, purged split; no predictor on purpose |
| `src/score.py`      | scores past projections against realised prices, with a baseline |
| `src/weight_model.py` | versioned grouped EMA-gradient weights; shadow direction only |
| `src/weight_study.py` | non-overlapping ATM outcome scoring, grouped by weight version |
| `src/plot.py`       | matplotlib chart, headless (Agg); the third panel is the compression series |
| `src/trade_journal.py` | append-only virtual observation events; binary outcomes and success rates |
| `src/dashboard.py`  | local Gradio view, 127.0.0.1:7860; Live / Score / Trades / Weights / About tabs |

## Commands

    ./venv/bin/python -m pytest tests -q                    # offline, no network
    ./venv/bin/python src/main.py --equity 100000           # decision + plan
    ./venv/bin/python src/main.py --equity 100000 --journal out/projections.jsonl
    ./venv/bin/python src/main.py --chain synthetic         # price off the stub, deliberately
    ./venv/bin/python src/main.py --no-bars                 # skip the OHLCV fetch
    ./venv/bin/python src/events.py                         # base rate + lift, from bars
    ./venv/bin/python src/events.py --backfill 40 --json out/base_rate.json
    ./venv/bin/python src/events.py --horizon 1             # sweep the label horizon
    ./venv/bin/python src/score.py out/projections.jsonl    # error vs baseline, per symbol
    ./venv/bin/python src/score.py out/projections.jsonl --symbol ETHUSD --json
    ./venv/bin/python src/dashboard.py                      # http://127.0.0.1:7860
    ./venv/bin/python src/dashboard.py --journal other.jsonl --tab Score
