# Trading Agent

A research agent for one question about ETH options: **when the market expects
volatility to expand, how cheap is the premium to sell — and does the trend say the
direction?**

It takes market data as input and produces an *analysis* (regime, expected price, credit-spread plan), never an order. No broker integration; no trading happens here.

## What it does

1. **Gradients** — how fast parts of the term structure are moving, and whether that
   speed is rising or falling (velocity, acceleration).
2. **Trend** — which direction the structure points and how much to trust it (3/3
   timeframe agreement, ADX).
3. **Expected price** — where price is most expected to be at a named horizon, with a
   band and the condition that would invalidate it.

Only then does it plan a **credit spread** — bullish regime → bull put, bearish → bear
call. No debit spreads, no naked options, no single legs.

## Current status

**The signal has not yet earned the right to be traded.** Over one year of 15m bars
(39,999 bars, 2025-09 → 2026-10), 272 non-overlapping 4h windows where the gate fired:
the trend read called the direction correctly **44.9%** of the time, lift **0.91x**
(95% interval 0.72–1.14). No edge demonstrated.

The pricing shadow (`src/pricing.py`) measures what a seller is actually paid —
`edge = credit − E[payoff | trailing realised vol]` — beside the trend read, and
deliberately **never overrides** it. The latest live reading disagreed with the trend
(bullish read, bear-call edge negative → NO TRADE), and that disagreement is the
measurement.

The real remaining gap is **sample size**, not plumbing: the scorer (`src/score.py`)
needs hundreds of scored runs before any claim of skill is reportable.

## Commands

    ./venv/bin/python -m pytest tests -q                    # offline, no network
    ./venv/bin/python src/main.py --equity 100000           # decision + plan
    ./venv/bin/python src/main.py --equity 100000 --journal out/projections.jsonl
    ./venv/bin/python src/main.py --chain synthetic         # price off the stub, deliberately
    ./venv/bin/python src/main.py --no-bars                 # skip the OHLCV fetch
    ./venv/bin/python src/events.py                         # base rate + lift, from bars
    ./venv/bin/python src/events.py --backfill 40 --json out/base_rate.json
    ./venv/bin/python src/score.py out/projections.jsonl    # error vs baseline, per symbol
    ./venv/bin/python src/dashboard.py                      # http://127.0.0.1:7860

## Layout

| file | role |
|---|---|
| `src/indicators.py` | fetch + validate snapshots, TTL cache, CSV loader |
| `src/analysis.py`   | pure maths: term structure, drift, regime, projection, planner |
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

## Invariants worth knowing

- **Slopes are % of spot per hour** across timeframes; the lookback axis is not the
  clock axis. The sign is part of the unit — never read a slope as a raw number.
- **No projection without validation.** Every estimate ships with a horizon, a band and
  the condition that invalidates it; nothing is quoted without the baseline beside it.
- **Freshness cannot be verified** from this feed — the scraper stamps its own clock.
- **Measured and assumed never look alike.** An unmeasured range prints
  `UNMEASURED`, not a zero; a synthetic credit prints `SYNTHETIC`, not a credit; the bar
  store is append-only and deduped on `open_time`.

Full detail lives in [`src/AGENTS.md`](src/AGENTS.md); the phased plan and its
acceptance tests live in [`src/plan.md`](src/plan.md).
