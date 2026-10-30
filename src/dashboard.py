"""Local Gradio dashboard for the trading agent.

    python dashboard.py            # then open http://127.0.0.1:7860

This is a *view* over ``main.analyse``. It holds no analysis logic, so the CLI
and the dashboard can never disagree about what the market is doing.

The Trades tab is a virtual observation tool. One click starts a bull-put or
bear-call observation; another click closes it and records a binary outcome.
It does not place orders and never edits the signal journal.

Security posture (deliberate):

* binds ``127.0.0.1`` only, with no authentication, because that is the only
  configuration where none is needed;
* ``share=True`` is never used and not exposed as a flag -- it would publish an
  unauthenticated tunnel that lets strangers use this machine as a free
  TradingView scraper;
* symbol and exchange are regex-validated before they reach the feed, so the
  form cannot be used as an open query proxy;
* every interpolation into HTML is escaped, so a crafted ticker cannot inject
  markup into the page.
"""

from __future__ import annotations

import os

# Must be set before gradio is imported: no telemetry from a box holding
# trading state.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import html
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image

import analysis
import bars
import indicators
import journal
import main as pipeline
import score
import trade_journal
import weight_model
import weight_study

log = logging.getLogger("dashboard")

HERE = Path(__file__).resolve().parent
CHART_PATH = HERE.parent / "out" / "dashboard" / "latest.png"

#: Where the collector appends runs. The dashboard only ever reads this file.
JOURNAL_PATH = HERE.parent / "out" / "projections.jsonl"

#: Output of `events.py --json`. Written by hand, read on page load: the base
#: rate needs 40,000 bars and ~2 s of CPU, which is not a page-load cost.
BASE_RATE_PATH = HERE.parent / "out" / "base_rate.json"
TRADE_JOURNAL_PATH = HERE.parent / "out" / "virtual_trades.jsonl"

#: Below this many scored projections the numbers are machinery, not evidence.
MIN_SCORED = 20

#: At a 15-minute cadence, three missed slots means the writer is probably dead.
STALE_AFTER_MINUTES = 45

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

TICKER_RE = re.compile(r"^[A-Z0-9]{2,15}$")
EXCHANGES = ["COINBASE", "BINANCE", "KRAKEN", "BITSTAMP", "BYBIT"]
SYMBOLS = ["ETHUSD", "BTCUSD", "SOLUSD"]

DECISION_STYLE = {
    "CANDIDATE": ("#0f5132", "#d1e7dd"),
    "NO TRADE": ("#842029", "#f8d7da"),
}
ERROR_STYLE = ("#842029", "#f8d7da")
NEUTRAL_STYLE = ("#41464b", "#e2e3e5")

DISCLAIMER = (
    "Indicative analysis only. Not investment advice, not an order. "
    "Option pricing uses the live Deribit chain when it is reachable; the header "
    "says SYNTHETIC when it is not, in which case the credit is arithmetic rather "
    "than a market. The upstream price feed cannot confirm candlestick freshness."
)

OUTPUTS = 11  # 8 live (incl. the shadow gate) + score verdict, scorecard, journal health

ABOUT = """
## What this page is

A local view over one analysis pass: three timeframes of TradingView indicators
(15m / 30m / 4h), read as an EMA term structure, turned into either a credit-spread
candidate or an explicit refusal. It never places an order. Option prices come from
the live Deribit chain when it answers; the header says **SYNTHETIC** when it does
not, and then the credit is arithmetic rather than a market. The upstream price feed
cannot confirm candle freshness, so range is measured from a separate bar feed and
labelled `UNMEASURED` when that feed is down.

## The three questions, in order

1. **Gradients** — how fast is each part of the term structure moving?
2. **Trend** — which way is it pointing, and how much should that be trusted?
3. **Expected price** — where is price most expected to be at a named horizon?

Trade guidance is the *last* step, never the first.

The **Trades** tab records virtual observations. Its strike rule is fixed: nearest
30-day expiry, sell the closest listed ATM option, and hedge with the cheapest
two-sided OTM option on that wing. Directional success means a bull put closes
above entry spot or a bear call below it. The entry reference is captured
automatically; premiums, fees, size, exit price and P&amp;L are not recorded.

The **Weights** tab is a separate shadow experiment. It stores versioned weights,
all EMA rung gradients and measured range context with each collected run, then
grades the direction against the frozen Deribit ATM strike four hours later.
Changing weights never rewrites past records or changes the production decision.

## Reading the term-structure chart (left panel)

This is the panel that misleads people, so read it twice.

- The x-axis is **hours of EMA lookback**, not time. `EMA10` is the most recent
  average and sits on the left; `EMA200` is the oldest and sits on the right.
- The y-axis is price.
- Therefore: **a curve that climbs left to right means price has FALLEN.** The older
  averages are higher than the newer ones. A curve that falls left to right means
  price has RISEN.
- The chart draws the past, not the future. The right-hand end of a curve is not a
  forecast.
- The dashed red line is spot — where price is right now.
- The vertical gap between the 15m curve and the 4h curve is the "wave inside
  gravity": short-term flow versus long-term inertia. When they point opposite ways,
  the agent stands down.

## Reading the drift chart (right panel)

One bar per EMA rung, per timeframe, in **% of spot per hour**.

- **Positive bar = price rising. Positive is up.** This panel is already flipped
  relative to the ladder slope, on purpose.
- Bars of equal height across rungs mean a smooth, uniform trend. Bars that shrink
  or change sign along the rungs mean the trend lives in one part of the ladder only.
- Compare the 15m and 4h groups: the difference is the micro-versus-macro divergence.

## Reading the tables

| column | meaning |
|---|---|
| `drift %/h` | trend rate on the clock. **Positive = rising.** This is what the regime reads. |
| `ladder slope %/h` | the same measurement against lookback hours, so it has the **opposite sign** by construction. Negative here means rising. Never treat it as a direction. |
| `term spread %` | `EMA10` minus `EMA200`, as % of spot. Negative means the long-term average sits above the current price, i.e. a long slide. |
| `ADX` | trend strength. Above ~25 is a trend; below 20 is chop, and the chop filter refuses to trade. |
| `ATR %` | average true range over the last 14 bars, as % of spot. This is **measured from real OHLCV**, not assumed. A blank means the bar feed was down, so the range is unknown — not zero. |
| `expected / low / high` | the projection and its band. |
| `band %` | half-width of that band, in % of spot. |
| `read` | the projection's direction. `neutral` means the expected move is *inside* the band — not a failed forecast, a declined one. |
| `from` | which timeframe's ladder produced the projection. |

## Reading the range line under the header

One public klines request gives real candles, so range is now measured rather than
assumed. The header carries the result:

- `ATR x.xx%` — average true range over 14 bars, as % of price, per timeframe.
  Bigger ATR means the same drift matters less: a 0.3 %/h rise means one thing when
  the 15m bar is 0.2 % tall and another when it is 1.0 % tall.
- `compression Nth pctile` — today's Keltner width ranked against the last 200 of its
  own past. 10th percentile is a coiled market; 90th is an expanded one. A percentile
  needs no arbitrary cut-off to be read, which is why it is reported instead of a
  green/red label.
- `band 30m ±0.31% vs 1-sigma 0.29%` — the honest test of the band. The right-hand
  number is the one-sigma move implied by realised volatility over that horizon. A
  band **narrower** than one sigma is claiming to know more than the random walk does.
  That is a diagnostic, not a promise, and it is why the band is not yet used to gate
  trades: it has to beat this number out of sample first.
- `range UNMEASURED` — the bar feed was unreachable. Every ATR and percentile is
  absent and the band falls back to its documented floor.
- `local EMAs DISAGREE with the feed` — the candles we fetched and the scanner do not
  describe the same series. The run continues, but nothing derived from the bars
  should be trusted until that is reconciled.

## Reading the base rate (top of the Score tab)

This is the number every prediction has to beat, and it is measured, not assumed.
`events.py` walks ~40,000 stored bars, rebuilds the regime at each point **using
only bars that had already closed**, and asks what price did over the next horizon.

- `up / down / flat` — the unconditional rate of a move of at least **k sigma**
  over the horizon. This is chance, with the market's own volatility baked in.
- `signal fired N times` — how often the live gate produced a direction (3/3
  timeframes agreeing, ADX out of chop). The credit gate is not modelled: it needs
  a history of option prices.
- `direction %` — how often price moved the claimed way *at all*. 50% is a coin
  flip; the interval in brackets is the 95% Wilson interval, so a wide bracket
  means the sample cannot tell 45% from 55%.
- `lift` — the signal's hit rate divided by the base rate for the **same
  direction**. `1.00x` means the signal is worth exactly what happens anyway.
  Below 1.00x means it is worse than doing nothing, which is possible and is worth
  knowing. Lift is computed against the base rate, never against 50%: in a market
  that rises 60% of the time, 60% is not an edge.

Small `n` is not rounded away. Every rate carries an interval, and below ~100
non-overlapping windows the banner says the numbers are provisional.

Regenerate it after a backfill:

    ./venv/bin/python src/events.py --backfill 40 --json out/base_rate.json

## Reading the decision card

`NO TRADE` is a result, not an error. The bullets underneath name the condition that
failed: timeframes disagreeing, ADX below 20, a flat macro leg, or a credit too thin
for its width. `CANDIDATE` means the structure passed; it is still not an order.

## Reading the scorecard

This tab scores **past projections** against what price actually did. The rule is
simple and it is not negotiable:

> **Look at `skill`, not `MAE`.** `skill = 1 - MAE / baseline`, where the baseline is
> "assume price does not move". Positive skill means the drift beat doing nothing.
> Zero or negative means it did not, however small the error looks.

| column | meaning |
|---|---|
| `n` | projections actually scored, not the number of runs. |
| `MAE %` | mean absolute error of the projection. |
| `baseline %` | the same error for predicting no change. |
| `skill` | how much the projection beat that baseline. The only number that matters. |
| `in-band %` | how often the realised price landed inside the published band. This tests whether the band is honest about its own uncertainty. |
| `dir hit %` | among projections that called a direction (neutral excluded). |
| `max gap h` | the latest any realising observation arrived after its horizon. Large gaps mean irregular sampling. |

Below ~20 scored projections the numbers are plumbing, not evidence: the tab says so
in a banner, and it is not being modest.

Consecutive projections overlap — a 4h projection made every 15 minutes shares 15/16 of
its window with the next one — so the effective sample is smaller than `n`. Any claim of
predictive skill needs purged, embargoed splits.

## What would make this wrong

- The 4h candle is mid-formation most of the time, so its EMAs move under you.
- The scanner supplies 91 scalars and no candle history. Bars now come from a
  separate Binance request, which means the ATR is measured on `ETHUSDT` while the
  trade and the options are `ETH`. That basis is not modelled.
- The projection band is still a convention (`0.15%` floor plus drift and dispersion).
  The measured one-sigma move is now shown beside it, and the two disagree often. The
  band is not replaced until the scored comparison says which one covers reality.
- The live chain gives one IV per strike per expiry. It does not tell you the fill you
  would get: the credit is a mark, not a bid. A spread sold at the mark is a spread
  sold to nobody.
- Deribit's `mark_iv` is a surface mark, and it is identical for a put and a call at
  the same strike. The skew that matters is across strikes, which the planner does
  interpolate, and across expiries, which it picks the nearest of rather than
  interpolating.
- If the chain is unreachable the run continues on the stub and the header says
  SYNTHETIC. Read that before reading the credit.
- Freshness is `UNVERIFIED`: the scraper stamps its own clock, so candle age cannot be
  confirmed.
"""


# --------------------------------------------------------------------------- #
# Rendering helpers (pure: they only format a payload)
# --------------------------------------------------------------------------- #


def _card(title: str, body: str, *, palette: tuple[str, str]) -> str:
    foreground, background = palette
    return (
        f"<div style='background:{background};color:{foreground};border-radius:8px;"
        f"padding:14px 18px;font-family:system-ui,sans-serif;'>"
        f"<div style='font-size:13px;letter-spacing:.08em;text-transform:uppercase;"
        f"opacity:.75'>{html.escape(title)}</div>"
        f"<div style='font-size:24px;font-weight:700;margin-top:2px'>{body}</div>"
        f"</div>"
    )


def _header(payload: dict[str, Any]) -> str:
    market = payload["market"]
    fresh = payload["freshness"]
    as_of = pipeline._pretty_ts(payload["analysis"]["as_of"])

    badge = (
        "<span style='color:#0f5132'>freshness verified</span>"
        if fresh["status"] == "verified"
        else "<span style='color:#b35c00;font-weight:600'>freshness UNVERIFIED</span>"
    )
    note = (
        f"<div style='font-size:12px;color:#6b7280;margin-top:4px'>{html.escape(fresh['note'])}</div>"
        if fresh.get("note")
        else ""
    )
    priced = payload.get("chain") or {}
    chain = ""
    if priced:
        source = str(priced.get("source", "unknown")).upper()
        colour = "#0f5132" if source == "DERIBIT" else "#b35c00"
        bits = [f"<span style='color:{colour};font-weight:600'>{html.escape(source)} chain</span>"]
        if isinstance(priced.get("iv_atm"), (int, float)):
            bits.append(f"ATM IV {priced['iv_atm']:.1%}")
        if priced.get("note"):
            bits.append(html.escape(str(priced["note"])))
        chain = "<div style='font-size:12px;margin-top:4px'>" + " · ".join(bits) + "</div>"
    return (
        "<div style='font-size:18px;font-weight:600;color:#111827'>"
        f"{html.escape(payload['symbol'])} @ {html.escape(payload['exchange'])}"
        "<span style='font-weight:400;color:#6b7280;font-size:14px'>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;spot <b>{market['spot']:,.2f}</b>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;as of {as_of}</span></div>"
        f"<div style='margin-top:6px'>{badge}</div>{note}{chain}{_range_line(payload)}"
    )


def _range_line(payload: dict[str, Any]) -> str:
    """Measured range, or a loud admission that it is not measured.

    A wide band and an unknown band must never look the same: one is confidence,
    the other is an absence of information. This sits directly under the header
    because the band-versus-sigma comparison is the honest answer to "how much do
    you trust that range?" -- and it used to be a single grey sentence.
    """
    ranges = payload.get("range") or {}
    status = ranges.get("status")
    if status not in {"ok", "mismatch"}:
        reason = ranges.get("reason", "not requested")
        return (
            "<div style='font-size:12px;margin-top:6px;color:#b35c00'>"
            f"range UNMEASURED · {html.escape(str(reason))}</div>"
        )

    measured = ranges.get("timeframes") or {}
    compression = ranges.get("compression") or {}
    rank = compression.get("percentile")
    mismatch = status == "mismatch"

    def pct(value: Any) -> str:
        return "—" if value is None else f"{value:.2f}%"

    rows = "".join(
        "<tr>"
        f"<td style='padding:1px 14px 1px 0;color:#374151'>{html.escape(timeframe)}</td>"
        f"<td style='padding:1px 14px 1px 0;text-align:right'>{pct(row.get('atr_pct'))}</td>"
        f"<td style='padding:1px 14px 1px 0;text-align:right'>{pct(row.get('keltner_width_pct'))}</td>"
        f"<td style='padding:1px 14px 1px 0;text-align:right'>"
        f"{pct((row.get('expected_move_pct') or {}).get('30m'))}</td>"
        f"<td style='padding:1px 0;text-align:right'>"
        f"{pct((row.get('expected_move_pct') or {}).get('4h'))}</td>"
        "</tr>"
        for timeframe, row in measured.items()
    )

    headline = "range &amp; volatility · measured from bars, not assumed"
    colour = "#b35c00" if mismatch else "#6b7280"
    warning = (
        "<div style='font-size:12px;color:#b35c00;margin-top:4px'>"
        "local EMAs DISAGREE with the feed — these numbers are reported but must not "
        "be trusted until the series is reconciled</div>"
        if mismatch
        else ""
    )

    bar = ""
    if rank is not None:
        bar = (
            "<div style='margin-top:5px;font-size:12px;color:#374151'>"
            f"compression <b>{rank:.0f}th</b> percentile "
            f"({html.escape(str(compression.get('reads', 'unranked')))})"
            "<span style='display:inline-block;width:120px;height:6px;background:#e5e7eb;"
            "border-radius:3px;vertical-align:middle;margin-left:8px;overflow:hidden'>"
            f"<span style='display:block;width:{max(1.0, min(100.0, float(rank))):.0f}%;height:6px;"
            "background:#26a69a'></span></span></div>"
        )

    # The band against a same-horizon random walk. This is the number that says
    # whether a published range means anything at all.
    bands = []
    for projection in payload.get("projection") or []:
        sigma = (
            (measured.get(projection.get("basis_timeframe")) or {})
            .get("expected_move_pct", {})
            .get(projection.get("horizon"))
        )
        if sigma in (None, 0):
            continue
        ratio = projection["band_pct"] / sigma
        verdict = "narrower than a random walk" if ratio < 1.0 else "at or wider than a random walk"
        tone = "#b35c00" if ratio < 1.0 else "#047857"
        bands.append(
            f"<div style='font-size:12px;color:#374151;margin-top:2px'>"
            f"band {html.escape(str(projection.get('horizon')))} "
            f"±{projection['band_pct']:.2f}% vs measured 1σ {sigma:.2f}% "
            f"<span style='color:{tone}'>({ratio:.2f}x — {verdict})</span></div>"
        )

    return (
        "<div style='margin-top:8px;padding:8px 10px;border:1px solid #e5e7eb;"
        "border-radius:8px;background:#fafafa'>"
        f"<div style='font-size:10px;letter-spacing:.08em;text-transform:uppercase;"
        f"color:{colour};font-weight:600'>{headline}</div>"
        "<table style='font-size:12px;border-collapse:collapse;margin-top:4px;width:auto'>"
        "<tr style='color:#9ca3af;font-size:10px'>"
        "<th style='text-align:left;font-weight:500;padding-right:14px'>tf</th>"
        "<th style='text-align:right;font-weight:500;padding-right:14px'>ATR</th>"
        "<th style='text-align:right;font-weight:500;padding-right:14px'>Keltner</th>"
        "<th style='text-align:right;font-weight:500;padding-right:14px'>1σ 30m</th>"
        "<th style='text-align:right;font-weight:500'>1σ 4h</th></tr>"
        f"{rows}</table>{bar}{''.join(bands)}{warning}</div>"
    )


def _decision(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    palette = DECISION_STYLE.get(decision, ERROR_STYLE)
    body = html.escape(decision)
    reason = html.escape(payload["reason"])

    # The reasoning trail behind the verdict, rendered from the same payload the
    # decision came from, so the screen never shows a conclusion without the read
    # that produced it.
    regime = payload.get("analysis") or {}
    regime_bits = [
        f"regime {html.escape(str(regime.get('macro_dir') or ''))}",
        f"direction {html.escape(str(regime.get('direction') or ''))}",
    ]
    agreement = regime.get("agreement")
    if isinstance(agreement, int):
        regime_bits.append(f"{agreement}/3 timeframes agree")
    adx = regime.get("adx")
    if adx is not None:
        regime_bits.append(f"ADX {float(adx):.1f}")
    confidence = regime.get("confidence")
    if confidence is not None:
        regime_bits.append(f"confidence {float(confidence):.2f}")
    regime_line = " · ".join(regime_bits)

    reasons = payload.get("analysis", {}).get("reasons") or []
    bullets = "".join(f"<li>{html.escape(str(item))}</li>" for item in reasons)
    detail = (
        f"<ul style='margin:6px 0 0 18px;padding:0;font-size:13px;color:#374151'>{bullets}</ul>"
        if bullets
        else ""
    )

    return (
        _card("decision", body, palette=palette)
        + f"<div style='font-size:13px;color:#374151;margin-top:6px'>"
        f"reason: {reason}</div>{detail}"
        + f"<div style='font-size:12px;color:#6b7280;margin-top:6px'>{regime_line}</div>"
    )


def _timeframe_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "timeframe": timeframe,
            "close": row["close"],
            # Clock-side trend rate: positive means price is rising.
            "drift %/h": row["drift_pct_per_hour"],
            # Lookback-side derivative, averaged over the rungs. Opposite sign.
            "ladder slope %/h": _median_ladder_slope(row),
            "term spread %": row["spectral_spread_pct"],
            "ADX": row["adx"],
            # Measured from real bars; None when the bar feed is down. A blank
            # here means "not measured", never "zero volatility".
            "ATR %": row.get("atr_pct"),
        }
        for timeframe, row in payload["market"]["timeframes"].items()
    ]
    return pd.DataFrame(rows)


def _median_ladder_slope(row: dict[str, Any]) -> float | None:
    rungs = [v for v in (row.get("ladder_slope_pct_per_hour") or {}).values() if v is not None]
    if not rungs:
        return None
    return round(sorted(rungs)[len(rungs) // 2], 5)


def _projection_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "horizon": projection["horizon"],
            "expected": projection["expected_price"],
            "low": projection["low"],
            "high": projection["high"],
            "band %": projection["band_pct"],
            "read": projection["direction"],
            "from": projection["basis_timeframe"],
        }
        for projection in payload.get("projection") or []
    ]
    return pd.DataFrame(rows)


def _ladder_table(payload: dict[str, Any]) -> pd.DataFrame:
    timeframes = list(payload["market"]["timeframes"])
    data: dict[str, Any] = {"EMA": list(analysis.MA_KEYS)}
    for timeframe in timeframes:
        emas = payload["market"]["timeframes"][timeframe].get("emas", {})
        data[timeframe] = [emas.get(key) for key in analysis.MA_KEYS]
    return pd.DataFrame(data)


SCORE_COLUMNS = [
    "horizon",
    "n",
    "MAE %",
    "baseline %",
    "skill",
    "in-band %",
    "dir hit %",
    "max gap h",
]
HEALTH_COLUMNS = ["metric", "value"]
WARN_STYLE = ("#664d03", "#fff3cd")
OK_STYLE = ("#0f5132", "#d1e7dd")


def _score_table(card: score.ScoreCard) -> pd.DataFrame:
    rows = [
        {
            "horizon": f"{item.horizon_hours:g}h",
            "n": item.n,
            "MAE %": round(item.mae_pct, 4),
            "baseline %": round(item.baseline_mae_pct, 4),
            "skill": round(item.skill, 3),
            "in-band %": round(item.band_coverage * 100.0, 1),
            "dir hit %": (
                None
                if item.directional_hit_rate is None
                else round(item.directional_hit_rate * 100.0, 1)
            ),
            "max gap h": round(item.max_gap_hours, 2),
        }
        for item in card.horizons
    ]
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def _journal_health(records: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    """What the journal looks like, so a dead collector cannot hide behind an empty table."""
    rows: list[tuple[str, Any]] = [
        ("path", str(path)),
        ("records", len(records)),
        ("symbols", ", ".join(score.symbols_of(records)) or "-"),
    ]
    if records:
        moments = sorted(score.observed_at(record) for record in records)
        first, last = moments[0], moments[-1]
        age_minutes = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
        gaps = [
            (later - earlier).total_seconds() / 3600.0
            for earlier, later in zip(moments, moments[1:])
        ]
        rows += [
            ("first observation", pipeline._pretty_ts(first.isoformat())),
            ("last observation", pipeline._pretty_ts(last.isoformat())),
            ("age of last", f"{age_minutes:.0f} min"),
            ("largest gap", f"{max(gaps):.2f} h" if gaps else "-"),
            (
                "writer",
                f"STALE: nothing written for {age_minutes:.0f} min "
                f"(expected every 15 min)"
                if age_minutes > STALE_AFTER_MINUTES
                else "running",
            ),
        ]
    return pd.DataFrame(rows, columns=HEALTH_COLUMNS)


def _score_panel(path: str | Path | None = None, symbol: str = "") -> tuple:
    """The scorecard, with the base rate pinned above it when one exists."""
    card, table, health = _score_panel_core(path, symbol)
    return (card + _base_rate_block(), table, health)


def _base_rate_block(path: str | Path | None = None) -> str:
    """Render ``out/base_rate.json`` if it is there.

    The base rate is the number any predictor has to beat, so it belongs next to
    the scorecard rather than in a README. It is *not* recomputed here: one pass
    over 40,000 bars is ~2 s, and a page load must not pay that.
    """
    source = Path(path or BASE_RATE_PATH)
    if not source.exists():
        return ""
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        base = report["base"]
        signal = (report.get("buckets") or {}).get("signal") or {}
        product = report["product_bars"]
        spec = report["spec"]
    except Exception as exc:  # a hand-edited file must not take the tab down
        log.warning("base rate file unreadable: %s", exc)
        return _card("base rate unreadable", html.escape(str(exc)), palette=WARN_STYLE)

    lift = signal.get("lift")
    span = signal.get("lift_ci") or [None, None]
    low, high = span[0], span[1]
    if lift is None:
        verdict = "the gate never fired in this history"
        colour = WARN_STYLE
    elif high is not None and high < 1.0:
        verdict = f"lift {lift:.2f}x: significantly worse than chance, even at the edge of the interval"
        colour = WARN_STYLE
    elif low is not None and low > 1.0:
        verdict = f"lift {lift:.2f}x: beats chance, and the interval clears 1.00x"
        colour = OK_STYLE
    elif lift >= 1.05:
        verdict = f"lift {lift:.2f}x: promising, but the interval still includes 1.00x (chance)"
        colour = WARN_STYLE
    else:
        verdict = f"lift {lift:.2f}x: indistinguishable from chance on this history"
        colour = WARN_STYLE

    direction = signal.get("signed") or {}
    hit = signal.get("hit") or {}
    interval = f" [{low:.2f}, {high:.2f}]" if low is not None else ""
    detail = (
        f"<div style='font-size:12px;color:#374151;margin-top:6px'>"
        f"{product['count']:,} bars · {html.escape(product['from'][:16])} → "
        f"{html.escape(product['to'][:16])} · horizon {spec['horizon_hours']:g}h · "
        f"event = |move| ≥ {spec['size_sigma']:g}σ<br>"
        f"base rate over {base['n']:,} windows: up {base['up']['rate']:.1f}% · "
        f"down {base['down']['rate']:.1f}% · flat {base['flat']['rate']:.1f}%<br>"
        f"signal fired {signal.get('n', 0):,} times · direction {direction.get('rate', 0):.1f}% "
        f"[{direction.get('ci', [0, 0])[0]:.0f}, {direction.get('ci', [0, 0])[1]:.0f}] · "
        f"≥{spec['size_sigma']:g}σ move {hit.get('rate', 0):.1f}% · lift{interval}</div>"
    )
    return _card("base rate", html.escape(verdict), palette=colour) + detail


def _score_panel_core(path: str | Path | None = None, symbol: str = "") -> tuple:
    """Read the journal and render the scorecard. Pure I/O, never raises.

    Deliberately independent of ``analyse``: a TradingView rate limit must not
    blank the only panel that tells you whether the projections work.
    """
    book = Path(path or JOURNAL_PATH)
    symbol = (symbol or "").strip().upper()
    empty = pd.DataFrame(columns=SCORE_COLUMNS)
    no_health = pd.DataFrame(columns=HEALTH_COLUMNS)

    if not book.exists():
        return (
            _card(
                "no journal",
                f"No journal at {html.escape(str(book))}. Nothing has been recorded yet.<br>"
                "Schedule the collector, then come back:<br><code>"
                "*/15 * * * * cd /path &amp;&amp; ./venv/bin/python src/main.py "
                "--no-chart --journal out/projections.jsonl</code>",
                palette=ERROR_STYLE,
            ),
            empty,
            no_health,
        )

    try:
        records = journal.load(book)
    except Exception as exc:  # never let the score tab take the page down
        log.warning("journal could not be read: %s", exc)
        return (
            _card("journal unreadable", html.escape(str(exc)), palette=ERROR_STYLE),
            empty,
            no_health,
        )

    if not records:
        return (
            _card(
                "journal empty",
                f"{html.escape(str(book))} exists but holds no usable line.",
                palette=ERROR_STYLE,
            ),
            empty,
            no_health,
        )

    health = _journal_health(records, book)
    mine = [record for record in records if str(record.get("symbol")) == symbol]
    if not mine:
        return (
            _card(
                "no records for this symbol",
                f"Journal holds {len(records)} records, none for "
                f"{html.escape(symbol or '(no symbol selected)')}. "
                f"Symbols present: {html.escape(', '.join(score.symbols_of(records)))}.",
                palette=WARN_STYLE,
            ),
            empty,
            health,
        )

    card = score.score_records(mine, symbol=symbol)
    table = _score_table(card)
    scored = len(card.scored)

    if scored == 0:
        return (
            _card(
                "nothing scored yet",
                f"{card.records} records for {html.escape(symbol)}, 0 scored. Every "
                "projection is still waiting for a later observation. This is normal "
                "for the first few hours.",
                palette=WARN_STYLE,
            ),
            table,
            health,
        )

    if scored < MIN_SCORED:
        return (
            _card(
                "plumbing, not evidence",
                f"n={scored} - far too few to read. About {MIN_SCORED} scored "
                "projections are needed before these numbers mean anything. The "
                "numbers are shown anyway so the machinery can be checked.",
                palette=WARN_STYLE,
            ),
            table,
            health,
        )

    lead = max(card.horizons, key=lambda item: item.n)
    if lead.skill > 0:
        verdict, palette = (
            f"n={lead.n} at {lead.horizon_hours:g}h - skill {lead.skill:+.2f}: beating "
            f"'price does not move' by {lead.skill:.0%}. Overlap still unmeasured.",
            OK_STYLE,
        )
    else:
        verdict, palette = (
            f"n={lead.n} at {lead.horizon_hours:g}h - skill {lead.skill:+.2f}: the drift "
            "did not beat 'price does not move'. No edge here yet.",
            WARN_STYLE,
        )
    return (_card("projection score", html.escape(verdict), palette=palette), table, health)


def _plan(payload: dict[str, Any]) -> str:
    plan = payload.get("plan")
    if not plan:
        return "_No spread proposed for this read._"

    kind = "put" if "put" in plan["strategy"] else "call"
    notes = "\n".join(f"- {html.escape(str(note))}" for note in plan["notes"])
    source = plan.get("credit_source", "model")
    source_line = {
        "book": "bid(short) - ask(long) - the price a seller receives",
        "mark": "the exchange's mark; one leg has no two-sided market",
        "model": "Black-Scholes, not a market quote",
    }.get(source, source)
    return (
        f"#### {html.escape(plan['strategy'].replace('_', ' '))}\n\n"
        f"| field | value |\n|:--|:--|\n"
        f"| sell | **{plan['short_strike']:,.2f} {kind}** |\n"
        f"| buy | {plan['long_strike']:,.2f} {kind} |\n"
        f"| credit | {plan['credit']:.2f} ({plan['credit_ratio']:.0%} of width) |\n"
        f"| credit source | {html.escape(source_line)} |\n"
        f"| mark | {plan.get('credit_mark', plan['credit']):.2f} |\n"
        f"| width | {plan['width']:.2f} |\n"
        f"| max loss | {plan['max_loss_per_contract']:.2f} / contract |\n"
        f"| breakeven | {plan['breakeven']:,.2f} |\n"
        f"| POP (keeps all) | {plan['pop']:.0%} |\n"
        f"| partial / max loss | {plan.get('pop_partial', 0.0):.0%} / "
        f"{plan.get('pop_max_loss', 0.0):.0%} |\n"
        f"| expiry | {plan.get('expiry') or 'n/a'} "
        f"({plan.get('dte_actual') or plan['dte_days']}d) |\n"
        f"| contracts | {plan['contracts']} |\n\n"
        f"{notes}"
    )


def _pricing(payload: dict[str, Any]) -> str:
    """The shadow gate on screen. A measurement that is not visible does not exist."""
    block = payload.get("pricing") or {}
    if block.get("status") != "ok":
        reason = html.escape(str(block.get("reason") or block.get("status") or "unavailable"))
        return _card(
            "pricing (shadow)",
            f"<span style='font-size:14px;font-weight:500'>not measured - {reason}</span>",
            palette=NEUTRAL_STYLE,
        )

    verdict = block.get("verdict", "NO TRADE")
    palette = DECISION_STYLE.get(verdict, NEUTRAL_STYLE)
    sigma = block.get("sigma") or {}
    value = sigma.get("value")
    window = sigma.get("primary")
    pick = block.get("pick")
    headline = f"{verdict} - {pick} side" if pick else verdict

    rows = []
    for side in ("put", "call"):
        row = (block.get("sides") or {}).get(side) or {}
        if "short_strike" not in row:
            rows.append(
                f"<tr><td>{side}</td><td colspan='6' style='opacity:.6'>"
                f"{html.escape(str(row.get('reason') or row.get('status') or 'unavailable'))}"
                "</td></tr>"
            )
            continue
        edge = row["edge"]
        colour = "#0f5132" if edge > 0 else "#842029"
        rows.append(
            "<tr>"
            f"<td>{side}</td>"
            f"<td>{row['short_strike']:,.0f} / {row['long_strike']:,.0f}</td>"
            f"<td>{row['credit']:.2f}"
            f"<span style='opacity:.6'> (mark {row['credit_mark']:.2f})</span></td>"
            f"<td>{row['credit_ratio']:.0%}</td>"
            f"<td>{row['p_keep']:.0%} / {row['p_partial']:.0%} / {row['p_max_loss']:.0%}</td>"
            f"<td style='color:{colour};font-weight:600'>{edge:+.2f}</td>"
            f"<td>{html.escape(row.get('verdict', ''))}</td>"
            "</tr>"
        )

    table = (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;"
        "margin-top:10px'>"
        "<tr style='text-align:left;opacity:.7'>"
        "<th>side</th><th>strikes</th><th>credit</th><th>% width</th>"
        "<th>keep / part / lose</th><th>edge</th><th>shadow</th></tr>"
        + "".join(rows)
        + "</table>"
    )

    agreement = block.get("agrees_with_direction")
    if agreement is None:
        agree_line = "No direction candidate to compare against."
    elif agreement:
        agree_line = "Agrees with the direction gate's side."
    else:
        agree_line = (
            "DISAGREES with the direction gate "
            f"(it says {block.get('direction_side') or 'no trade'})."
        )

    detail = (
        f"<div style='font-size:12px;line-height:1.5;margin-top:8px'>"
        f"edge measured at <b>{value:.1%}</b> realised vol ({html.escape(str(window))}) - "
        "credit minus the payoff that vol expects. "
        f"{html.escape(agree_line)}. "
        "<b>Shadow only</b>: recorded and journaled, never emitted as a trade."
        f"<div style='opacity:.75;margin-top:4px'>{html.escape(str(block.get('reason', '')))}</div>"
        "</div>"
    )

    return (
        f"<div style='border:1px solid rgba(0,0,0,.12);border-radius:8px;padding:12px 14px;"
        f"font-family:system-ui,sans-serif;background:{palette[1]};color:{palette[0]}'>"
        f"<div style='font-size:13px;letter-spacing:.08em;text-transform:uppercase;opacity:.75'>"
        f"pricing (shadow)</div>"
        f"<div style='font-size:20px;font-weight:700'>{html.escape(headline)}</div>"
        f"{table}{detail}</div>"
    )


OPEN_OBSERVATION_COLUMNS = ["id", "side", "started", "expiry"]
CLOSED_OBSERVATION_COLUMNS = ["started", "closed", "side", "outcome"]


def _observation_tables() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    report = trade_journal.summary(trade_journal.load_events(TRADE_JOURNAL_PATH))
    by_side = report["by_side"]
    rate = report["success_rate_pct"]
    cards = [
        ("open", str(len(report["open"]))),
        ("closed", str(report["total_closed"])),
        ("directional success", str(report["successes"])),
        ("directional failure", str(report["failures"])),
        ("directional success rate", "—" if rate is None else f"{rate:.1f}%"),
        ("bull put", _side_rate(by_side["bull_put"])),
        ("bear call", _side_rate(by_side["bear_call"])),
    ]
    metrics = (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));"
        "gap:8px;margin:8px 0'>"
        + "".join(
            "<div style='border:1px solid #e5e7eb;border-radius:7px;padding:9px 11px'>"
            f"<div style='font-size:11px;color:#6b7280'>{html.escape(label)}</div>"
            f"<div style='font-size:18px;font-weight:650'>{html.escape(value)}</div></div>"
            for label, value in cards
        )
        + "</div><div style='font-size:12px;color:#6b7280'>"
        "Directional success means bull put closes above its entry spot or bear call closes below it. "
        "The entry spot is captured automatically as a comparison reference, not shown as a trade input. "
        "No premium, P&amp;L, size, or fee is part of this score."
        "</div>"
    )
    open_df = pd.DataFrame(
        [
            {
                "id": row["observation_id"][:8],
                "side": row["side"],
                "started": row["time"],
                "expiry": row["expiry"],
            }
            for row in reversed(report["open"])
        ],
        columns=OPEN_OBSERVATION_COLUMNS,
    )
    closed_df = pd.DataFrame(
        [
            {
                "started": row["time"],
                "closed": row["closed_at"],
                "side": row["side"],
                "outcome": row["outcome"],
            }
            for row in reversed(report["closed"][-100:])
        ],
        columns=CLOSED_OBSERVATION_COLUMNS,
    )
    return metrics, open_df, closed_df


def _observation_panel() -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    """Rehydrate both the persistent observations and their visible status."""
    events = trade_journal.load_events(TRADE_JOURNAL_PATH)
    report = trade_journal.summary(events)
    opened = report["open"]
    if opened:
        details = "; ".join(
            f"{row['side'].replace('_', ' ')} · started {row['time']} · id {row['observation_id'][:8]}"
            for row in opened
        )
        status = _card(
            f"{len(opened)} active observation(s) saved",
            html.escape(details),
            palette=OK_STYLE,
        )
    else:
        status = _card(
            "no open observations",
            "Start a bull put or bear call observation below.",
            palette=NEUTRAL_STYLE,
        )
    metrics, open_df, closed_df = _observation_tables()
    return status, metrics, open_df, closed_df


WEIGHT_STUDY_COLUMNS = [
    "weights version", "samples", "active predictions", "coverage %", "hit rate %",
    "95% low %", "95% high %", "majority base %", "lift", "4h vol MAE %", "settings",
]
MANUAL_WEIGHT_COLUMNS = [
    "weights version", "samples", "active predictions", "coverage %", "hit rate %",
    "95% low %", "95% high %", "majority base %", "lift", "settings",
]


def _weight_study_panel() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Read the prediction journal and local bars; never fetch market data here."""
    records = journal.load(JOURNAL_PATH)
    candles = bars.load_bars(bars.store_path(pipeline.BARS_ROOT))
    rows, accounting = weight_study.evaluate(records, candles)
    manual = weight_study.evaluate_closed_observations(
        trade_journal.load_events(TRADE_JOURNAL_PATH)
    )
    summary = (
        f"Scheduled: {accounting['eligible']} non-overlapping reads; "
        f"{accounting['pending']} awaiting a future bar; "
        f"{accounting['overlapping']} overlapping reads skipped; "
        f"{accounting['unusable']} without a live ATM reference. "
        "Rows are descriptive; use the confidence interval and sample size before comparing settings."
    )
    return (
        _card("EMA weight study", html.escape(summary), palette=NEUTRAL_STYLE),
        pd.DataFrame(rows, columns=WEIGHT_STUDY_COLUMNS),
        pd.DataFrame(manual, columns=MANUAL_WEIGHT_COLUMNS),
    )


def _weight_values(values: list[float]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    index = 0
    for timeframe in weight_model.TIMEFRAMES:
        result[timeframe] = {}
        for group in weight_model.GROUPS:
            result[timeframe][group] = float(values[index])
            index += 1
    return weight_model.clean_weights(result)


def save_weight_settings(*values: float) -> str:
    try:
        saved = weight_model.save_settings(_weight_values(list(values)))
        return _card(
            "weights saved for future reads",
            f"version {saved['version']}; existing records are unchanged",
            palette=OK_STYLE,
        )
    except Exception as exc:
        return _card("weights not saved", html.escape(str(exc)), palette=ERROR_STYLE)


def _side_rate(row: dict[str, Any]) -> str:
    if row["success_rate_pct"] is None:
        return "—"
    return f"{row['success_rate_pct']:.1f}% ({row['successes']}/{row['closed']})"


def _virtual_result(message: str) -> tuple:
    metrics, opened, closed = _observation_tables()
    return message, metrics, opened, closed


def _live_chain():
    chain, note = pipeline.default_chain(source="deribit")
    snapshot = getattr(chain, "snapshot", None)
    if note or snapshot is None:
        raise RuntimeError("live Deribit chain unavailable; observation was not changed")
    return chain


def start_virtual_observation(
    side: str, weights: dict[str, dict[str, float]] | None = None
) -> tuple:
    side = side.strip().lower()
    kind = "put" if side == "bull_put" else "call"
    try:
        snapshots = pipeline.collect()
        chain = _live_chain()
        book = chain.book(kind, dte_days=30)
        short_legs = [leg for leg in book.legs if leg.two_sided]
        if not short_legs:
            raise RuntimeError("no two-sided listed ATM option; observation was not started")
        short = min(short_legs, key=lambda leg: (abs(leg.strike - book.spot), leg.strike))
        wings = [
            leg for leg in book.legs
            if leg.two_sided
            and leg.ask > 0
            and (leg.strike < short.strike if side == "bull_put" else leg.strike > short.strike)
        ]
        if not wings:
            raise RuntimeError("no two-sided OTM hedge quote; observation was not started")
        # Cheapest positive OTM ask; tie-break toward the closest listed wing.
        hedge = min(wings, key=lambda leg: (leg.ask, abs(leg.strike - short.strike), leg.strike))
        state = analysis.regime(snapshots)
        drifts = analysis.drifts(snapshots)
        rung_slopes = {
            timeframe: analysis.rung_slopes(snapshot)
            for timeframe, snapshot in snapshots.items()
        }
        market_features = {
            "timeframes": {
                timeframe: {"ladder_slope_pct_per_hour": slopes}
                for timeframe, slopes in rung_slopes.items()
            }
        }
        chosen_weights = weight_model.clean_weights(weights or weight_model.load_settings())
        shadow_signal = weight_model.from_market(market_features, chosen_weights)
        volatility = pipeline.bars_block(snapshots)
        context = {
            "macro_dir": state.macro_dir,
            "micro_shift": state.micro_shift,
            "direction": state.direction,
            "agreement": state.agreement,
            "confidence": state.confidence,
            "drift_pct_per_hour": drifts,
            "ema_gradient_pct_per_hour": rung_slopes,
            "weight_model": shadow_signal,
            "volatility": {
                "status": volatility.get("status"),
                "compression_percentile": (volatility.get("compression") or {}).get("percentile"),
                "timeframes": {
                    timeframe: {
                        "atr_pct": (row or {}).get("atr_pct"),
                        "expected_move_pct": (row or {}).get("expected_move_pct"),
                    }
                    for timeframe, row in (volatility.get("timeframes") or {}).items()
                },
            },
            "ladder_front_back": {
                timeframe: {
                    "front": slopes["EMA10"],
                    "back": slopes["EMA200"],
                }
                for timeframe, slopes in rung_slopes.items()
            },
        }
        event = trade_journal.start_observation(
            side=side,
            short_strike=short.strike,
            hedge_strike=hedge.strike,
            entry_spot=book.spot,
            expiry=book.expiry.date().isoformat(),
            entry_context=context,
            path=TRADE_JOURNAL_PATH,
        )
        label = "bull put" if side == "bull_put" else "bear call"
        status = _card(
            "observation started",
            f"{label} · id {event['observation_id'][:8]} · expiry {book.expiry:%d %b %Y}",
            palette=OK_STYLE,
        )
        return _virtual_result(status)
    except Exception as exc:
        log.warning("virtual observation could not start: %s", exc)
        return _virtual_result(
            _card("observation not started", html.escape(str(exc)), palette=ERROR_STYLE)
        )


def close_virtual_observation(side: str) -> tuple:
    try:
        chain = _live_chain()
        spot = float(chain.snapshot.underlying)
        event = trade_journal.close_observation(
            side=side,
            exit_spot=spot,
            path=TRADE_JOURNAL_PATH,
        )
        label = "DIRECTIONAL SUCCESS" if event["success"] else "DIRECTIONAL FAILURE"
        status = _card(
            f"observation closed · {label}",
            f"{side.replace('_', ' ')} · id {event['observation_id'][:8]}",
            palette=OK_STYLE if event["success"] else ERROR_STYLE,
        )
        return _virtual_result(status)
    except Exception as exc:
        log.warning("virtual observation could not close: %s", exc)
        return _virtual_result(
            _card("observation still open", html.escape(str(exc)), palette=ERROR_STYLE)
        )


def _failure(message: str, scored: tuple | None = None) -> tuple:
    """Every output filled, so Gradio never shows a raw traceback.

    The score panel is passed through untouched: a dead feed has nothing to do
    with whether past projections were any good.
    """
    card = _card("decision", html.escape(message), palette=ERROR_STYLE)
    empty = pd.DataFrame()
    panel = scored if scored is not None else _score_panel()
    failed = "<div style='font-size:18px;font-weight:600;color:#842029'>Run failed</div>"
    shadow = _card(
        "pricing (shadow)",
        "<span style='font-size:14px;font-weight:500'>not measured - the run did not "
        "reach the pricing gate</span>",
        palette=NEUTRAL_STYLE,
    )
    return (failed, card, None, empty, empty, "", empty, shadow) + panel


# --------------------------------------------------------------------------- #
# Callback
# --------------------------------------------------------------------------- #


def refresh(
    symbol: str,
    exchange: str,
    delta: float,
    dte: int,
    equity: float | None,
    risk_percent: float,
    chop_filter: bool,
) -> tuple:
    """Run the pipeline once and return every display component.

    The scorecard is built *first*, from the journal only, so a feed failure
    still leaves the one panel that says whether the projections work.
    """
    symbol = (symbol or "").strip().upper()
    exchange = (exchange or "").strip().upper()
    scored = _score_panel(JOURNAL_PATH, symbol)

    if not TICKER_RE.match(symbol):
        return _failure(
            f"Symbol {symbol!r} is not a valid ticker (A-Z, 0-9, 2-15 chars).", scored
        )
    if not TICKER_RE.match(exchange):
        return _failure(
            f"Exchange {exchange!r} is not valid (A-Z, 0-9, 2-15 chars).", scored
        )

    try:
        payload = pipeline.analyse(
            symbol=symbol,
            exchange=exchange,
            delta=float(delta),
            dte=int(dte),
            equity=None if not equity else float(equity),
            risk_pct=float(risk_percent) / 100.0,
            chop_filter=bool(chop_filter),
            chart=True,
            chart_path=CHART_PATH,
        )
    except analysis.StaleDataError as exc:
        return _failure(f"Stale data, refusing to signal: {exc}", scored)
    except (indicators.DataFetchError, indicators.MissingIndicatorError) as exc:
        return _failure(f"Market data unavailable: {exc}", scored)
    except Exception as exc:  # last resort: the UI must stay usable
        log.exception("unexpected dashboard failure")
        return _failure(f"Unexpected failure: {exc}", scored)

    chart: Image.Image | None = None
    if payload.get("chart"):
        # Copy while the handle is open: the next refresh overwrites the file.
        with Image.open(payload["chart"]) as image:
            chart = image.copy()

    return (
        _header(payload),
        _decision(payload),
        chart,
        _timeframe_table(payload),
        _ladder_table(payload),
        _plan(payload),
        _projection_table(payload),
        _pricing(payload),
    ) + scored


def load_scoreboard(symbol: str) -> tuple:
    """Page-load callback: read the journal only. No network, no scrape quota.

    Opening the dashboard used to run three TradingView fetches per page view,
    which is how the rate limit was exhausted. A page load now costs nothing.
    """
    return _score_panel(JOURNAL_PATH, (symbol or "").strip().upper())


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def build_ui(selected_tab: str | None = None) -> gr.Blocks:
    saved_weights = weight_model.load_settings()
    with gr.Blocks(title="Multi-scale trading agent") as demo:
        gr.Markdown(
            "# Multi-scale trading agent\n"
            "EMA term-structure regime on 15m / 30m / 4h, restricted to "
            "bull put and bear call spreads."
        )

        # Symbol and exchange sit above the tabs: they scope both the live read
        # and the scorecard, and a page-load callback cannot depend on a control
        # that lives inside a tab nobody has opened.
        with gr.Row():
            symbol = gr.Dropdown(
                SYMBOLS, value="ETHUSD", label="Symbol", allow_custom_value=True
            )
            exchange = gr.Dropdown(
                EXCHANGES, value="COINBASE", label="Exchange", allow_custom_value=True
            )

        with gr.Tabs(selected=selected_tab):
            with gr.Tab("Live", id="live"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        delta = gr.Slider(
                            0.10, 0.50, value=0.30, step=0.01, label="Short-leg delta"
                        )
                        dte = gr.Dropdown([7, 14, 30, 45], value=30, label="Days to expiry")
                        equity = gr.Number(value=100_000, label="Account equity (sizing only)")
                        risk = gr.Slider(
                            0.25, 5.0, value=1.0, step=0.25, label="Max risk per trade (%)"
                        )
                        chop = gr.Checkbox(value=True, label="ADX chop filter")
                        run_button = gr.Button("Refresh", variant="primary")

                    with gr.Column(scale=3):
                        # HTML, not Markdown: the header carries the measured
                        # range table, and Gradio's Markdown sanitises styles.
                        header = gr.HTML("Press <b>Refresh</b> to load data.")
                        decision = gr.HTML()
                        chart = gr.Image(
                            label="Term structure and drift",
                            show_label=False,
                            interactive=False,
                        )
                        projection = gr.Dataframe(
                            label="Expected price (diagnostic, not a forecast)",
                            headers=["horizon", "expected", "low", "high", "band %", "read", "from"],
                            interactive=False,
                            wrap=True,
                        )

                timeframe_table = gr.Dataframe(
                    label="Timeframes",
                    headers=[
                        "timeframe",
                        "close",
                        "drift %/h",
                        "ladder slope %/h",
                        "term spread %",
                        "ADX",
                        "ATR %",
                    ],
                    interactive=False,
                    wrap=True,
                )
                ladder = gr.Dataframe(
                    label="EMA ladder (price)",
                    headers=["EMA", "15m", "30m", "4h"],
                    interactive=False,
                    wrap=True,
                )
                plan = gr.Markdown()
                pricing_card = gr.HTML()

            with gr.Tab("Score", id="score"):
                gr.Markdown(
                    "Past projections scored against what price did. "
                    "**Look at `skill`, not `MAE`** - it is the error compared with "
                    "predicting no change. See the About tab before reading these."
                )
                verdict = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        scorecard = gr.Dataframe(
                            label="Scorecard",
                            headers=SCORE_COLUMNS,
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Column(scale=2):
                        health = gr.Dataframe(
                            label="Journal health",
                            headers=HEALTH_COLUMNS,
                            interactive=False,
                            wrap=True,
                        )
                score_button = gr.Button("Reload scorecard (no network)")

            with gr.Tab("Trades", id="trades"):
                gr.Markdown(
            "Virtual directional observations only. No prices, sizes, premiums, or fees to enter. "
                    "Each click starts a separate observation, so you can add positions. "
                    "Exit closes the oldest active observation on that side; click again "
                    "to close the next. This tab never places orders."
                )
                trade_metrics = gr.HTML()
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Bull put · predicts spot stays above the short put")
                        start_put_button = gr.Button("Start bull put observation", variant="primary")
                        close_put_button = gr.Button("Exit oldest bull put")
                    with gr.Column():
                        gr.Markdown("### Bear call · predicts spot stays below the short call")
                        start_call_button = gr.Button("Start bear call observation", variant="primary")
                        close_call_button = gr.Button("Exit oldest bear call")
                gr.Markdown(
                    "**Deterministic structure:** choose the expiry nearest 30 days; sell the "
                    "listed ATM option with a two-sided quote; buy the cheapest two-sided OTM "
                    "option on that wing. Ties go to the closer hedge. At exit, directional success means "
                    "bull put spot is above entry spot or bear call spot is below entry spot. "
                    "An exact tie is not a directional win. "
                    "Every start is kept separately in the active list below. Exit closes "
                    "the oldest item on that side; repeat to close more."
                )
                trade_status = gr.HTML()
                gr.Markdown("### Open observations")
                open_trades_table = gr.Dataframe(
                    headers=OPEN_OBSERVATION_COLUMNS, interactive=False, wrap=True
                )
                gr.Markdown("### Closed observations (latest 100)")
                closed_trades_table = gr.Dataframe(
                    headers=CLOSED_OBSERVATION_COLUMNS, interactive=False, wrap=True
                )
                trade_reload = gr.Button("Reload observations")

            with gr.Tab("Weights", id="weights"):
                gr.Markdown(
                    "### EMA gradient experiment\n"
                    "This is a shadow model. It predicts whether the close four hours "
                    "after a read will be above or below the live listed ATM strike. "
                    "It does not change the production regime or trade buttons."
                )
                gr.Markdown(
                    "Weights combine fast (EMA10/20), middle (EMA30/50), and slow "
                    "(EMA100/200) gradient groups across 15m, 30m, and 4h. Values "
                    "apply to future main.py reads and new virtual entries. Save a "
                    "setting to assign it a version; earlier records keep their version."
                )
                weight_inputs = []
                for timeframe in weight_model.TIMEFRAMES:
                    with gr.Row():
                        gr.Markdown(f"**{timeframe}**")
                        for group in weight_model.GROUPS:
                            weight_inputs.append(
                                gr.Slider(
                                    0,
                                    2,
                                    value=saved_weights[timeframe][group],
                                    step=0.05,
                                    label=f"{group} EMA weight",
                                )
                            )
                save_weights_button = gr.Button("Save weights for future reads")
                weight_settings_status = gr.HTML()
                gr.Markdown(
                    "### Scheduled predictions (non-overlapping 4h outcomes)\n"
                    "Rows group results by saved settings version. The comparison is "
                    "descriptive; a small or time-clustered sample cannot establish a best setting."
                )
                weight_study_status = gr.HTML()
                weight_study_table = gr.Dataframe(
                    headers=WEIGHT_STUDY_COLUMNS, interactive=False, wrap=True
                )
                gr.Markdown("### Virtual entries (outcome at your Exit click)")
                manual_weight_table = gr.Dataframe(
                    headers=MANUAL_WEIGHT_COLUMNS, interactive=False, wrap=True
                )
                reload_weight_study = gr.Button("Reload weight study (no network)")

            with gr.Tab("About", id="about"):
                gr.Markdown(ABOUT)

        gr.Markdown(f"---\n*{DISCLAIMER}*")

        inputs = [symbol, exchange, delta, dte, equity, risk, chop]
        live_outputs = [
            header,
            decision,
            chart,
            timeframe_table,
            ladder,
            plan,
            projection,
            pricing_card,
        ]
        score_outputs = [verdict, scorecard, health]
        outputs = live_outputs + score_outputs

        # api_name=False keeps these off the HTTP API surface.
        run_button.click(
            refresh,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
            api_name=False,
        )
        score_button.click(
            load_scoreboard,
            inputs=[symbol],
            outputs=score_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        trade_outputs = [trade_status, trade_metrics, open_trades_table, closed_trades_table]
        start_put_button.click(
            lambda *values: start_virtual_observation("bull_put", _weight_values(list(values))),
            inputs=weight_inputs,
            outputs=trade_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        start_call_button.click(
            lambda *values: start_virtual_observation("bear_call", _weight_values(list(values))),
            inputs=weight_inputs,
            outputs=trade_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        close_put_button.click(
            lambda: close_virtual_observation("bull_put"),
            inputs=[],
            outputs=trade_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        close_call_button.click(
            lambda: close_virtual_observation("bear_call"),
            inputs=[],
            outputs=trade_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        trade_reload.click(
            _observation_panel,
            inputs=[],
            outputs=trade_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        save_weights_button.click(
            save_weight_settings,
            inputs=weight_inputs,
            outputs=[weight_settings_status],
            concurrency_limit=1,
            api_name=False,
        )
        reload_weight_study.click(
            _weight_study_panel,
            inputs=[],
            outputs=[weight_study_status, weight_study_table, manual_weight_table],
            concurrency_limit=1,
            api_name=False,
        )
        # Changing the symbol should re-read the journal: it costs nothing, and a
        # scorecard for the previous symbol left on screen is actively misleading.
        symbol.change(
            load_scoreboard,
            inputs=[symbol],
            outputs=score_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        # Page load reads the journal only. It must never scrape: every browser
        # refresh used to cost three TradingView requests.
        demo.load(
            load_scoreboard,
            inputs=[symbol],
            outputs=score_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        demo.load(
            _observation_panel,
            inputs=[],
            outputs=trade_outputs,
            concurrency_limit=1,
            api_name=False,
        )
        demo.load(
            _weight_study_panel,
            inputs=[],
            outputs=[weight_study_status, weight_study_table, manual_weight_table],
            concurrency_limit=1,
            api_name=False,
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local trading dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--journal",
        default=str(JOURNAL_PATH),
        help="journal to score (read-only; the dashboard never writes it)",
    )
    parser.add_argument(
        "--trade-journal",
        default=str(TRADE_JOURNAL_PATH),
        help="append-only virtual observation journal",
    )
    parser.add_argument("--open", action="store_true", help="open a browser tab")
    parser.add_argument(
        "--tab",
        choices=["Live", "Score", "Trades", "Weights", "About"],
        default=None,
        help="tab to open first",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global JOURNAL_PATH, TRADE_JOURNAL_PATH

    args = build_parser().parse_args(argv)
    JOURNAL_PATH = Path(args.journal)
    TRADE_JOURNAL_PATH = Path(args.trade_journal)

    logging.basicConfig(level="WARNING", format="%(levelname)s %(message)s")

    demo = build_ui(args.tab.lower() if args.tab else None)
    demo.queue(default_concurrency_limit=1)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "binding %s exposes this dashboard, including the equity field, to your "
            "network WITHOUT authentication; prefer an SSH tunnel",
            args.host,
        )

    print(f"dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            inbrowser=args.open,
            share=False,  # never: an unauthenticated public tunnel to a scraper
        )
    except OSError as exc:
        # Overwhelmingly the common failure: another dashboard still running.
        # Gradio's own message buries that fact under a 20-frame traceback.
        if "empty port" not in str(exc):
            raise
        print(
            f"port {args.port} is already in use - another instance is probably still "
            f"running.\n"
            f"  find it:  ss -ltnp | grep :{args.port}\n"
            f"  kill it:  kill <pid>\n"
            f"  or pass:  --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
