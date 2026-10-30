"""Single entry point for the trading agent.

This module owns the data inflow and hands it to ``analysis`` in one pass:

    fetch (indicators) -> validate freshness (analysis.guard)
        -> regime + term structure (analysis)
            -> credit-spread plan (analysis)
                -> chart (plot) + JSON

Design rules:

* **One scrape per run.** ``analysis`` and ``plot`` no longer have their own
  CLIs; before this file existed, three entry points each fetched the same
  three timeframes independently.
* **Failures are exit codes, not tracebacks.** ``2`` means stale data, ``1``
  means the feed was unavailable. That makes the script safe to schedule.
* **No trading without a readable reason.** A neutral regime or an unaffordable
  structure prints *why*, and the JSON carries the same text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import analysis
import bars
import indicators
import iv
import pricing

log = logging.getLogger("main")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STALE = 2
EXIT_INTERRUPTED = 130

#: Where the append-only bar store lives. **Anchored to the module, never to the
#: working directory.** As a relative path it silently resolved against whatever
#: directory the caller started from: launching the dashboard from `src/` created a
#: second, ten-day store there, and the backfilled year was invisible to it.
BARS_ROOT = Path(__file__).resolve().parent.parent / "out" / "bars"

#: An EMA is compared with the feed's only once it has ``period * this`` bars
#: behind it. A seed that is still bleeding out is not evidence either way.
CROSS_CHECK_WARMUP = 3

#: Only the tail of the store is read. Ten days of 15m bars is 960 rows; keeping
#: years of ticks in one file to compute a 14-bar ATR is not worth the read.
STORE_TAIL = 5_000

#: Points of Keltner-width history carried into the payload for the chart panel.
#: 120 bars is 30 h at 15m: enough to see a squeeze form, small enough to ship.
CHART_WIDTH_POINTS = 120


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def collect(
    *,
    symbol: str = indicators.DEFAULT_SYMBOL,
    exchange: str = indicators.DEFAULT_EXCHANGE,
    screener: str = indicators.DEFAULT_SCREENER,
    ttl: float = 60.0,
    max_age: float = 3600.0,
) -> dict[str, indicators.Snapshot]:
    """Pull every timeframe, then refuse to continue on stale prices.

    Raises ``analysis.StaleDataError`` if the data is too old, and
    ``indicators.DataFetchError`` if the feed could not be reached.
    """
    snapshots = indicators.call_any_analysis(
        symbol=symbol, exchange=exchange, screener=screener, ttl=ttl
    )
    analysis.guard(snapshots, max_age_seconds=max_age)
    return snapshots


def default_chain(*, source: str = "deribit") -> tuple[analysis.OptionsChain, str | None]:
    """The option chain to price against, and why it is not the live one.

    Falling back to the stub is not an error: a missing chain must not cost you
    the market read. It *is* a fact the payload has to carry, or a synthetic
    credit and a market credit become indistinguishable on screen.
    """
    if source == "synthetic":
        return analysis.SyntheticChain(), "synthetic chain requested (--chain synthetic)"
    try:
        snapshot = iv.fetch_chain()
    except Exception as exc:  # network, parse, empty: none of them justify a crash
        log.warning("live option chain unavailable (%s); using the synthetic stub", exc)
        return analysis.SyntheticChain(), f"live chain unavailable: {exc}"
    return iv.DeribitChain(snapshot), None


def bars_block(
    snapshots: Mapping[str, Any],
    *,
    root: str | Path = BARS_ROOT,
    product: str = bars.DEFAULT_PRODUCT,
    enabled: bool = True,
) -> dict[str, Any]:
    """Range and compression from real OHLCV, on a fail-soft path.

    The scanner gives 91 scalars and no candle history, so until now every band
    width was a convention. This fetches one klines page, appends it to the
    store, and derives ATR, Keltner and Bollinger widths, a compression
    percentile and the realised 1-sigma move at each published horizon.

    It never raises. A missing bar feed must cost the run its *precision*, not
    its existence -- but the payload says ``status: unavailable`` and the reason,
    so a wide band never looks like a measured one.
    """
    block: dict[str, Any] = {
        "status": "unavailable",
        "source": "binance klines",
        "product": product.upper(),
        "basis_note": (
            f"bars are {product.upper()} on Binance; the traded symbol and the "
            "option chain are ETH. Cross-venue basis is not modelled."
        ),
    }
    if not enabled:
        block["reason"] = "disabled for this run (--no-bars)"
        return block

    path = bars.store_path(root, product=product)
    try:
        fresh = bars.fetch_klines(product=product)
        written = bars.append_bars(path, fresh)
        everything = bars.load_bars(path)
        stored = everything[-STORE_TAIL:]
    except bars.BarError as exc:
        log.warning("bar feed unavailable (%s); range measures omitted", exc)
        block["reason"] = str(exc)
        return block

    if not stored:
        block["reason"] = "the bar store is empty"
        return block

    block["status"] = "ok"
    block["store"] = str(path)
    block["appended"] = written
    block["stored"] = len(everything)
    block["last_closed_bar"] = stored[-1].open_time.isoformat()
    block["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # Realised vol over the windows the pricing shadow scores its edge against.
    # Computed from the *whole* store, not the tail the chart needs: a one-year
    # window cannot be measured from 5,000 bars, and a window silently shortened
    # is a different number wearing the same label.
    import pricing

    block["volatility"] = pricing.volatility_profile(everything)

    series: dict[str, list[bars.Bar]] = {"15m": stored}
    for timeframe, factor in (("30m", 2), ("4h", 16)):
        series[timeframe] = bars.resample(stored, factor, "15m")

    timeframes: dict[str, Any] = {}
    for timeframe, candles in series.items():
        hours = bars.INTERVAL_MINUTES[timeframe] / 60.0
        timeframes[timeframe] = {
            "bars": len(candles),
            "atr_pct": _rounded(bars.atr_pct(candles, 14), 4),
            "keltner_width_pct": _rounded(bars.keltner_width_pct(candles), 4),
            "bollinger_width_pct": _rounded(bars.bollinger_width_pct(candles), 4),
            "expected_move_pct": {
                analysis.horizon_label(horizon): _rounded(
                    bars.expected_move_pct(candles, horizon), 4
                )
                for horizon in analysis.PROJECTION_HORIZONS
            },
            "bar_hours": hours,
        }
    block["timeframes"] = timeframes

    # Compression: today's Keltner width against the last 200 of its own past.
    # This is the one number that says "the market is coiled", and it needs no
    # threshold to be meaningful -- only a rank.
    widths = bars.rolling_keltner_width_pct(stored, window=200)
    current = bars.keltner_width_pct(stored)
    rank = bars.percentile_rank(widths, current) if current is not None else None
    block["compression"] = {
        "percentile": _rounded(rank, 1),
        "samples": len(widths),
        "reads": _compression_read(rank),
        # The shape, not just the rank: the chart's third panel draws this. It
        # travels in the payload so the plot stays a pure function of the run.
        "series": [round(value, 4) for value in widths[-CHART_WIDTH_POINTS:]],
        "p10": _rounded(_percentile(widths, 10), 4),
        "p90": _rounded(_percentile(widths, 90), 4),
    }

    checks: dict[str, Any] = {}
    for timeframe, candles in series.items():
        needed = {
            key: int(key.removeprefix("EMA"))
            for key in analysis.MA_KEYS
            if len(candles) >= int(key.removeprefix("EMA")) * CROSS_CHECK_WARMUP
        }
        if not needed:
            checks[timeframe] = {"status": "not enough bars", "bars": len(candles)}
            continue
        try:
            feed = analysis.ema_values(snapshots[timeframe])
        except (KeyError, indicators.MissingIndicatorError) as exc:
            checks[timeframe] = {"status": "no feed values", "reason": str(exc)}
            continue
        theirs = {key: float(value) for key, value in zip(analysis.MA_KEYS, feed)}
        try:
            checks[timeframe] = bars.cross_check(candles, theirs, periods=needed)
        except bars.BarError as exc:
            # Not fatal: the feed is the primary read and the store is the
            # secondary. A disagreement has to be loud, not decisive.
            log.warning("EMA cross-check failed for %s: %s", timeframe, exc)
            checks[timeframe] = {"status": "mismatch", "reason": str(exc)}
    block["cross_check"] = checks
    if any(check.get("status") == "mismatch" for check in checks.values()):
        block["status"] = "mismatch"
        block["reason"] = (
            "local EMAs disagree with the feed; range measures are reported but "
            "must not be trusted until the series is reconciled"
        )
    return block


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def _compression_read(rank: float | None) -> str:
    """A percentile is a rank, so it needs no arbitrary threshold to be read."""
    if rank is None:
        return "unranked"
    if rank <= 10:
        return "very compressed"
    if rank <= 25:
        return "compressed"
    if rank >= 90:
        return "very expanded"
    if rank >= 75:
        return "expanded"
    return "normal"


def _percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear-interpolated percentile. Small samples, so no numpy dependency."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * max(0.0, min(100.0, percent)) / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)



def evaluate(
    snapshots: Mapping[str, Any],
    *,
    delta: float = 0.30,
    dte: int = 30,
    equity: float | None = None,
    risk_pct: float = 0.01,
    chop_filter: bool = True,
    chain: analysis.OptionsChain | None = None,
) -> tuple[analysis.Regime, dict[str, Any]]:
    """Read the regime, then plan a credit spread against it."""
    regime = analysis.regime(snapshots, use_chop_filter=chop_filter)
    result = analysis.trade_scenario(
        regime,
        analysis.spot(snapshots["15m"]),
        chain if chain is not None else analysis.SyntheticChain(),
        target_delta=delta,
        dte_days=dte,
        equity=equity,
        max_risk_pct=risk_pct,
    )
    return regime, result


def _market(snapshots: Mapping[str, Any], ranges: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Per-timeframe numbers behind the regime call, for the record."""
    measured = (ranges or {}).get("timeframes") or {}
    rows: dict[str, Any] = {}
    for timeframe, snapshot in snapshots.items():
        structure = analysis.term_structure(snapshot)
        adx = analysis.indicators_of(snapshot).get("ADX")
        ladder = analysis.ema_values(snapshot)
        bars_at_tf = measured.get(timeframe) or {}
        rows[timeframe] = {
            "close": round(structure["spot"], 2),
            # Clock-side trend rate: positive means price is rising. This is the
            # number the regime reads and the projection extrapolates.
            "drift_pct_per_hour": round(analysis.drift_pct_per_hour(snapshot), 5),
            # Lookback-side derivative, kept for the record. Negated by drift.
            "ladder_slope_pct_per_hour": {
                key: round(value, 5)
                for key, value in analysis.rung_slopes(snapshot).items()
            },
            "spectral_spread_pct": round(structure["spectral_spread_pct"], 4),
            "curvature_pct": round(structure["curvature_pct"], 4),
            "adx": round(float(adx), 2) if isinstance(adx, (int, float)) else None,
            # Measured, not assumed: None until the bar feed has warmed up.
            "atr_pct": bars_at_tf.get("atr_pct"),
            "emas": {
                key: round(float(value), 2)
                for key, value in zip(analysis.MA_KEYS, ladder)
            },
        }
    return {"spot": round(analysis.spot(snapshots["15m"]), 2), "timeframes": rows}


def _freshness(snapshots: Mapping[str, Any], limit: float) -> dict[str, Any]:
    ages = {
        tf: snap.age_seconds()
        for tf, snap in snapshots.items()
        if callable(getattr(snap, "age_seconds", None))
    }
    sources = {getattr(s, "as_of_source", "unknown") for s in snapshots.values()}
    verified = bool(sources) and sources == {"exchange"}

    record: dict[str, Any] = {
        "status": "verified" if verified else "unverified",
        "as_of_source": sorted(sources)[0] if len(sources) == 1 else sorted(sources),
        "oldest_age_seconds": round(max(ages.values()), 1) if ages else None,
        "limit_seconds": limit,
    }
    if not verified:
        # Be blunt: the upstream scraper stamps its own clock, so age tells us
        # how long ago *we asked*, not how fresh the candle is.
        record["note"] = (
            "upstream supplies no exchange timestamp; candlestick freshness "
            "cannot be confirmed"
        )
    return record


def build_payload(
    *,
    symbol: str,
    exchange: str,
    snapshots: Mapping[str, Any],
    regime: analysis.Regime,
    result: Mapping[str, Any],
    max_age: float,
    chart_path: str | None = None,
    acceleration: float | None = None,
    chain: analysis.OptionsChain | None = None,
    chain_note: str | None = None,
    dte: int = 30,
    bars_summary: Mapping[str, Any] | None = None,
    pricing_block: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    projections = analysis.project_all(snapshots, acceleration_pct_per_hour2=acceleration)
    # Never None: if the caller did not name a chain, ``evaluate`` priced this
    # run off the synthetic stub, and the payload has to admit that rather than
    # leave the provenance blank.
    priced_with = iv.describe(
        chain if chain is not None else analysis.SyntheticChain(),
        dte_days=dte,
        spot=analysis.spot(snapshots["15m"]),
        note=chain_note,
    )
    atm_reference: dict[str, Any] = {"status": "unavailable"}
    chain_snapshot = getattr(chain, "snapshot", None)
    expected_currency = symbol.upper().removesuffix("USD")
    if chain_snapshot is not None and str(getattr(chain_snapshot, "currency", "")).upper() == expected_currency:
        try:
            book = chain.book("call", dte_days=dte)
            liquid = [leg for leg in book.legs if leg.two_sided]
            if liquid:
                atm = min(liquid, key=lambda leg: (abs(leg.strike - book.spot), leg.strike))
                atm_reference = {
                    "status": "ok",
                    "source": "deribit",
                    "strike": float(atm.strike),
                    "expiry": book.expiry.date().isoformat(),
                    "horizon_hours": 4.0,
                }
        except Exception as exc:
            atm_reference = {"status": "unavailable", "reason": str(exc)}
    return {
        "symbol": symbol,
        "exchange": exchange,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freshness": _freshness(snapshots, max_age),
        "market": _market(snapshots, bars_summary),
        "projection": [projection.to_dict() for projection in projections],
        "range": bars_summary or {"status": "unavailable", "reason": "not requested"},
        "analysis": result["regime"],
        "plan": result["plan"],
        "pricing": dict(pricing_block) if pricing_block else {"status": "unavailable"},
        "chain": priced_with,
        # Frozen listed ATM reference for the shadow weight experiment. The
        # four-hour outcome is resolved later from stored OHLCV bars.
        "atm_reference": atm_reference,
        "decision": result["decision"],
        "reason": result["reason"],
        "chart": chart_path,
        "disclaimer": result["disclaimer"],
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _pretty_ts(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(value)


def summarize(payload: Mapping[str, Any]) -> str:
    """A compact, human-readable read of the same payload the JSON carries."""
    market = payload["market"]
    fresh = payload["freshness"]
    reg = payload["analysis"]
    lines: list[str] = []

    lines.append(
        f"{payload['symbol']} @ {payload['exchange']}"
        f"   spot {market['spot']:,.2f}"
        f"   as of {_pretty_ts(reg['as_of'])}"
    )

    age = fresh["oldest_age_seconds"]
    stamp = "verified" if fresh["status"] == "verified" else "UNVERIFIED"
    lines.append(
        f"freshness  {stamp}"
        + (f"   oldest {age:.0f}s / limit {fresh['limit_seconds']:.0f}s" if age is not None else "")
    )
    if fresh.get("note"):
        lines.append(f"           {fresh['note']}")

    lines.append("")
    lines.append(f"  {'tf':<5}{'drift %/h':>12}{'spread %':>11}{'ADX':>8}{'ATR %':>8}")
    for timeframe, row in market["timeframes"].items():
        adx = "-" if row["adx"] is None else f"{row['adx']:.1f}"
        atr = "-" if row.get("atr_pct") is None else f"{row['atr_pct']:.2f}"
        lines.append(
            f"  {timeframe:<5}{row['drift_pct_per_hour']:>+12.4f}"
            f"{row['spectral_spread_pct']:>11.2f}{adx:>8}{atr:>8}"
        )

    ranges = payload.get("range") or {}
    if ranges.get("status") in {"ok", "mismatch"}:
        measured = ranges.get("timeframes") or {}
        compression = ranges.get("compression") or {}
        atr = "  ".join(
            f"{tf} {row['atr_pct']:.2f}%"
            for tf, row in measured.items()
            if row.get("atr_pct") is not None
        )
        rank = compression.get("percentile")
        lines.append("")
        lines.append(
            f"  range      ATR {atr or 'n/a'}"
            + (f"   compression {rank:.0f}th pctile ({compression['reads']})" if rank is not None else "")
        )
        for projection in payload.get("projection") or []:
            sigma = (
                (measured.get(projection["basis_timeframe"]) or {})
                .get("expected_move_pct", {})
                .get(projection["horizon"])
            )
            if sigma is None:
                continue
            verdict = (
                "narrower than a random walk"
                if projection["band_pct"] < sigma
                else "at or wider than a random walk"
            )
            lines.append(
                f"             band {projection['horizon']} ±{projection['band_pct']:.2f}%"
                f" vs 1-sigma {sigma:.2f}%  -> {verdict}"
            )
        for timeframe, check in (ranges.get("cross_check") or {}).items():
            if check.get("status") == "mismatch":
                lines.append(f"  bars       MISMATCH {timeframe}: {check.get('reason')}")
    else:
        lines.append("")
        lines.append(f"  range      unmeasured ({ranges.get('reason', 'not requested')})")

    lines.append("")
    lines.append(f"  macro      {reg['macro_dir']}   strength {reg['macro_strength']:.2f}")
    lines.append(f"  micro      {reg['micro_shift']}")
    lines.append(
        f"  direction  {reg['direction'].upper()}   confidence {reg['confidence']:.0%}"
        f"   ({reg['agreement']}/3 timeframes agree)"
    )
    for reason in reg["reasons"]:
        lines.append(f"             - {reason}")

    projections = payload.get("projection") or []
    if projections:
        lines.append("")
        lines.append(f"  expected   {'at':<5}{'price':>11}   {'band (±)':<23}{'read':<8}")
        for projection in projections:
            lines.append(
                f"             {projection['horizon']:<5}{projection['expected_price']:>11,.2f}"
                f"   {projection['low']:,.2f} .. {projection['high']:,.2f}      "
                f"{projection['direction']}"
            )
        lines.append(
            "             diagnostic only: extrapolated from the ladder, "
            "not scored against realised prices"
        )

    lines.append("")
    lines.append(f"  decision   {payload['decision']}")
    lines.append(f"  reason     {payload['reason']}")

    priced = payload.get("chain")
    if priced:
        source = priced.get("source", "unknown").upper()
        iv_atm = priced.get("iv_atm")
        detail = f"   ATM IV {iv_atm:.1%}" if isinstance(iv_atm, (int, float)) else ""
        quotes = priced.get("quotes")
        if isinstance(quotes, int):
            detail += f"   {quotes} instruments"
        lines.append(f"  chain      {source}{detail}")
        if priced.get("note"):
            lines.append(f"             - {priced['note']}")

    shadow_block = payload.get("pricing") or {}
    if shadow_block:
        lines.append("")
        lines.append(f"  {pricing.describe(shadow_block)}")
        for row in (shadow_block.get("sides") or {}).values():
            if "short_strike" not in row:
                continue
            lines.append(
                f"             {row['strategy']:<16}"
                f" {row['short_strike']:,.0f}/{row['long_strike']:,.0f}"
                f"   credit {row['credit']:6.2f} ({row['credit_ratio']:>4.0%} of width)"
                f"   edge {row['edge']:+6.2f}"
                f"   keep {row['p_keep']:>4.0%} part {row['p_partial']:>4.0%}"
                f" lose {row['p_max_loss']:>4.0%}"
            )
        if shadow_block.get("status") == "ok" and shadow_block.get("verdict") == "NO TRADE":
            lines.append("             shadow verdict is a record, not a gate: nothing here is emitted")

    plan = payload["plan"]
    if plan:
        kind = "put" if "put" in plan["strategy"] else "call"
        lines.append("")
        lines.append(
            f"  plan       {plan['strategy']}   sell {plan['short_strike']:,.2f} {kind} /"
            f" buy {plan['long_strike']:,.2f} {kind}"
        )
        lines.append(
            f"             credit {plan['credit']:.2f}   width {plan['width']:.2f}"
            f"   max loss {plan['max_loss_per_contract']:.2f}/contract"
            f"   ({plan['credit_ratio']:.0%} of width)"
        )
        lines.append(
            f"             breakeven {plan['breakeven']:,.2f}   POP {plan['pop']:.0%}"
            f"   (partial {plan['pop_partial']:.0%} / max loss {plan['pop_max_loss']:.0%})"
            f"   expiring in {plan['dte_days']}d   contracts {plan['contracts']}"
        )
        if plan.get("credit_source") in {"book", "mark"}:
            lines.append(
                f"             credit source {plan['credit_source']}"
                f" (mark {plan.get('credit_mark', 0.0):.2f})"
                f"   expiry {plan.get('expiry') or 'n/a'}"
                f"   DTE {plan.get('dte_actual') or plan['dte_days']}"
            )
        for note in plan["notes"]:
            lines.append(f"             - {note}")

    if payload.get("chart"):
        lines.append("")
        lines.append(f"  chart      {payload['chart']}")

    recorded = payload.get("journal")
    if recorded:
        if recorded["status"] == "ok":
            lines.append(f"  journal    {recorded['path']}")
        else:
            lines.append(f"  journal    FAILED ({recorded['error']}) - the analysis above still stands")

    lines.append("")
    lines.append(f"  {payload['disclaimer']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def analyse(
    *,
    symbol: str = indicators.DEFAULT_SYMBOL,
    exchange: str = indicators.DEFAULT_EXCHANGE,
    screener: str = indicators.DEFAULT_SCREENER,
    delta: float = 0.30,
    dte: int = 30,
    equity: float | None = None,
    risk_pct: float = 0.01,
    chop_filter: bool = True,
    max_age: float = 3600.0,
    ttl: float = 60.0,
    chart: bool = False,
    chart_path: str | Path | None = None,
    title: str | None = None,
    chain: analysis.OptionsChain | None = None,
    chain_source: str = "deribit",
    acceleration: float | None = None,
    journal_path: str | Path | None = None,
    use_bars: bool = True,
    bars_root: str | Path = BARS_ROOT,
) -> dict[str, Any]:
    """The whole pipeline as one keyword call.

    This is the entry point for non-CLI callers (the dashboard); ``run`` is just
    argparse plumbing around it. ``chart=True`` renders a PNG, ``chart_path``
    chooses where (``None`` lets ``plot`` pick a default name).

    ``chain`` overrides the option chain; otherwise one is built from
    ``chain_source``. ``chain_source="deribit"`` prices against the live chain
    and falls back to the synthetic stub (recorded in ``payload["chain"]``)
    rather than failing the run.

    ``acceleration`` is in %/hour^2 and can only come from successive runs; when
    it is omitted the price projection is first-order and says so.

    ``journal_path`` appends one JSONL line per run so ``score.py`` can later
    compare the projections with realised prices. A journal failure is recorded
    in the payload and logged, never fatal: losing a line is not losing a run.

    ``use_bars`` fetches one page of real klines and derives ATR, width and
    realised vol. It is fail-soft by design: without it the run still produces a
    decision, with the band left at its documented floor.
    """
    snapshots = collect(
        symbol=symbol,
        exchange=exchange,
        screener=screener,
        ttl=ttl,
        max_age=max_age,
    )

    ranges = bars_block(snapshots, root=bars_root, enabled=use_bars)

    active_chain, chain_note = (
        (chain, None) if chain is not None else default_chain(source=chain_source)
    )

    regime, result = evaluate(
        snapshots,
        delta=delta,
        dte=dte,
        equity=equity,
        risk_pct=risk_pct,
        chop_filter=chop_filter,
        chain=active_chain,
    )

    # The second opinion. It is measured, journaled and shown, and it never
    # changes ``result``: with lift 0.91x on the direction signal there is no
    # basis yet for letting a second gate override it, or for letting this one
    # emit a candidate of its own.
    direction_side = None
    if result.get("plan"):
        direction_side = "put" if result["plan"].get("strategy") == "bull_put_spread" else "call"
    try:
        pricing_block = pricing.shadow(
            analysis.spot(snapshots["15m"]),
            active_chain,
            profile=(ranges or {}).get("volatility"),
            dte_days=dte,
            direction_side=direction_side,
        )
    except Exception as exc:  # a shadow that breaks must not cost the run
        log.warning("pricing shadow failed: %s", exc)
        pricing_block = {
            "status": "error",
            "mode": "shadow",
            "reason": str(exc),
            "verdict": "NO TRADE",
            "sides": {},
            "pick": None,
            "agrees_with_direction": None,
        }

    written: str | None = None
    if chart:
        # Imported lazily: matplotlib is only needed when a chart is wanted.
        import plot as charting

        try:
            written = str(
                charting.render(
                    snapshots,
                    regime,
                    path=chart_path,
                    title=title or f"{symbol} multi-scale dynamical analysis",
                    ranges=ranges,
                )
            )
        except Exception as exc:  # a chart failure must not sink the analysis
            log.warning("chart could not be rendered: %s", exc)

    payload = build_payload(
        symbol=symbol,
        exchange=exchange,
        snapshots=snapshots,
        regime=regime,
        result=result,
        max_age=max_age,
        chart_path=written,
        acceleration=acceleration,
        chain=active_chain,
        chain_note=chain_note,
        dte=dte,
        bars_summary=ranges,
        pricing_block=pricing_block,
    )

    if journal_path:
        # Imported lazily, like the chart: an unused module costs nothing.
        import journal as journal_store

        try:
            record = journal_store.record_from_payload(payload)
            journal_store.append(journal_path, record)
            payload["journal"] = {
                "status": "ok",
                "path": str(journal_path),
                "collected_at": record["collected_at"],
            }
        except Exception as exc:  # a lost journal line must not sink the analysis
            log.warning("journal could not be written: %s", exc)
            payload["journal"] = {
                "status": "error",
                "path": str(journal_path),
                "error": str(exc),
            }

    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI adapter: argparse namespace -> :func:`analyse`."""
    return analyse(
        symbol=args.symbol,
        exchange=args.exchange,
        screener=args.screener,
        delta=args.delta,
        dte=args.dte,
        equity=args.equity,
        risk_pct=args.risk_pct,
        chop_filter=not args.no_chop_filter,
        max_age=args.max_age,
        ttl=args.ttl,
        chart=not args.no_chart,
        chart_path=args.chart,
        chain_source=args.chain,
        journal_path=args.journal,
        use_bars=not args.no_bars,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect market data, read the regime, plan a credit spread.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default=indicators.DEFAULT_SYMBOL)
    parser.add_argument("--exchange", default=indicators.DEFAULT_EXCHANGE)
    parser.add_argument("--screener", default=indicators.DEFAULT_SCREENER)
    parser.add_argument("--delta", type=float, default=0.30, help="short-leg target delta")
    parser.add_argument("--dte", type=int, default=30, help="days to expiry")
    parser.add_argument("--equity", type=float, default=None, help="account equity for sizing")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="max fraction of equity per trade")
    parser.add_argument("--max-age", type=float, default=3600.0, help="refuse data older than this")
    parser.add_argument("--ttl", type=float, default=60.0, help="fetch cache lifetime")
    parser.add_argument("--no-chop-filter", action="store_true", help="ignore ADX")
    parser.add_argument("--no-chart", action="store_true", help="skip the PNG")
    parser.add_argument(
        "--no-bars",
        action="store_true",
        help="skip the OHLCV fetch, leaving ATR and width unmeasured",
    )
    parser.add_argument("--chart", default=None, help="explicit chart output path")
    parser.add_argument("--json", default=None, help="write the payload to this path")
    parser.add_argument(
        "--chain",
        choices=["deribit", "synthetic"],
        default="deribit",
        help="price the spread off the live Deribit chain, or the stub",
    )
    parser.add_argument(
        "--journal",
        default=None,
        help="append this run to a JSONL journal for later scoring "
        "(e.g. out/projections.jsonl); off by default",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def _fail(message: str) -> None:
    """Fatal errors go to stderr directly, not through logging configuration."""
    print(f"error: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    try:
        payload = run(args)
    except analysis.StaleDataError as exc:
        _fail(str(exc))
        return EXIT_STALE
    except (indicators.DataFetchError, indicators.MissingIndicatorError) as exc:
        _fail(f"market data unavailable: {exc}")
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        _fail("interrupted")
        return EXIT_INTERRUPTED

    print(summarize(payload))

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON written to {destination}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
