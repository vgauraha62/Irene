"""Real OHLCV bars, so range and volatility stop being conventions.

The TradingView scanner supplies 91 scalars and no candle history: no ATR, no
percentiles, and a freshness stamp that is the scraper's own clock. One public
klines request returns 1,000 completed bars with *exchange* timestamps, which is
10 days of 15m history for one HTTP call and no key.

Three rules, all learned the hard way elsewhere in this repo:

* **Store raw, derive on read.** Only OHLCV hits disk. ATR, widths, percentiles
  and realised vol are computed every time, so fixing a formula does not throw
  away the history that proves it was wrong.
* **Append only, dedupe on open time.** A re-fetch of the same window must not
  inflate the sample.
* **The last kline is still forming.** Binance includes the current, incomplete
  candle; using it makes ATR and every percentile depend on when you polled.

The product is ``ETHUSDT`` on Binance while the agent trades ``ETHUSD`` on
Coinbase and prices options on Deribit. That is a real basis risk, recorded in
the payload rather than smoothed away: basis is small in calm markets and is not
small when it matters.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

log = logging.getLogger("bars")

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

#: The exchange product, not the traded symbol. Recorded in every payload.
DEFAULT_PRODUCT = "ETHUSDT"
DEFAULT_INTERVAL = "15m"

INTERVAL_SECONDS: dict[str, int] = {"15m": 900, "30m": 1_800, "4h": 14_400}
INTERVAL_MINUTES: dict[str, float] = {name: secs / 60.0 for name, secs in INTERVAL_SECONDS.items()}

COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


class BarError(RuntimeError):
    """Bars could not be fetched, parsed or stored."""


@dataclass(frozen=True)
class Bar:
    open_time: datetime  # tz-aware UTC, the bar's OPEN time
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_flat(self) -> bool:
        return self.high == self.low


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def parse_klines(rows: Iterable[Sequence[Any]]) -> list[Bar]:
    """Binance's positional rows -> bars. Rejects anything that is not a price.

    The positional schema is ``[openTime, o, h, l, c, v, closeTime, ...]``. A
    longer or shorter row is a schema change, and dropping it quietly would
    leave a hole in a series whose whole value is being contiguous.
    """
    bars: list[Bar] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise BarError(f"row {index} is not a kline: {row!r}")
        try:
            bar = Bar(
                open_time=_as_datetime(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        except (TypeError, ValueError) as exc:
            raise BarError(f"row {index} is not numeric: {exc}") from None
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            raise BarError(f"row {index} has a non-positive price")
        if bar.high < bar.low:
            raise BarError(f"row {index} has high < low")
        bars.append(bar)

    bars.sort(key=lambda bar: bar.open_time)
    for earlier, later in zip(bars, bars[1:]):
        if earlier.open_time == later.open_time:
            raise BarError(f"duplicate open_time {earlier.open_time.isoformat()}")
    return bars


def _get(url: str, params: Mapping[str, Any], *, timeout: float, retries: int, backoff: float) -> Any:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "trading-agent/1.0", "Accept": "application/json"}
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise BarError(f"{params}: {last}")


def fetch_klines(
    *,
    product: str = DEFAULT_PRODUCT,
    interval: str = DEFAULT_INTERVAL,
    limit: int = 1000,
    timeout: float = 8.0,
    retries: int = 2,
    backoff: float = 1.5,
    settled: bool = True,
    end_time_ms: int | None = None,
) -> list[Bar]:
    """Fetch closed bars. ``settled`` drops the in-progress final candle.

    ``end_time_ms`` walks backwards: Binance returns the ``limit`` bars that
    *close at or before* that millisecond. That is the whole pagination
    mechanism -- there is no cursor and no page limit, only a weight budget.
    """
    if interval not in INTERVAL_SECONDS:
        raise BarError(f"unsupported interval {interval!r}")
    if not 1 <= limit <= 1000:
        raise BarError(f"limit must be 1..1000, got {limit}")

    params: dict[str, Any] = {
        "symbol": product.upper(),
        "interval": interval,
        "limit": limit,
    }
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    payload = _get(
        BINANCE_KLINES,
        params,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )
    if not isinstance(payload, list):
        raise BarError(f"unexpected kline payload: {str(payload)[:80]}")
    bars = parse_klines(payload)

    if settled and bars and end_time_ms is None:
        now = datetime.now(timezone.utc)
        span = timedelta(seconds=INTERVAL_SECONDS[interval])
        if bars[-1].open_time + span > now:
            bars.pop()
    return bars


def backfill(
    path: str | Path,
    *,
    product: str = DEFAULT_PRODUCT,
    interval: str = DEFAULT_INTERVAL,
    pages: int = 10,
    limit: int = 1000,
    pause: float = 0.25,
    **fetch_kwargs: Any,
) -> dict[str, Any]:
    """Walk backwards in history and append, page by page.

    A base rate needs enough independent windows to mean anything: ten days of
    15m bars is ~60 non-overlapping 4h windows, which is a shrug. Ten pages is
    ~100 days and ~600 windows, which is still small but is at least an estimate.

    Each page is appended as it arrives, so an interruption costs the last page,
    not the backfill. One page per 0.25 s keeps the weight budget comfortable --
    Binance allows 1,200 request weight per minute and klines cost 1-2.
    """
    if pages < 1:
        raise BarError("pages must be >= 1")

    total = 0
    result: dict[str, Any] = {"pages": 0, "appended": 0}
    end_time_ms: int | None = None
    span = timedelta(seconds=INTERVAL_SECONDS[interval])

    for page in range(pages):
        page_bars = fetch_klines(
            product=product,
            interval=interval,
            limit=limit,
            end_time_ms=end_time_ms,
            settled=False,  # the forming bar is dropped below, so the page size stays honest
            **fetch_kwargs,
        )
        raw_count = len(page_bars)
        now = datetime.now(timezone.utc)
        batch = [bar for bar in page_bars if bar.open_time + span <= now]
        if not batch:
            break
        total += append_bars(path, batch)
        result["pages"] = page + 1
        end_time_ms = int(batch[0].open_time.timestamp() * 1000) - 1
        # A short page means the start of the product's history. Testing the raw
        # count, not the filtered one: the settled drop costs one bar on the
        # first page every time and would stop the backfill after page one.
        if raw_count < limit:
            break
        if pause:
            time.sleep(pause)

    result["appended"] = total
    stored = load_bars(path)
    result["stored"] = len(stored)
    if stored:
        result["from"] = stored[0].open_time.isoformat()
        result["to"] = stored[-1].open_time.isoformat()
    return result


# --------------------------------------------------------------------------- #
# Append-only store
# --------------------------------------------------------------------------- #


def store_path(root: str | Path, product: str = DEFAULT_PRODUCT, interval: str = DEFAULT_INTERVAL) -> Path:
    return Path(root) / f"{product.upper()}_{interval}.csv"


def load_bars(path: str | Path) -> list[Bar]:
    """Read the store: sorted by open time, one bar per open time.

    A backfill appends *older* bars to the end of the file, so the file is not
    sorted on disk. Order is restored here, and duplicates are dropped keeping
    the first occurrence -- the store is append-only at rest, normalised only in
    memory, which is what lets history be extended without rewriting it.
    """
    source = Path(path)
    if not source.exists():
        return []
    seen: dict[datetime, Bar] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bar = Bar(
                open_time=datetime.fromisoformat(row["open_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            seen.setdefault(bar.open_time, bar)
    return [seen[moment] for moment in sorted(seen)]


def append_bars(path: str | Path, bars: Sequence[Bar]) -> int:
    """Append every bar the store does not already hold. Returns the count.

    Never rewrites the file: a crash mid-write costs one line, not the history.
    Overlapping a previous fetch is normal (every poll returns the same last
    1,000 bars), and a backfill deliberately appends bars that are *older* than
    everything on disk -- so dedupe is by ``open_time`` membership, not by
    "newer than the last line".
    """
    destination = Path(path)
    known = {bar.open_time for bar in load_bars(destination)}
    unique: dict[datetime, Bar] = {}
    for bar in bars:
        if bar.open_time not in known:
            unique[bar.open_time] = bar
    fresh = [unique[moment] for moment in sorted(unique)]
    if not fresh:
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    new_file = not destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(COLUMNS)
        for bar in fresh:
            writer.writerow(
                [
                    bar.open_time.isoformat(),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                ]
            )
        handle.flush()
    return len(fresh)


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #


def resample(bars: Sequence[Bar], factor: int, source_interval: str) -> list[Bar]:
    """Aggregate N consecutive bars, keeping only *complete* groups.

    Grouping is by clock time, not by index: a gap in the source would otherwise
    slide every later group out of alignment. An incomplete tail group is
    dropped, because a half-built 4h bar is not a 4h bar.
    """
    if factor < 1:
        raise BarError("factor must be >= 1")
    if factor == 1:
        return list(bars)
    span = timedelta(seconds=INTERVAL_SECONDS[source_interval] * factor)

    groups: dict[datetime, list[Bar]] = {}
    for bar in bars:
        epoch = int(bar.open_time.timestamp())
        bucket = datetime.fromtimestamp(epoch - epoch % int(span.total_seconds()), tz=timezone.utc)
        groups.setdefault(bucket, []).append(bar)

    out: list[Bar] = []
    for bucket in sorted(groups):
        group = sorted(groups[bucket], key=lambda bar: bar.open_time)
        if len(group) != factor:
            continue
        out.append(
            Bar(
                open_time=bucket,
                open=group[0].open,
                high=max(bar.high for bar in group),
                low=min(bar.low for bar in group),
                close=group[-1].close,
                volume=sum(bar.volume for bar in group),
            )
        )
    return out


def ema_series(values: Sequence[float], period: int) -> list[float | None]:
    """An EMA seeded with the first ``period`` values' mean.

    ``None`` until the seed is complete. TradingView seeds differently, so an
    exact match with the feed is not expected -- which is why the cross-check
    compares magnitudes with a tolerance rather than asserting equality.
    """
    if period <= 0:
        raise BarError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        previous = out[index - 1]
        assert previous is not None
        out[index] = alpha * values[index] + (1.0 - alpha) * previous
    return out


def ema_last(values: Sequence[float], period: int) -> float | None:
    series = ema_series(values, period)
    return series[-1] if series else None


def true_ranges(bars: Sequence[Bar]) -> list[float]:
    out: list[float] = []
    for index, bar in enumerate(bars):
        if index == 0:
            out.append(bar.high - bar.low)
            continue
        previous = bars[index - 1].close
        out.append(max(bar.high - bar.low, abs(bar.high - previous), abs(bar.low - previous)))
    return out


def adx(rows: Sequence[Bar], period: int = 14) -> float | None:
    """Wilder's ADX at the last bar. ``None`` before it has warmed up."""
    return adx_series(rows, period)[-1] if rows else None


def atr_series(bars: Sequence[Bar], period: int = 14) -> list[float | None]:
    """Wilder's ATR per bar, ``None`` until it has warmed up.

    A series rather than a scalar because the state at index ``i`` must use only
    bars ``<= i``: computing "the ATR now" and applying it to history is the
    lookahead bias this whole phase exists to avoid.
    """
    if period <= 0:
        raise BarError("period must be positive")
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period + 1:
        return out
    ranges = true_ranges(bars)
    value = sum(ranges[1 : period + 1]) / period
    out[period] = value
    for index in range(period + 1, len(bars)):
        value = (value * (period - 1) + ranges[index]) / period
        out[index] = value
    return out


def atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    """Wilder's ATR. ``None`` when there are too few bars to warm it up."""
    return atr_series(bars, period)[-1] if bars else None


def atr_pct(bars: Sequence[Bar], period: int = 14) -> float | None:
    """ATR as % of the last close: the scale-free version, comparable across prices."""
    value = atr(bars, period)
    if value is None or not bars:
        return None
    return value / bars[-1].close * 100.0


def _directional_movement(bars: Sequence[Bar]) -> tuple[list[float], list[float]]:
    """Wilder's +DM and -DM, one per bar (the first is always zero)."""
    plus: list[float] = [0.0]
    minus: list[float] = [0.0]
    for previous, bar in zip(bars, bars[1:]):
        up = bar.high - previous.high
        down = previous.low - bar.low
        plus.append(up if (up > down and up > 0) else 0.0)
        minus.append(down if (down > up and down > 0) else 0.0)
    return plus, minus


def _dx(smoothed_tr: float, smoothed_plus: float, smoothed_minus: float) -> float:
    if smoothed_tr <= 0:
        return 0.0
    plus_di = 100.0 * smoothed_plus / smoothed_tr
    minus_di = 100.0 * smoothed_minus / smoothed_tr
    total = plus_di + minus_di
    return 0.0 if total == 0 else 100.0 * abs(plus_di - minus_di) / total


def adx_series(bars: Sequence[Bar], period: int = 14) -> list[float | None]:
    """Wilder's ADX per bar, so trend strength is available ex-ante.

    The trade gate needs ADX >= 20, and the feed's ADX cannot be replayed over
    history. Computing it from the same bars means a backtest asks the question
    the live path actually asks.
    """
    if period <= 0:
        raise BarError("period must be positive")
    out: list[float | None] = [None] * len(bars)
    if len(bars) < 2 * period + 1:
        return out

    ranges = true_ranges(bars)
    plus, minus = _directional_movement(bars)

    smoothed_tr = sum(ranges[1 : period + 1])
    smoothed_plus = sum(plus[1 : period + 1])
    smoothed_minus = sum(minus[1 : period + 1])

    running: list[tuple[int, float]] = [(period, _dx(smoothed_tr, smoothed_plus, smoothed_minus))]
    for index in range(period + 1, len(bars)):
        smoothed_tr = smoothed_tr - smoothed_tr / period + ranges[index]
        smoothed_plus = smoothed_plus - smoothed_plus / period + plus[index]
        smoothed_minus = smoothed_minus - smoothed_minus / period + minus[index]
        running.append((index, _dx(smoothed_tr, smoothed_plus, smoothed_minus)))

    window: list[float] = []
    average: float | None = None
    for index, value in running:
        if average is None:
            window.append(value)
            if len(window) == period:
                average = sum(window) / period
                out[index] = average
        else:
            average = (average * (period - 1) + value) / period
            out[index] = average
    return out



def bollinger_width_pct(bars: Sequence[Bar], period: int = 20, mult: float = 2.0) -> float | None:
    if len(bars) < period:
        return None
    closes = [bar.close for bar in bars[-period:]]
    deviation = statistics.pstdev(closes)
    return 4.0 * mult * deviation / closes[-1] * 100.0


def keltner_width_pct(
    bars: Sequence[Bar], *, ema_period: int = 20, atr_period: int = 10, mult: float = 2.0
) -> float | None:
    """(upper - lower) / close x 100, with a Keltner built on EMA and ATR."""
    if len(bars) < max(ema_period, atr_period + 1):
        return None
    middle = ema_last([bar.close for bar in bars], ema_period)
    spread = atr(bars, atr_period)
    if middle is None or spread is None:
        return None
    return 2.0 * mult * spread / bars[-1].close * 100.0


def rolling_keltner_width_pct(
    bars: Sequence[Bar], *, window: int = 200, **kwargs: Any
) -> list[float]:
    """The width series, so the current value can be ranked against its own past."""
    out: list[float] = []
    start = max(0, len(bars) - window)
    for end in range(start, len(bars)):
        value = keltner_width_pct(bars[: end + 1], **kwargs)
        if value is not None:
            out.append(value)
    return out


def percentile_rank(values: Sequence[float], value: float) -> float | None:
    """Fraction of the sample at or below ``value``, as 0..100."""
    if not values:
        return None
    below = sum(1 for item in values if item <= value)
    return below / len(values) * 100.0


def expected_move_pct(bars: Sequence[Bar], horizon_hours: float, *, lookback: int = 200) -> float | None:
    """One-sigma move over ``horizon_hours``, in % of price, from realised vol.

    This is the number the projection band claims to be and has never been
    compared with. Log returns, scaled by sqrt(time): the random-walk floor that
    any band narrower than this is lying about.
    """
    if horizon_hours <= 0:
        raise BarError("horizon_hours must be positive")
    if len(bars) < 3:
        return None
    recent = list(bars)[-lookback:]
    returns = [
        math.log(later.close / earlier.close)
        for earlier, later in zip(recent, recent[1:])
        if earlier.close > 0 and later.close > 0
    ]
    if len(returns) < 2:
        return None
    bar_hours = (recent[1].open_time - recent[0].open_time).total_seconds() / 3600.0
    if bar_hours <= 0:
        return None
    sigma = statistics.stdev(returns) * math.sqrt(horizon_hours / bar_hours)
    return 100.0 * sigma


# --------------------------------------------------------------------------- #
# Cross-check against the indicator feed
# --------------------------------------------------------------------------- #


def cross_check(
    bars: Sequence[Bar],
    emas: Mapping[str, float | None],
    *,
    periods: Mapping[str, int],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare locally computed EMAs with the feed's, and refuse a wrong series.

    The two will never agree exactly: TradingView seeds differently and computes
    on a live, part-formed candle, while these are closed bars. A difference of a
    few tenths of a percent is expected. Five percent is not a seed difference --
    it is a different instrument, a different timeframe, or a broken series, and
    that must fail loudly rather than quietly feed ATR into a decision.
    """
    rows: dict[str, Any] = {}
    worst = 0.0
    for key, period in periods.items():
        theirs = emas.get(key)
        if theirs is None:
            continue
        ours = ema_last([bar.close for bar in bars], period)
        if ours is None:
            rows[key] = {"status": "not enough bars", "bars": len(bars), "period": period}
            continue
        error = abs(ours - theirs) / theirs
        worst = max(worst, error)
        rows[key] = {"ours": round(ours, 2), "theirs": round(float(theirs), 2), "rel_error": round(error, 6)}

    record: dict[str, Any] = {
        "compared": len([r for r in rows.values() if "rel_error" in r]),
        "max_rel_error": round(worst, 6),
        "tolerance": tolerance,
        "emas": rows,
    }
    if worst > tolerance:
        record["status"] = "mismatch"
        raise BarError(
            f"local EMAs disagree with the feed by {worst:.1%} "
            f"(tolerance {tolerance:.0%}); these are not the same series"
        )
    record["status"] = "ok" if record["compared"] else "unchecked"
    return record
