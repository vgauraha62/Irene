"""Market-data access for the trading agent.

Design rules (these are the fixes for the previous version):

* **No network I/O at import time.** Importing this module must be free and
  instant. Data is fetched explicitly via :func:`fetch` / :func:`call_any_analysis`.
* **Every request has a timeout.** ``timeout=None`` used to mean "block forever".
* **Retries with backoff**, because the scraper behind ``tradingview_ta``
  rate-limits (HTTP 429) when hit three times in a row.
* **Explicit timestamps.** A signal you cannot date-stamp is not auditable, and
  stale data silently becomes a losing trade.
* **Validated payloads.** A missing/``None`` EMA used to surface as an opaque
  ``TypeError`` deep inside numpy.

``analysis.py`` is deliberately NOT imported-for-side-effects here: this module
is the data layer and depends on the domain constants that ``analysis`` owns.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from tradingview_ta import TA_Handler
from tradingview_ta import Interval as TVInterval

from analysis import MA_KEYS, TIMEFRAME_SECONDS
from analysis import MissingIndicatorError  # re-exported: one class, one catch

log = logging.getLogger(__name__)

DEFAULT_SYMBOL = "ETHUSD"
DEFAULT_EXCHANGE = "COINBASE"
DEFAULT_SCREENER = "crypto"


#: timeframe -> the library's own interval attribute name, plus the expected
#: value. The names are verified, not guessed: tradingview_ta 3.3.0 spells them
#: ``INTERVAL_15_MINUTES`` / ``INTERVAL_4_HOURS`` (there is no ``INTERVAL_15m``).
_INTERVAL_ATTRS: dict[str, tuple[str, str]] = {
    "15m": ("INTERVAL_15_MINUTES", "15m"),
    "30m": ("INTERVAL_30_MINUTES", "30m"),
    "4h": ("INTERVAL_4_HOURS", "4h"),
}


def _tv_interval(attribute: str, expected: str) -> str:
    """Read the library constant, warning loudly on any drift."""
    value = getattr(TVInterval, attribute, None)
    if value is None:
        log.warning(
            "tradingview_ta.Interval.%s is missing; falling back to %r", attribute, expected
        )
        return expected
    if value != expected:
        log.warning(
            "tradingview_ta.Interval.%s is %r, expected %r; using the library value",
            attribute,
            value,
            expected,
        )
    return value


#: timeframe -> the interval token handed to TA_Handler.
#: ``TIMEFRAME_SECONDS`` lives in ``analysis`` because the slope maths needs it.
TIMEFRAMES: dict[str, str] = {
    tf: _tv_interval(attr, expected) for tf, (attr, expected) in _INTERVAL_ATTRS.items()
}

if set(TIMEFRAMES) != set(TIMEFRAME_SECONDS):  # pragma: no cover - config guard
    raise RuntimeError(
        f"timeframe mismatch: indicators has {sorted(TIMEFRAMES)}, "
        f"analysis has {sorted(TIMEFRAME_SECONDS)}"
    )


class DataFetchError(RuntimeError):
    """Raised when the upstream market data could not be retrieved."""


@dataclass(frozen=True)
class Snapshot:
    """One timeframe's indicators, plus where its timestamp came from.

    ``as_of_source`` matters: if the upstream payload carries no timestamp we
    fall back to the local wall clock, which means we genuinely do not know how
    fresh the data is. Callers should not pretend otherwise.
    """

    timeframe: str
    indicators: Mapping[str, Any]
    as_of: datetime
    as_of_source: str = "exchange"
    fetched_at: datetime | None = None

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.as_of).total_seconds()

    @property
    def close(self) -> float:
        return float(self.indicators["close"])


def _resolve_as_of(analysis: Any, fetched_at: datetime) -> tuple[datetime, str]:
    """Best-effort exchange timestamp, with an honest provenance label.

    Verified against tradingview_ta 3.3.0: ``Analysis.time`` is set to
    ``datetime.datetime.now()`` inside the scraper. It is a *local scrape clock*,
    not exchange time, so a naive datetime is labelled ``fetch-clock`` and the
    caller is warned. Only a tz-aware timestamp or an epoch value is trusted as
    exchange time -- and only a real market-data feed would supply one.
    """
    raw = getattr(analysis, "time", None)
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return fetched_at, "fetch-clock"
        return raw.astimezone(timezone.utc), "exchange"
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # Guard against milliseconds-since-epoch as well as seconds.
        seconds = raw / 1000.0 if raw > 1e11 else float(raw)
        return datetime.fromtimestamp(seconds, tz=timezone.utc), "exchange"
    return fetched_at, "fetch-clock"


def _validate(indicators: Mapping[str, Any], timeframe: str) -> None:
    if not indicators:
        raise MissingIndicatorError(f"{timeframe}: empty indicator payload")
    missing = [key for key in MA_KEYS if indicators.get(key) is None]
    if missing:
        raise MissingIndicatorError(
            f"{timeframe}: missing or null indicators: {', '.join(missing)}"
        )
    for key in MA_KEYS:
        value = indicators[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MissingIndicatorError(
                f"{timeframe}: indicator {key} is not numeric ({value!r})"
            )
    if indicators.get("close") is None:
        raise MissingIndicatorError(f"{timeframe}: no 'close' price")
    if float(indicators["close"]) <= 0:
        raise MissingIndicatorError(
            f"{timeframe}: non-positive close price {indicators['close']!r}"
        )


def fetch(
    timeframe: str,
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    screener: str = DEFAULT_SCREENER,
    timeout: float = 10.0,
    retries: int = 2,
    backoff: float = 1.5,
) -> Snapshot:
    """Fetch one timeframe, with a hard timeout and bounded retries."""
    if timeframe not in TIMEFRAMES:
        raise KeyError(
            f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
        )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        fetched_at = datetime.now(timezone.utc)
        try:
            handler = TA_Handler(
                symbol=symbol,
                interval=TIMEFRAMES[timeframe],
                exchange=exchange,
                screener=screener,
                timeout=timeout,
            )
            analysis = handler.get_analysis()
            indicators = getattr(analysis, "indicators", None)
            _validate(indicators, timeframe)
            as_of, source = _resolve_as_of(analysis, fetched_at)
            if source == "fetch-clock":
                log.warning(
                    "%s: upstream supplies no exchange timestamp (only a local "
                    "scrape clock); freshness cannot be verified",
                    timeframe,
                )
            return Snapshot(
                timeframe=timeframe,
                indicators=dict(indicators),
                as_of=as_of,
                as_of_source=source,
                fetched_at=fetched_at,
            )
        except MissingIndicatorError:
            # A schema problem will not fix itself on retry.
            raise
        except Exception as exc:  # network, 429, parse errors
            last_error = exc
            if attempt < retries:
                delay = backoff * (2**attempt)
                log.warning(
                    "%s: fetch failed (%s), retrying in %.1fs (%d/%d)",
                    timeframe,
                    exc,
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)

    raise DataFetchError(
        f"{symbol} {timeframe}: could not fetch data after {retries + 1} attempts: "
        f"{last_error}"
    ) from last_error


# Module-level TTL cache so analysis + plotting share one round trip.
# Guarded by a lock because a web dashboard serves callbacks on threads: without
# this, two concurrent refreshes each perform their own three scrapes.
_CACHE: dict[tuple, tuple[float, Snapshot]] = {}
_CACHE_LOCK = threading.Lock()


def call_any_analysis(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    screener: str = DEFAULT_SCREENER,
    timeframes: tuple[str, ...] = tuple(TIMEFRAMES),
    ttl: float = 60.0,
    **fetch_kwargs: Any,
) -> dict[str, Snapshot]:
    """Fetch the requested timeframes and return ``{timeframe: Snapshot}``.

    Keeps the original public name so callers read the same, but now returns
    timestamped snapshots and never executes at import time.
    """
    out: dict[str, Snapshot] = {}
    for timeframe in timeframes:
        key = (symbol, exchange, screener, timeframe)
        with _CACHE_LOCK:
            now = time.monotonic()
            cached = _CACHE.get(key)
            if cached and ttl > 0 and (now - cached[0]) < ttl:
                out[timeframe] = cached[1]
                continue

        # Fetch outside the lock so one slow symbol cannot block the others.
        snapshot = fetch(
            timeframe, symbol=symbol, exchange=exchange, screener=screener, **fetch_kwargs
        )
        with _CACHE_LOCK:
            _CACHE[key] = (time.monotonic(), snapshot)
        out[timeframe] = snapshot
    return out


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------- #
# Offline history loader (the bundled Ethereum CSV)
# --------------------------------------------------------------------------- #

_VOLUME_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9}


def _parse_volume(value: Any) -> float:
    """``"193.65K"`` -> ``193650.0``."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "n/a", "N/A"}:
        return float("nan")
    multiplier = 1.0
    if text[-1].upper() in _VOLUME_SUFFIXES:
        multiplier = _VOLUME_SUFFIXES[text[-1].upper()]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def load_csv_history(path: str | Any):
    """Load the bundled daily OHLCV CSV into a clean, ascending DataFrame.

    Handles the real-world mess in ``ethereum-data/Ethereum Historical Data.csv``:
    a UTF-8 BOM, quoted thousands separators (``"2,314.98"``), percent strings
    (``"-0.41%"``), ``K``-suffixed volume and ``DD-MM-YYYY`` dates.

    Note: the bundled file has only ~30 daily rows, which cannot warm up a
    200-period EMA. Callers should treat long-horizon indicators as indicative.
    """
    import pandas as pd  # local import: keeps ``import indicators`` light

    df = pd.read_csv(path, encoding="utf-8-sig", thousands=",", na_values=["", "-"])

    rename = {
        "date": "date",
        "price": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "vol": "volume",
        "vol.": "volume",
        "change_%": "change_pct",
        "change%": "change_pct",
    }
    df.columns = [
        rename.get(c.strip().lower().replace(" ", "_"), c.strip().lower())
        for c in df.columns
    ]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path}: expected date and price columns, got {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"{path}: unparseable dates (expected DD-MM-YYYY)")

    if "change_pct" in df.columns:
        df["change_pct"] = pd.to_numeric(
            df["change_pct"].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    if "volume" in df.columns:
        df["volume"] = df["volume"].map(_parse_volume)

    for column in ("open", "high", "low", "close"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["close"].isna().any():
        raise ValueError(f"{path}: unparseable close prices")

    df = df.sort_values("date").reset_index(drop=True)

    # Be explicit rather than quietly producing an unconverged EMA200.
    from analysis import MIN_BARS_PER_PERIOD, MA_PERIODS, warmup_sufficient

    longest = MA_PERIODS[-1]
    if not warmup_sufficient(len(df), longest):
        log.warning(
            "%s: %d rows is short of the ~%d bars an EMA%d needs; long-horizon "
            "indicators from this file are indicative only",
            path,
            len(df),
            MIN_BARS_PER_PERIOD * longest,
            longest,
        )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for tf, snap in call_any_analysis().items():
        print(
            f"{tf:>3}  close={snap.close:,.2f}  "
            f"as_of={snap.as_of.isoformat()}  ({snap.as_of_source})"
        )
