"""Multi-scale term-structure analysis and credit-spread planning.

Everything here is a pure function over in-memory data: no network, no file
I/O, no import-time side effects. That is what makes it testable, and it is the
main structural fix over the previous version, which did all its work at module
scope against names that did not exist.

The two ideas inherited from the original file are kept deliberately:

* the EMA ladder (10/20/30/50/100/200) is a *term structure*, and its shape is
  meaningful;
* micro (15m) vs macro (4h) disagreement is the trade-relevant event.

The four ideas fixed are:

1. **Units.** ``np.gradient`` used to run with default spacing of 1 *bar*, so a
   4h slope was ~16x smaller than a 15m slope for the same price move. Every
   cross-timeframe comparison was therefore biased bullish. Slopes are now
   %/hour, computed against the EMA horizon in hours.
2. **Sign.** That slope is a derivative against *lookback*, so it is negative
   while price rises. Feeding it straight into ``regime`` inverted every call:
   a clean +0.30 %/hour uptrend was read as "BEARISH REGIME (GRAVITY)". The
   ladder slope is now negated into a clock-side drift before it is read as a
   direction (``drift_pct_per_hour``).
3. **Contract.** ``trade_scenario`` used to build colours and labels and return
   a bare float, so nothing downstream could use them.
4. **Risk.** Credit spreads have a defined max loss. It is now computed, and a
   spread that does not pay enough for its width is refused outright.

``project_price`` is the answer to "where is price most expected to be": a
conditional expectation with an explicit band, labelled a diagnostic until it
has been scored against realised prices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import NormalDist
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

log = logging.getLogger("analysis")

# --------------------------------------------------------------------------- #
# Domain constants (single source of truth; ``indicators`` imports these)
# --------------------------------------------------------------------------- #

MA_KEYS: tuple[str, ...] = ("EMA10", "EMA20", "EMA30", "EMA50", "EMA100", "EMA200")
MA_PERIODS: tuple[int, ...] = (10, 20, 30, 50, 100, 200)

#: Bar length in seconds. Required to make slopes comparable across timeframes.
TIMEFRAME_SECONDS: dict[str, int] = {"15m": 900, "30m": 1_800, "4h": 14_400}

#: Slopes below this magnitude (in %/hour) count as flat, not as a trend.
DEFAULT_EPSILON = 0.01

#: Below this ADX the market is chop and credit spreads get whipsawed.
DEFAULT_ADX_MIN = 20.0

#: An EMA needs roughly 5x its period in bars before it stops being dominated by
#: the seed value. A 200-period EMA on a 30-row file is not a 200-period EMA.
MIN_BARS_PER_PERIOD = 5

#: A credit spread paying less than this fraction of its width is not worth it.
#: Measured at 0.30 delta / 30 DTE: ~32% of width for a bull put and ~24% for a
#: bear call (put skew makes puts richer). A 20% floor therefore sits just under
#: the standard structures while still rejecting junk -- a 0.02-delta strike pays
#: ~1.7% and is refused.
DEFAULT_MIN_CREDIT_RATIO = 0.20

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"

MACRO_BULL = "BULLISH REGIME (EXPANSION)"
MACRO_BEAR = "BEARISH REGIME (GRAVITY)"
MACRO_FLAT = "NEUTRAL REGIME (BALANCE)"

MICRO_BULL = "BULLISH SHIFT (Consolidation/Recovery)"
MICRO_BEAR = "BEARISH CONTINUATION"
MICRO_FLAT = "FLAT (No micro divergence)"


class MissingIndicatorError(ValueError):
    """A required indicator is absent or null."""


class StaleDataError(RuntimeError):
    """Data is too old to act on."""


def warmup_sufficient(n_bars: int, period: int) -> bool:
    """Whether ``n_bars`` is enough for an EMA of ``period`` to have converged."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period!r}")
    if n_bars < 0:
        raise ValueError(f"n_bars must be non-negative, got {n_bars!r}")
    return n_bars >= MIN_BARS_PER_PERIOD * period


# --------------------------------------------------------------------------- #
# Accessors — accept Snapshot, tradingview Analysis, or a plain dict
# --------------------------------------------------------------------------- #


def indicators_of(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        # A bare indicator dict, or {"indicators": {...}} from a Snapshot dump.
        return obj["indicators"] if "indicators" in obj and isinstance(obj["indicators"], Mapping) else obj
    ind = getattr(obj, "indicators", None)
    if isinstance(ind, Mapping):
        return ind
    raise TypeError(f"cannot read indicators from {type(obj).__name__}")


def timeframe_of(obj: Any) -> str:
    tf = getattr(obj, "timeframe", None)
    if tf is None and isinstance(obj, Mapping):
        tf = obj.get("timeframe")
    if tf is None:
        raise ValueError("object does not declare a timeframe")
    return str(tf)


def spot(obj: Any) -> float:
    price = indicators_of(obj).get("close")
    if price is None:
        raise MissingIndicatorError("no 'close' price in payload")
    return float(price)


def ema_values(obj: Any) -> np.ndarray:
    """The EMA ladder as floats, ascending by period."""
    ind = indicators_of(obj)
    missing = [k for k in MA_KEYS if ind.get(k) is None]
    if missing:
        raise MissingIndicatorError(f"missing indicators: {', '.join(missing)}")
    return np.asarray([float(ind[k]) for k in MA_KEYS], dtype=float)


# --------------------------------------------------------------------------- #
# Term structure, slopes, velocity
# --------------------------------------------------------------------------- #


def term_structure(obj: Any) -> dict[str, Any]:
    """EMA spreads, expressed as % of spot so they are scale-free."""
    values = ema_values(obj)
    price = spot(obj)
    spreads = {
        f"{a}-{b}": float((values[i] - values[j]) / price * 100.0)
        for i, a in enumerate(MA_PERIODS)
        for j, b in enumerate(MA_PERIODS)
        if i < j
    }
    return {
        "spot": price,
        "spectral_spread_pct": float((values[0] - values[-1]) / price * 100.0),
        "curvature_pct": float(values[0] - 2 * values[len(values) // 2] + values[-1]) / price * 100.0,
        "spreads_pct": spreads,
    }


def ema_horizons_hours(timeframe: str) -> np.ndarray:
    """EMA period -> elapsed time in hours for that timeframe.

    This is the fix for the cross-timeframe unit bug: EMA10 on 15m is 2.5h of
    history, EMA10 on 4h is 40h. Differentiating against *these* x-values puts
    every timeframe in the same unit, %/hour.
    """
    try:
        seconds = TIMEFRAME_SECONDS[timeframe]
    except KeyError:
        raise KeyError(
            f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAME_SECONDS)}"
        ) from None
    return np.asarray(MA_PERIODS, dtype=float) * seconds / 3600.0


def slopes(obj: Any) -> np.ndarray:
    """d(% of spot)/d(hour of EMA horizon): the term-structure slope, %/hour."""
    values = ema_values(obj)
    price = spot(obj)
    if price <= 0:
        raise ValueError(f"non-positive spot price {price!r}")
    x_hours = ema_horizons_hours(timeframe_of(obj))
    return np.gradient(values / price * 100.0, x_hours)


def slope_summary(obj: Any) -> dict[str, float]:
    grad = slopes(obj)
    return {
        "median": float(np.median(grad)),
        "front": float(grad[0]),
        "back": float(grad[-1]),
        "dispersion": float(np.std(grad)),
    }


def rung_slopes(obj: Any) -> dict[str, float]:
    """The ladder slope at every rung, so the *shape* stays visible.

    A single median hides whether a trend is front-loaded (only the fast EMAs
    are sloping) or back-loaded (the whole ladder is). Same %/hour units as
    ``slopes``, one value per EMA period.
    """
    return {key: float(value) for key, value in zip(MA_KEYS, slopes(obj))}


def drift_pct_per_hour(obj: Any) -> float:
    """Trend rate in %/hour **of clock time** -- the sign ``regime`` needs.

    ``slopes`` differentiates %-of-spot against *lookback* in hours, and the two
    axes run opposite ways: in a rise the long-lookback EMAs sit below spot, so
    the ladder slope is negative while the market goes up. Taking a ladder slope
    as a trend direction therefore inverts every call -- a calm rise used to be
    read as a bearish gravity regime.

    Negating it recovers the trend rate and matches the sign convention of
    ``velocity_and_acceleration``. It is an average over the ladder's window
    (~2.5h..50h on 15m), not an instantaneous rate.
    """
    return -float(np.median(slopes(obj)))


def drifts(snapshots: Mapping[str, Any]) -> dict[str, float]:
    """``drift_pct_per_hour`` for every timeframe, in %/hour, mutually comparable."""
    return {tf: drift_pct_per_hour(snap) for tf, snap in snapshots.items()}


def velocity_and_acceleration(
    prices: Sequence[float], bar_seconds: float
) -> tuple[float, float]:
    """First and second time-derivatives of one price series.

    Returned as (%/hour, %/hour^2) using the most recent bar. This is the real
    velocity/acceleration the old code claimed to compute: ``np.gradient`` over
    the EMA ladder is a derivative across *periods*, not across *time*.
    """
    series = np.asarray(prices, dtype=float)
    if series.size < 3:
        raise ValueError(f"need at least 3 observations, got {series.size}")
    if np.any(series <= 0):
        raise ValueError("prices must be positive")
    if bar_seconds <= 0:
        raise ValueError(f"bar_seconds must be positive, got {bar_seconds!r}")

    hours = bar_seconds / 3600.0
    relative = series / series[0] - 1.0
    velocity = np.gradient(relative, hours) * 100.0
    acceleration = np.gradient(velocity, hours)
    return float(velocity[-1]), float(acceleration[-1])


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Regime:
    """The market read: macro direction, micro divergence, and how much to trust it."""

    macro_dir: str
    micro_shift: str
    direction: str
    macro_strength: float
    confidence: float
    agreement: int
    as_of: str
    reasons: list[str] = field(default_factory=list)
    drift: dict[str, float] = field(default_factory=dict)
    adx: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_dir": self.macro_dir,
            "micro_shift": self.micro_shift,
            "direction": self.direction,
            "macro_strength": round(self.macro_strength, 4),
            "confidence": round(self.confidence, 4),
            "agreement": self.agreement,
            "adx": None if self.adx is None else round(self.adx, 2),
            "as_of": self.as_of,
            "drift_pct_per_hour": {k: round(v, 5) for k, v in self.drift.items()},
            "reasons": list(self.reasons),
        }


def _sign(value: float, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _as_of(snapshots: Mapping[str, Any]) -> str:
    stamps = [getattr(s, "as_of", None) for s in snapshots.values()]
    stamps = [s for s in stamps if isinstance(s, datetime)]
    if not stamps:
        return datetime.now(timezone.utc).isoformat()
    return min(stamps).astimezone(timezone.utc).isoformat()


def regime(
    snapshots: Mapping[str, Any],
    *,
    epsilon: float = DEFAULT_EPSILON,
    adx_min: float = DEFAULT_ADX_MIN,
    use_chop_filter: bool = True,
) -> Regime:
    """Classify the regime from micro + macro term structures.

    Differs from the original in four ways that change the answer:

    * all slopes are in %/hour, so they may be compared;
    * the ladder slope is negated into a clock-side **drift** before it is read
      as a direction (see ``drift_pct_per_hour``); reading the raw ladder slope
      inverted every call;
    * a trend needs the timeframes to **agree**, not just one leg to exceed
      another (the old ``or`` fired the bullish label almost always);
    * chop and weak edges resolve to NEUTRAL, so the agent can decline to trade.
    """
    missing = [tf for tf in ("15m", "30m", "4h") if tf not in snapshots]
    if missing:
        raise KeyError(f"regime needs timeframes {missing}")

    medians = drifts(snapshots)
    reasons: list[str] = []

    macro_median = medians["4h"]
    macro_sign = _sign(macro_median, epsilon)
    if macro_sign > 0:
        macro_dir = MACRO_BULL
    elif macro_sign < 0:
        macro_dir = MACRO_BEAR
    else:
        macro_dir = MACRO_FLAT
        reasons.append(
            f"4h median drift {macro_median:+.4f} %/h is within the flat band (+-{epsilon})"
        )

    signs = [_sign(medians[tf], epsilon) for tf in ("15m", "30m", "4h")]
    agreeing = sum(1 for s in signs if s == macro_sign) if macro_sign != 0 else 0
    agreement = agreeing

    macro_strength = min(1.0, abs(macro_median) / (3.0 * epsilon)) if epsilon > 0 else 0.0
    if macro_sign == 0:
        agreement_note = "the 4h leg is flat, so there is no macro direction to agree with"
    else:
        agreement_note = f"{agreeing}/3 agree with the macro leg"
    reasons.append(
        f"drift 15m {medians['15m']:+.4f}, 30m {medians['30m']:+.4f}, "
        f"4h {medians['4h']:+.4f} %/h; {agreement_note}"
    )

    divergence = medians["15m"] - medians["4h"]
    if divergence > epsilon:
        micro_shift = MICRO_BULL
    elif divergence < -epsilon:
        micro_shift = MICRO_BEAR
    else:
        micro_shift = MICRO_FLAT

    adx_value: float | None = None
    for tf in ("4h", "30m", "15m"):
        raw = indicators_of(snapshots[tf]).get("ADX")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            adx_value = float(raw)
            break

    confidence = macro_strength * (agreement / 3.0)

    direction = NEUTRAL
    if macro_sign > 0 and agreement == 3 and medians["15m"] > 0:
        direction = BULLISH
    elif macro_sign < 0 and agreement == 3 and medians["15m"] < 0:
        direction = BEARISH

    if macro_sign != 0 and agreement < 3:
        reasons.append("timeframes disagree; no confirmed trend, standing down")

    if use_chop_filter and adx_value is not None:
        if adx_value < adx_min and direction != NEUTRAL:
            reasons.append(f"ADX {adx_value:.1f} < {adx_min}: chop, refusing directional trade")
            direction = NEUTRAL
            confidence *= 0.5
        elif adx_value < adx_min:
            reasons.append(f"ADX {adx_value:.1f} < {adx_min}: chop")
    elif adx_value is None:
        reasons.append("ADX unavailable; chop filter skipped")

    return Regime(
        macro_dir=macro_dir,
        micro_shift=micro_shift,
        direction=direction,
        macro_strength=macro_strength,
        confidence=confidence,
        agreement=agreement,
        as_of=_as_of(snapshots),
        reasons=reasons,
        drift=medians,
        adx=adx_value,
    )


def guard(
    snapshots: Mapping[str, Any],
    *,
    max_age_seconds: float = 3600.0,
    now: datetime | None = None,
) -> None:
    """Refuse to emit a signal on stale data.

    If a snapshot's timestamp came from the local clock (``as_of_source ==
    'fetch-clock'``) freshness is genuinely unverifiable; that provenance
    warning belongs to the fetch (``indicators.fetch``), so it is not repeated
    here. This guard's single job is to refuse stale age.
    """
    log = logging.getLogger(__name__)
    now = now or datetime.now(timezone.utc)

    for tf, snap in snapshots.items():
        age_fn = getattr(snap, "age_seconds", None)
        if not callable(age_fn):
            continue
        age = age_fn(now)
        if age > max_age_seconds:
            raise StaleDataError(
                f"{tf}: data is {age:.0f}s old (limit {max_age_seconds:.0f}s) - "
                "refusing to emit a signal on stale prices"
            )
        if getattr(snap, "as_of_source", "exchange") != "exchange":
            log.debug("%s: freshness unverified", tf)


# --------------------------------------------------------------------------- #
# Expected price: a conditional expectation with a band, never a point target
# --------------------------------------------------------------------------- #

#: Which ladder to read a projection from, by horizon in hours. The ladder is a
#: lagged price path, so a short horizon is read from a short ladder.
HORIZON_LADDERS: tuple[tuple[float, str], ...] = ((1.0, "15m"), (3.0, "30m"))

#: The horizons the agent publishes by default: the band described in AGENTS.md.
PROJECTION_HORIZONS: tuple[float, ...] = (0.5, 4.0)

#: Irreducible half-width of the band, in % of spot. Even a perfect drift
#: extrapolation cannot say where inside the bar price will land.
BAND_FLOOR_PCT = 0.15


@dataclass(frozen=True)
class Projection:
    """``E[log P(t+H)]`` from the current drift, with its honest uncertainty."""

    horizon_hours: float
    horizon_label: str
    timeframe: str
    spot: float
    drift_pct_per_hour: float
    acceleration_pct_per_hour2: float | None
    expected_price: float
    low: float
    high: float
    band_pct: float
    direction: str
    invalidates: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon_label,
            "horizon_hours": self.horizon_hours,
            "basis_timeframe": self.timeframe,
            "spot": round(self.spot, 2),
            "drift_pct_per_hour": round(self.drift_pct_per_hour, 5),
            "acceleration_pct_per_hour2": (
                None
                if self.acceleration_pct_per_hour2 is None
                else round(self.acceleration_pct_per_hour2, 5)
            ),
            "expected_price": round(self.expected_price, 2),
            "low": round(self.low, 2),
            "high": round(self.high, 2),
            "band_pct": round(self.band_pct, 3),
            "direction": self.direction,
            "invalidates": self.invalidates,
            "notes": list(self.notes),
        }


def _ladder_for(horizon_hours: float) -> str:
    for limit, timeframe in HORIZON_LADDERS:
        if horizon_hours <= limit:
            return timeframe
    return "4h"


def _adx_of(snapshots: Mapping[str, Any]) -> float | None:
    for tf in ("4h", "30m", "15m"):
        raw = indicators_of(snapshots[tf]).get("ADX")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return None


def horizon_label(horizon_hours: float) -> str:
    if horizon_hours < 1.0:
        return f"{horizon_hours * 60:.0f}m"
    return f"{horizon_hours:g}h"


def project_price(
    snapshots: Mapping[str, Any],
    *,
    horizon_hours: float = 4.0,
    acceleration_pct_per_hour2: float | None = None,
    adx_min: float = DEFAULT_ADX_MIN,
) -> Projection:
    """The expected price at ``horizon_hours``, with the band that contains it.

    First-order local extrapolation of the current drift, plus a second-order
    term when an acceleration is supplied:

        E[log P(t+H)] ~= log spot + drift * H + 0.5 * accel * H^2

    Acceleration needs a *time series* of one series, so it cannot be derived
    from a single snapshot; pass ``acceleration_pct_per_hour2`` from successive
    runs (``velocity_and_acceleration``) and it is used. Left out, the
    projection is first-order and says so.

    The band is not decoration: it is the only thing that makes the number
    falsifiable. It widens with cross-timeframe drift dispersion, in chop, and
    with the floor below. Per AGENTS.md invariant 7 this is a **diagnostic**,
    not a validated forecast -- it is labelled as such on every call.
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours!r}")
    missing = [tf for tf in ("15m", "30m", "4h") if tf not in snapshots]
    if missing:
        raise KeyError(f"projection needs timeframes {missing}")

    timeframe = _ladder_for(horizon_hours)
    price = spot(snapshots[timeframe])
    if price <= 0:
        raise ValueError(f"non-positive spot price {price!r}")

    drift = drift_pct_per_hour(snapshots[timeframe])
    per_timeframe = drifts(snapshots)
    dispersion = max(abs(value - drift) for value in per_timeframe.values())
    adx = _adx_of(snapshots)

    notes: list[str] = []
    horizon_pct = abs(drift) * horizon_hours
    band_pct = BAND_FLOOR_PCT + 0.5 * horizon_pct + dispersion * horizon_hours

    if adx is None:
        band_pct *= 1.5
        notes.append("ADX unavailable; band widened 50%")
    elif adx < adx_min:
        band_pct *= 1.0 + (adx_min - adx) / adx_min
        notes.append(f"ADX {adx:.1f} < {adx_min:g}: chop widens the band")

    accel = acceleration_pct_per_hour2
    if accel is None:
        notes.append(
            "acceleration needs successive runs (no time series in one snapshot); "
            "projection is first-order"
        )
        growth = drift / 100.0 * horizon_hours
    else:
        growth = (
            drift / 100.0 * horizon_hours
            + 0.5 * accel / 100.0 * horizon_hours**2
        )

    expected = price * math.exp(growth)
    half_width = expected * band_pct / 100.0
    low = max(expected - half_width, 1e-9)
    high = expected + half_width

    move_pct = (expected / price - 1.0) * 100.0
    if abs(move_pct) <= band_pct:
        direction = NEUTRAL
        invalidates = (
            f"the {horizon_label(horizon_hours)} band straddles spot: "
            "no direction is established"
        )
    else:
        direction = BULLISH if move_pct > 0 else BEARISH
        invalidates = (
            f"the {timeframe} drift crossing to the opposite sign, or ADX < "
            f"{adx_min:g}, before {horizon_label(horizon_hours)}"
        )

    notes.append(
        f"band = {BAND_FLOOR_PCT:g}% floor + half the drift move + "
        f"{dispersion:.4f} %/h cross-timeframe dispersion"
    )
    notes.append("diagnostic only: never scored against realised prices yet")

    return Projection(
        horizon_hours=horizon_hours,
        horizon_label=horizon_label(horizon_hours),
        timeframe=timeframe,
        spot=price,
        drift_pct_per_hour=drift,
        acceleration_pct_per_hour2=accel,
        expected_price=expected,
        low=low,
        high=high,
        band_pct=band_pct,
        direction=direction,
        invalidates=invalidates,
        notes=notes,
    )


def project_all(
    snapshots: Mapping[str, Any],
    *,
    horizons: Sequence[float] = PROJECTION_HORIZONS,
    **kwargs: Any,
) -> list[Projection]:
    """Project every published horizon. Cheap: pure arithmetic on one snapshot set."""
    return [project_price(snapshots, horizon_hours=h, **kwargs) for h in horizons]


# --------------------------------------------------------------------------- #
# Options: a seam, a stub, and Black-Scholes without extra dependencies
# --------------------------------------------------------------------------- #

_NORMAL = NormalDist()


class OptionsChain(Protocol):
    """The one thing the planner needs from an options data source.

    ``kind`` selects *which instrument* to read, not which skew: a chain quotes
    puts and calls as separate instruments, and either side can be the one with
    a live two-sided quote. Deribit marks both legs of a strike from a single
    surface, so on its ``mark_iv`` the two agree -- the skew that makes a bull
    put and a bear call pay different credits lives across *strikes*, not across
    sides. A traded bid/ask IV does differ slightly by side, so the parameter
    stays in the protocol rather than being assumed away.
    """

    def iv(self, kind: str, dte_days: int, strike: float, spot: float) -> float: ...


@dataclass(frozen=True)
class SyntheticChain:
    """Placeholder chain with a crude moneyness skew.

    Exists so the planner's maths is real and testable today. Replace with a
    broker/feed implementation of :class:`OptionsChain`; nothing else changes.

    The skew is kind-agnostic on purpose: a Black-Scholes surface has one vol
    per strike, and the downward slope already makes out-of-the-money puts
    richer than equidistant calls. Only a live chain differentiates the two.
    """

    base_iv: float = 0.60
    skew: float = 0.15

    def iv(self, kind: str, dte_days: int, strike: float, spot: float) -> float:
        if spot <= 0:
            raise ValueError("spot must be positive")
        moneyness = math.log(strike / spot)
        iv = self.base_iv - self.skew * moneyness
        return max(iv, 0.05)

    def book(self, kind: str, dte_days: int, spot: float | None = None) -> Any:
        """A declared synthetic ladder, so the executable path is testable.

        The prices are Black-Scholes at this chain's own skew with a flat 1%
        touch either side. Nothing here is a market: ``source="synthetic"``
        travels with the book and the payload, because a synthetic credit and a
        market credit must never look alike.
        """
        if spot is None:
            raise ValueError("SyntheticChain.book needs a spot")
        if spot <= 0:
            raise ValueError("spot must be positive")
        if dte_days <= 0:
            raise ValueError("dte_days must be positive")

        # Local import: ``analysis`` is pure maths and must not need the chain
        # module at import time. Sharing the real ``Book`` type is still worth
        # it -- one shape means the planner cannot silently take two paths.
        import iv as iv_mod

        years = dte_days / 365.0
        step = max(round(spot * 0.02, -1), 10.0)
        moment = datetime.now(timezone.utc)
        expiry = moment + timedelta(days=dte_days)
        legs: list[Any] = []
        strike = max(round(spot * 0.4 / step) * step, step)
        ceiling = spot * 1.6
        while strike <= ceiling:
            sigma = self.iv(kind, dte_days, strike, spot)
            # ``Book`` prices are in the underlying, as Deribit quotes them;
            # this stub is the same shape so the planner cannot take two paths.
            mark = bs_price(kind, spot, strike, years, sigma) / spot
            legs.append(
                iv_mod.Quote(
                    kind=kind,
                    strike=float(strike),
                    expiry=expiry,
                    iv=sigma,
                    open_interest=0.0,
                    bid=mark * 0.99,
                    ask=mark * 1.01,
                    mark=mark,
                )
            )
            strike += step
        return iv_mod.Book(
            kind=kind,
            expiry=expiry,
            spot=spot,
            legs=tuple(legs),
            observed_at=moment,
            source="synthetic",
        )


def _d1(spot_price: float, strike: float, years: float, sigma: float, rate: float) -> float:
    if years <= 0 or sigma <= 0:
        raise ValueError("time and volatility must be positive")
    return (math.log(spot_price / strike) + (rate + 0.5 * sigma**2) * years) / (
        sigma * math.sqrt(years)
    )


def bs_price(
    kind: str, spot_price: float, strike: float, years: float, sigma: float, rate: float = 0.0
) -> float:
    """Black-Scholes price of a European option (no dividends)."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if spot_price <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    d1 = _d1(spot_price, strike, years, sigma, rate)
    d2 = d1 - sigma * math.sqrt(years)
    if kind == "call":
        return spot_price * _NORMAL.cdf(d1) - strike * math.exp(-rate * years) * _NORMAL.cdf(d2)
    return strike * math.exp(-rate * years) * _NORMAL.cdf(-d2) - spot_price * _NORMAL.cdf(-d1)


def probability_below(spot_price: float, strike: float, years: float, sigma: float) -> float:
    """P(S_T < strike) under a flat lognormal. This is N(-d2), not the delta.

    Delta is N(-d1); for an out-of-the-money put N(-d2) is *higher*, so quoting
    ``1 - delta`` as a probability of profit overstates it. The gap is ~5 points
    at 0.30 delta and 30 days, which is the same order as the credit ratio the
    planner gates on -- so the approximation can flip a decision.
    """
    if years <= 0 or sigma <= 0:
        raise ValueError("time and volatility must be positive")
    if spot_price <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    d2 = (math.log(spot_price / strike) - 0.5 * sigma * sigma * years) / (sigma * math.sqrt(years))
    return _NORMAL.cdf(-d2)


def zone_probabilities(
    kind: str,
    spot_price: float,
    short_strike: float,
    long_strike: float,
    years: float,
    sigma: float,
) -> tuple[float, float, float]:
    """``(keep all, partial loss, max loss)`` for one credit spread.

    The payoff between the strikes is continuous, so a spread has three
    outcomes, not two: the credit is kept in full only outside the short
    strike, is partly kept between the strikes, and is fully lost beyond the
    hedge. Reporting only ``1 - delta`` collapses the middle band into the
    win, which is where a 2%-wide spread spends a few percent of its outcomes.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    near = probability_below(spot_price, short_strike, years, sigma)
    far = probability_below(spot_price, long_strike, years, sigma)
    if kind == "put":
        # short strike is the higher one: below it the spread starts to lose
        p_max_loss = far
        p_partial = max(near - far, 0.0)
        p_keep = max(1.0 - near, 0.0)
    else:
        p_max_loss = 1.0 - far
        p_partial = max(far - near, 0.0)
        p_keep = max(near, 0.0)
    total = p_keep + p_partial + p_max_loss
    if total <= 0:
        raise ValueError("zone probabilities collapsed to zero")
    return p_keep / total, p_partial / total, p_max_loss / total


def expected_intrinsic(
    kind: str,
    spot_price: float,
    short_strike: float,
    long_strike: float,
    years: float,
    sigma: float,
    rate: float = 0.0,
) -> float:
    """E[payoff at expiry] for the spread, in USD per unit of underlying.

    The payoff is the short leg minus the long leg, so the expectation is
    ``bs(short) - bs(long)`` whatever the caller's view of ``sigma`` is
    (the leading sign differs by kind, and the two legs are the same kind, so
    it cancels). That is what makes the edge computable: the credit is the
    market's price of this payoff, this is the payoff under a vol *you* choose,
    and the difference is the bet.
    """
    return bs_price(kind, spot_price, short_strike, years, sigma, rate) - bs_price(
        kind, spot_price, long_strike, years, sigma, rate
    )


def strike_for_delta(
    kind: str,
    target_delta: float,
    spot_price: float,
    years: float,
    sigma: float,
    rate: float = 0.0,
) -> float:
    """Invert delta to a strike. ``target_delta`` is magnitude (0 < d < 1)."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta!r}")
    if years <= 0 or sigma <= 0:
        raise ValueError("time and volatility must be positive")

    z = _NORMAL.inv_cdf(target_delta)
    d1 = z if kind == "call" else -z
    log_moneyness = d1 * sigma * math.sqrt(years) - (rate + 0.5 * sigma**2) * years
    return spot_price * math.exp(-log_moneyness)


@dataclass(frozen=True)
class SpreadPlan:
    """A defined-risk credit spread, priced and sized."""

    strategy: str
    short_strike: float
    long_strike: float
    width: float
    credit: float
    max_loss_per_contract: float
    breakeven: float
    pop: float
    credit_ratio: float
    dte_days: int
    contracts: int
    iv_short: float
    iv_long: float
    #: What the exchange marks the spread at, for comparison with ``credit``.
    credit_mark: float = 0.0
    #: ``book`` = priced at bid(short)/ask(long); ``mark`` = mid only;
    #: ``model`` = Black-Scholes because the chain had no prices at all.
    credit_source: str = "model"
    #: The expiry actually priced, and its real distance in days. ``dte_days``
    #: is what was *asked for*; a chain rarely has an expiry exactly there, so
    #: reporting the request as if it were the fact is how "7d" becomes 9d.
    expiry: str | None = None
    dte_actual: float | None = None
    #: The two bands a two-outcome POP hides.
    pop_partial: float = 0.0
    pop_max_loss: float = 0.0
    #: Underlying per contract. Deribit ETH options are 1 ETH, not 100 shares.
    contract_size: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_credit(self) -> float:
        return self.credit * self.contract_size * self.contracts

    @property
    def total_max_loss(self) -> float:
        return self.max_loss_per_contract * self.contract_size * self.contracts

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "short_strike": round(self.short_strike, 2),
            "long_strike": round(self.long_strike, 2),
            "width": round(self.width, 2),
            "credit": round(self.credit, 4),
            "credit_mark": round(self.credit_mark, 4),
            "credit_source": self.credit_source,
            "max_loss_per_contract": round(self.max_loss_per_contract, 4),
            "breakeven": round(self.breakeven, 2),
            "pop": round(self.pop, 4),
            "pop_partial": round(self.pop_partial, 4),
            "pop_max_loss": round(self.pop_max_loss, 4),
            "credit_ratio": round(self.credit_ratio, 4),
            "dte_days": self.dte_days,
            "dte_actual": None if self.dte_actual is None else round(self.dte_actual, 2),
            "expiry": self.expiry,
            "contracts": self.contracts,
            "contract_size": self.contract_size,
            "total_credit": round(self.total_credit, 2),
            "total_max_loss": round(self.total_max_loss, 2),
            "iv_short": round(self.iv_short, 4),
            "iv_long": round(self.iv_long, 4),
            "notes": list(self.notes),
        }


def position_size(
    equity: float, max_risk_pct: float, width: float, credit: float, *, multiplier: float = 100.0
) -> int:
    """Whole contracts such that max loss stays inside the risk budget."""
    if equity <= 0:
        raise ValueError("equity must be positive")
    if not 0 < max_risk_pct <= 1:
        raise ValueError("max_risk_pct must be a fraction in (0, 1]")
    max_loss = (width - credit) * multiplier
    if max_loss <= 0:
        raise ValueError("max loss must be positive; credit exceeds width")
    budget = equity * max_risk_pct
    return max(0, int(budget // max_loss))


def _leg_delta(kind: str, spot_price: float, strike: float, years: float, sigma: float) -> float:
    """|delta| of one leg, signed as a magnitude so puts and calls compare."""
    d1 = _d1(spot_price, strike, years, sigma, 0.0)
    return _NORMAL.cdf(d1) if kind == "call" else _NORMAL.cdf(-d1)


def _listed_book(chain: Any, kind: str, dte_days: int, spot_price: float) -> Any | None:
    """The tradable ladder, when the chain can supply one.

    A chain that only implements ``iv`` still works -- it just cannot say
    whether a strike exists or what it bids, and the plan records ``model``.
    """
    getter = getattr(chain, "book", None)
    if not callable(getter):
        return None
    try:
        book = getter(kind, dte_days, spot_price)
    except Exception as exc:  # a chain that cannot build a book is not fatal
        log.warning("no listed ladder for %s %dd: %s", kind, dte_days, exc)
        return None
    return book if book is not None and len(getattr(book, "legs", ())) >= 2 else None


def _pick_atm(
    book: Any, kind: str, spot_price: float, width: float | None = None
) -> tuple[Any, Any] | None:
    """Pick the spread legs around the strike nearest the spot (the ATM leg).

    ``ATM`` here means the listed strike closest to the current price, and the
    candidate sells that leg: for a call spread the short strike is the lowest
    listed call above spot, for a put spread the highest listed put below it.
    The hedge is one step further out (or ``width`` away), so the structure is
    still two listed strikes with the width the planner gates on.

    Selling closer to the money is what makes a spread tradeable at all: a
    far-OTM short leg produces a thin credit against a wide spread, and a thin
    credit fails the ``min_credit_ratio`` gate even in a clean trend.
    """
    legs = sorted(getattr(book, "legs", ()) or (), key=lambda leg: float(leg.strike))
    if kind == "call":
        above = [leg for leg in legs if float(leg.strike) > spot_price]
        if not above:
            return None
        short_leg = above[0]
        far = [leg for leg in above if float(leg.strike) > float(short_leg.strike)]
    elif kind == "put":
        below = [leg for leg in legs if float(leg.strike) < spot_price]
        if not below:
            return None
        short_leg = below[-1]
        far = [leg for leg in below if float(leg.strike) < float(short_leg.strike)]
    else:
        return None
    if not far:
        return None
    if width is None:
        long_leg = far[0] if kind == "call" else far[-1]
    else:
        wanted = float(short_leg.strike) + (width if kind == "call" else -width)
        long_leg = min(far, key=lambda leg: abs(float(leg.strike) - wanted))
    if abs(float(long_leg.strike) - float(short_leg.strike)) < 1e-9:
        return None
    return short_leg, long_leg


def _pick_strikes(
    book: Any, kind: str, spot_price: float, years: float, target_delta: float, width: float | None
) -> tuple[Any, Any] | None:
    """Short leg at the delta closest to target, hedge one step (or ``width``) away.

    Both legs come from the ladder, so the strikes are instruments that exist.
    """
    legs = list(book.legs)
    short = min(
        legs, key=lambda leg: abs(_leg_delta(kind, spot_price, leg.strike, years, leg.iv) - target_delta)
    )
    hedges = [leg for leg in legs if (leg.strike < short.strike if kind == "put" else leg.strike > short.strike)]
    if not hedges:
        return None
    if width is None:
        long_leg = max(hedges, key=lambda leg: leg.strike) if kind == "put" else min(hedges, key=lambda leg: leg.strike)
    else:
        wanted = short.strike - width if kind == "put" else short.strike + width
        long_leg = min(hedges, key=lambda leg: abs(leg.strike - wanted))
    if abs(long_leg.strike - short.strike) < 1e-9:
        return None
    return short, long_leg


def spread_plan(
    direction: str,
    spot_price: float,
    chain: OptionsChain,
    *,
    target_delta: float = 0.30,
    dte_days: int = 30,
    width: float | None = None,
    min_credit_ratio: float = DEFAULT_MIN_CREDIT_RATIO,
    equity: float | None = None,
    max_risk_pct: float | None = None,
    rate: float = 0.0,
    contract_size: float = 1.0,
    anchor: str = "atm",
) -> tuple[SpreadPlan | None, str]:
    """Build a bull put spread (bullish) or bear call spread (bearish).

    Returns ``(plan, reason)``. ``plan`` is ``None`` when the trade should not be
    taken, with ``reason`` explaining why -- notably a credit too thin to justify
    the width. AGENTS.md restricts this agent to credit spreads only, so no
    debit structure is ever produced.

    ``anchor`` selects where the short leg sits: ``"atm"`` takes the listed
    strike nearest the spot (selling into the money-adjacent zone) and hedges one
    step out; ``"delta"`` takes the leg nearest the requested short-leg delta.
    The production candidate is ``"atm"``: a nearer short strike is where the
    credit actually is, and a far-OTM short leg is a thin credit wearing a wide
    spread, which the credit gate then refuses.

    Two rules keep the credit honest:

    * **listed strikes only.** The short leg is the nearest strike that exists,
      not an interpolated 2211 that no one can sell. ``chain.iv`` still
      interpolates when it is asked about an arbitrary strike; the planner no
      longer asks.
    * **the credit is what you could get.** When both legs have a two-sided
      market it is ``bid(short) - ask(long)``, and ``credit_mark`` keeps the
      mid beside it. A mark is what the exchange thinks the spread is worth;
      the bid/ask is what a seller receives.
    """
    if direction == BULLISH:
        kind, strategy = "put", "bull_put_spread"
    elif direction == BEARISH:
        kind, strategy = "call", "bear_call_spread"
    else:
        return None, "regime is neutral; credit spreads need a directional bias"

    if spot_price <= 0:
        raise ValueError("spot must be positive")
    if dte_days <= 0:
        raise ValueError("dte_days must be positive")
    if width is not None and width <= 0:
        raise ValueError("width must be positive")

    notes: list[str] = []
    contract_size = float(getattr(chain, "contract_size", contract_size) or contract_size)

    #: The frame the strikes are chosen and priced in. The chain's own
    #: underlying when there is a ladder, the caller's spot otherwise.
    price_spot = spot_price
    book = _listed_book(chain, kind, dte_days, spot_price)
    expiry: str | None = None
    dte_actual: float | None = None
    credit_mark = 0.0
    credit_source = "model"

    if book is not None:
        # Price the structure in the venue's own frame. The scanner's spot and
        # the chain's underlying are the same asset seen by two venues, and a
        # 2% gap moves the 0.30-delta strike by a whole rung.
        price_spot = float(getattr(book, "spot", 0.0) or spot_price)
        if price_spot > 0 and abs(price_spot - spot_price) / spot_price > 0.005:
            notes.append(
                f"chain underlying {price_spot:,.2f} vs scanner spot {spot_price:,.2f} "
                f"({price_spot / spot_price - 1:+.2%}); priced in the chain's frame"
            )
        actual_days = book.days_to_expiry()
        years = (actual_days / 365.0) if actual_days > 0 else dte_days / 365.0
        if anchor == "atm":
            picked = _pick_atm(book, kind, price_spot, width)
            if picked is None:
                return None, f"no listed {kind} pair brackets the money at {dte_days}d"
            short_leg, long_leg = picked
        elif anchor == "delta":
            picked = _pick_strikes(book, kind, price_spot, years, target_delta, width)
            if picked is None:
                return None, f"no listed {kind} pair brackets {target_delta:.0%} delta at {dte_days}d"
            short_leg, long_leg = picked
        else:
            raise ValueError(f"unknown strike anchor {anchor!r}")
        short_strike = float(short_leg.strike)
        long_strike = float(long_leg.strike)
        width = abs(short_strike - long_strike)
        expiry = book.expiry.date().isoformat()
        dte_actual = round(actual_days, 2)
        iv_short = max(float(short_leg.iv), 0.01)
        iv_long = max(float(long_leg.iv), 0.01)

        credit_mark = (float(short_leg.mark) - float(long_leg.mark)) * price_spot
        if short_leg.two_sided and long_leg.two_sided:
            credit = (float(short_leg.bid) - float(long_leg.ask)) * price_spot
            credit_source = "book"
            notes.append(
                "credit is executable: bid(short) - ask(long), "
                f"mark would be {credit_mark:.2f}"
            )
        else:
            credit = credit_mark
            credit_source = "mark"
            notes.append("one leg has no two-sided market; credit is the mark, not a price")
        if credit <= 0 and credit_source != "model":
            credit = expected_intrinsic(
                kind, price_spot, short_strike, long_strike, years, iv_short, rate
            )
            credit_source = "model"
            notes.append("book credit was not positive; fell back to Black-Scholes")
        notes.append(f"strikes snapped to the listed ladder ({book.source}); width is one step")
    else:
        years = dte_days / 365.0
        if width is None:
            width = max(round(spot_price * 0.02, -1), 10.0)
        sigma = max(chain.iv(kind, dte_days, spot_price, spot_price), 0.01)
        for _ in range(2):
            short_strike = strike_for_delta(kind, target_delta, spot_price, years, sigma, rate)
            sigma = max(chain.iv(kind, dte_days, short_strike, spot_price), 0.01)
        long_strike = short_strike - width if kind == "put" else short_strike + width
        if long_strike <= 0:
            return None, f"long strike {long_strike:.2f} is not a valid price"
        iv_short = max(chain.iv(kind, dte_days, short_strike, spot_price), 0.01)
        iv_long = max(chain.iv(kind, dte_days, long_strike, spot_price), 0.01)
        credit = expected_intrinsic(kind, spot_price, short_strike, long_strike, years, iv_short, rate)
        credit_mark = credit
        notes.append("no listed ladder available; strikes and credit are interpolated")

    if credit <= 0:
        return None, f"structure is not a credit (credit {credit:.4f})"

    max_loss = width - credit
    if max_loss <= 0:
        return None, f"credit {credit:.2f} exceeds width {width:.2f}; not a spread"

    credit_ratio = credit / width
    if credit_ratio < min_credit_ratio:
        return None, (
            f"credit {credit:.2f} is only {credit_ratio:.0%} of width {width:.2f} "
            f"(minimum {min_credit_ratio:.0%}); risk/reward does not justify the trade"
        )

    breakeven = short_strike - credit if kind == "put" else short_strike + credit
    pop, pop_partial, pop_max_loss = zone_probabilities(
        kind, price_spot, short_strike, long_strike, years, iv_short
    )
    notes.append(
        f"POP {pop:.0%} is N(-d2) at the short strike's own vol ({iv_short:.1%}), "
        "not 1 - delta; the middle band loses part of the credit"
    )

    contracts = 0
    if equity is None:
        notes.append("sizing skipped: no account equity supplied")
    else:
        risk = max_risk_pct if max_risk_pct is not None else 0.01
        contracts = position_size(equity, risk, width, credit, multiplier=contract_size)
        if contracts == 0:
            # A structure you cannot size is not a trade.
            return None, (
                f"max loss {max_loss * contract_size:.2f} per contract exceeds the "
                f"{risk:.1%} risk budget of {equity * risk:.2f}; not affordable"
            )
        notes.append(
            f"sized to {risk:.1%} of {equity:,.2f} equity "
            f"(max loss {max_loss * contract_size * contracts:,.2f})"
        )

    plan = SpreadPlan(
        strategy=strategy,
        short_strike=short_strike,
        long_strike=long_strike,
        width=width,
        credit=credit,
        max_loss_per_contract=max_loss,
        breakeven=breakeven,
        pop=pop,
        credit_ratio=credit_ratio,
        dte_days=dte_days,
        contracts=contracts,
        iv_short=iv_short,
        iv_long=iv_long,
        credit_mark=credit_mark,
        credit_source=credit_source,
        expiry=expiry,
        dte_actual=dte_actual,
        pop_partial=pop_partial,
        pop_max_loss=pop_max_loss,
        contract_size=contract_size,
        notes=notes,
    )
    return plan, "ok"


def trade_scenario(
    regime_result: Regime,
    spot_price: float,
    chain: OptionsChain,
    **plan_kwargs: Any,
) -> dict[str, Any]:
    """Combine the regime read with the spread plan (keeps the original name)."""
    plan, reason = spread_plan(regime_result.direction, spot_price, chain, **plan_kwargs)
    return {
        "regime": regime_result.to_dict(),
        "plan": None if plan is None else plan.to_dict(),
        "decision": "NO TRADE" if plan is None else "CANDIDATE",
        "reason": reason,
        "disclaimer": "indicative analysis only; not investment advice, not an order",
    }
