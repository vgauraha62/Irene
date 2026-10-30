"""Labels, base rates, and the split that keeps them honest.

This module answers one question and deliberately refuses a second.

**Asked:** given the state the agent saw at time ``t``, how often did the move it
would have bet on actually happen -- and how does that compare with the rate at
which such a move happens anyway? That second number, the base rate, is the only
thing a predictor has to beat. "70% of squeezes led to a trend" is worse than
useless if a trend starts after 70% of *all* windows.

**Not asked:** whether any signal works. No fit, no weights, no model. Phase 3
builds the ruler; Phase 4 spends it.

Three rules that make the number mean something:

* **The state is computed the live way.** ``regime`` is called on snapshots built
  from the same bars, with the same EMA ladder, the same flat band and the same
  ADX floor. A backtest that asks a different question than production is a
  different backtest.
* **Nothing sees the future.** ``state_at`` reads only bars up to ``i``; every
  derived series (EMA, ATR, ADX) is indexed at ``i``, never computed on the full
  history and then applied backwards. ``test_state_at_cannot_see_the_future`` is
  the guard, and it works by grafting a spike onto the future and asserting the
  past is unchanged.
* **The threshold is ex-ante.** "A real move" is ``k`` times the *trailing*
  realised sigma, scaled by sqrt(horizon). A fixed percentage would call the same
  move significant in a dead market and noise in a volatile one.

Overlap is the quiet killer: a 4h label sampled every 15 minutes shares 15/16 of
its window with the next one, so 96 observations a day are not 96 independent
trials. ``Spec.stride`` defaults to the horizon, which makes consecutive samples
disjoint by construction, and ``purged_split`` removes the seam between train and
test. The reported ``n`` is then the number of *non-overlapping* windows.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import analysis
import bars

#: Timeframes the regime needs, shortest first. Same tuple ``analysis.regime``
#: demands -- if this drifts, the backtest stops asking the live question.
TIMEFRAMES: tuple[str, ...] = ("15m", "30m", "4h")

#: Resample factor from the stored 15m series.
FACTORS: dict[str, int] = {"15m": 1, "30m": 2, "4h": 16}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Spec:
    """What counts as an event, and how often it is sampled.

    ``horizon_bars`` is in 15m bars: 16 is 4h, 96 is a day. ``stride`` defaults
    to the horizon so consecutive labels do not overlap; lowering it buys samples
    and pays for them in correlation, which the Wilson intervals will then
    understate.
    """

    horizon_bars: int = 16
    stride: int = 0  # 0 -> stride = horizon_bars
    size_sigma: float = 0.5
    vol_lookback: int = 200
    adx_period: int = 14

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.stride < 0:
            raise ValueError("stride must be >= 0")
        if self.size_sigma <= 0:
            raise ValueError("size_sigma must be positive")

    @property
    def step(self) -> int:
        return self.stride or self.horizon_bars

    @property
    def horizon_hours(self) -> float:
        return self.horizon_bars * bars.INTERVAL_MINUTES["15m"] / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_bars": self.horizon_bars,
            "horizon_hours": self.horizon_hours,
            "stride_bars": self.step,
            "size_sigma": self.size_sigma,
            "vol_lookback": self.vol_lookback,
            "overlapping": self.step < self.horizon_bars,
        }


# --------------------------------------------------------------------------- #
# State, computed from bars with the live maths
# --------------------------------------------------------------------------- #


@dataclass
class Ladder:
    """Every derived series the state needs, computed once over the whole store.

    Built once and indexed, rather than recomputed per sample: the point is that
    ``state_at(i)`` observes exactly the bars up to ``i``, and a precomputed
    series indexed at ``i`` satisfies that without an O(n^2) rescrape of the
    history.
    """

    bars: list[bars.Bar]
    series: dict[str, list[bars.Bar]]
    ema: dict[str, dict[str, list[float | None]]]
    adx: dict[str, list[float | None]]
    position: dict[str, list[int]]
    first_valid: dict[str, int] = field(default_factory=dict)

    def state_at(self, index: int) -> analysis.Regime | None:
        """The regime at ``index``, or ``None`` before the ladder has converged."""
        for timeframe in TIMEFRAMES:
            if self.position[timeframe][index] < self.first_valid.get(timeframe, 0):
                return None
        return analysis.regime(self.snapshots_at(index))

    def snapshots_at(self, index: int) -> dict[str, dict[str, Any]]:
        """Snapshot dictionaries shaped like the live feed, for ``analysis``."""
        out: dict[str, dict[str, Any]] = {}
        for timeframe in TIMEFRAMES:
            position = self.position[timeframe][index]
            values: dict[str, Any] = {}
            for key in analysis.MA_KEYS:
                values[key] = self.ema[timeframe][key][position]
            values["close"] = self.series[timeframe][position].close
            strength = self.adx[timeframe][position]
            if strength is not None:
                values["ADX"] = strength
            out[timeframe] = {"timeframe": timeframe, "indicators": values}
        return out


def _last_closed(opens: Sequence[datetime], moment: datetime, span: timedelta) -> int:
    """Index of the last bar in ``opens`` that had closed by ``moment``."""
    return bisect.bisect_right(opens, moment - span) - 1


def build_ladder(store: Sequence[bars.Bar], *, adx_period: int = 14) -> Ladder:
    """Resample once, derive once, and record how the timeframes line up."""
    if len(store) < 2:
        raise ValueError("need at least two bars")

    series: dict[str, list[bars.Bar]] = {}
    for timeframe in TIMEFRAMES:
        factor = FACTORS[timeframe]
        series[timeframe] = list(store) if factor == 1 else bars.resample(store, factor, "15m")

    ema: dict[str, dict[str, list[float | None]]] = {}
    adx: dict[str, list[float | None]] = {}
    position: dict[str, list[int]] = {}
    first_valid: dict[str, int] = {}

    span_15m = timedelta(seconds=bars.INTERVAL_SECONDS["15m"])

    for timeframe in TIMEFRAMES:
        rows = series[timeframe]
        closes = [bar.close for bar in rows]
        ema[timeframe] = {
            key: bars.ema_series(closes, int(key.removeprefix("EMA")))
            for key in analysis.MA_KEYS
        }
        adx[timeframe] = bars.adx_series(rows, adx_period)

        opens = [bar.open_time for bar in rows]
        span = timedelta(seconds=bars.INTERVAL_SECONDS[timeframe])
        position[timeframe] = [
            _last_closed(opens, bar.open_time + span_15m, span) for bar in store
        ]

        # Convergence, not mere non-emptiness: an EMA200 seeded 200 bars ago and
        # never revisited is still bleeding its seed into the value.
        needed = max(
            analysis.MIN_BARS_PER_PERIOD * int(key.removeprefix("EMA"))
            for key in analysis.MA_KEYS
        )
        first = next(
            (
                index
                for index in range(len(rows))
                if index >= needed - 1 and adx[timeframe][index] is not None
            ),
            len(rows),
        )
        first_valid[timeframe] = first

    return Ladder(
        bars=list(store),
        series=series,
        ema=ema,
        adx=adx,
        position=position,
        first_valid=first_valid,
    )


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def forward_return_pct(store: Sequence[bars.Bar], index: int, horizon_bars: int) -> float | None:
    """Log return from the close at ``index`` to the close ``horizon`` later."""
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if index < 0 or index + horizon_bars >= len(store):
        return None
    entry = store[index].close
    exit_ = store[index + horizon_bars].close
    if entry <= 0 or exit_ <= 0:
        return None
    return 100.0 * math.log(exit_ / entry)


def sigma_pct(store: Sequence[bars.Bar], index: int, spec: Spec) -> float | None:
    """Trailing realised sigma over ``spec.horizon_hours``, in % of price.

    Ex-ante by construction: the window ends at ``index``. This is the number the
    threshold scales with, so a threshold never becomes easier just because the
    future turned out calm.
    """
    if index < spec.vol_lookback:
        return None
    window = list(store[index - spec.vol_lookback : index + 1])
    return bars.expected_move_pct(window, spec.horizon_hours, lookback=spec.vol_lookback)


def classify(forward_pct: float, sigma: float, spec: Spec) -> str:
    """``up`` / ``down`` / ``flat`` against ``size_sigma`` trailing sigmas."""
    threshold = spec.size_sigma * sigma
    if forward_pct >= threshold:
        return "up"
    if forward_pct <= -threshold:
        return "down"
    return "flat"


@dataclass(frozen=True)
class Sample:
    """One decision point and what followed it."""

    index: int
    moment: str
    direction: str
    agreement: int
    adx: float | None
    drift_pct_per_hour: dict[str, float]
    sigma_pct: float
    forward_pct: float
    label: str

    @property
    def traded(self) -> bool:
        """Exactly the live gate: directional, unanimous, and out of chop."""
        return self.direction in {analysis.BULLISH, analysis.BEARISH} and self.agreement == 3

    @property
    def hit(self) -> bool:
        """Did price move the way the label claimed, by at least the threshold?"""
        if self.direction == analysis.BULLISH:
            return self.label == "up"
        if self.direction == analysis.BEARISH:
            return self.label == "down"
        return False

    @property
    def signed(self) -> bool:
        """Did price move the claimed way *at all*, threshold ignored?

        Kept beside ``hit`` because the two fail differently: a middling sign
        rate with a poor thresholded rate means the direction is a coin flip; a
        good sign rate with a poor thresholded rate means the direction is right
        but the move is too small to pay for a spread.
        """
        if self.direction == analysis.BULLISH:
            return self.forward_pct > 0
        if self.direction == analysis.BEARISH:
            return self.forward_pct < 0
        return False


def sample(ladder: Ladder, spec: Spec) -> list[Sample]:
    """Every decision point the store supports, with no lookahead."""
    out: list[Sample] = []
    last_start = len(ladder.bars) - spec.horizon_bars
    index = 0
    while index < last_start:
        state = ladder.state_at(index)
        if state is None:
            index += 1  # warmup is a scan, not a fixed constant: it depends on history
            continue
        forward = forward_return_pct(ladder.bars, index, spec.horizon_bars)
        sigma = sigma_pct(ladder.bars, index, spec)
        if forward is None or sigma is None or sigma <= 0:
            index += 1
            continue
        out.append(
            Sample(
                index=index,
                moment=ladder.bars[index].open_time.isoformat(),
                direction=state.direction,
                agreement=int(state.agreement),
                adx=_adx_of(state),
                drift_pct_per_hour=analysis.drifts(ladder.snapshots_at(index)),
                sigma_pct=sigma,
                forward_pct=forward,
                label=classify(forward, sigma, spec),
            )
        )
        index += spec.step
    return out


def _adx_of(state: analysis.Regime) -> float | None:
    return state.adx


# --------------------------------------------------------------------------- #
# Rates, with an interval the sample size actually earns
# --------------------------------------------------------------------------- #


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent.

    A plain ``k/n`` on 12 observations says 8.3% with invisible error bars; the
    Wilson interval says 8.3% with a floor near 2% and a ceiling near 26%. The
    second is the honest one, and it is why small buckets are reported rather
    than rounded away.
    """
    if trials <= 0:
        return (0.0, 100.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = (z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))) / denominator
    return (max(0.0, 100.0 * (centre - half)), min(100.0, 100.0 * (centre + half)))


def _rate(successes: int, trials: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, trials)
    return {
        "n": trials,
        "count": successes,
        "rate": round(100.0 * successes / trials, 2) if trials else None,
        "ci": [round(low, 2), round(high, 2)],
    }


def _summarise(rows: Sequence[Sample]) -> dict[str, Any]:
    n = len(rows)
    counts = {label: sum(1 for row in rows if row.label == label) for label in ("up", "down", "flat")}
    summary: dict[str, Any] = {
        "n": n,
        "up": _rate(counts["up"], n),
        "down": _rate(counts["down"], n),
        "flat": _rate(counts["flat"], n),
        "mean_forward_pct": round(statistics.fmean([row.forward_pct for row in rows]), 4) if n else None,
        "mean_forward_sigma": (
            round(statistics.fmean([row.forward_pct / row.sigma_pct for row in rows]), 3) if n else None
        ),
    }
    return summary


def base_rate(samples: Sequence[Sample]) -> dict[str, Any]:
    """The unconditional rates every conditional claim has to beat."""
    return _summarise(samples)


def buckets(samples: Sequence[Sample]) -> dict[str, dict[str, Any]]:
    """Rates grouped by the agent's own label, plus the gated set.

    ``signal`` is the bucket that matters: the moments where the live regime
    produced a direction (3/3 agreement, out of chop). The third gate -- credit
    at least 20% of width -- is *not* modelled, because it needs a history of
    option prices this agent does not have. Treat ``signal`` as an upper bound on
    how often a trade was available, never as the trade count.
    """
    groups: dict[str, list[Sample]] = {
        "bullish": [row for row in samples if row.direction == analysis.BULLISH],
        "bearish": [row for row in samples if row.direction == analysis.BEARISH],
        "neutral": [row for row in samples if row.direction == analysis.NEUTRAL],
        "signal": [row for row in samples if row.traded],
    }
    base = base_rate(samples)
    out: dict[str, dict[str, Any]] = {}
    for name, rows in groups.items():
        summary = _summarise(rows)
        summary["hit"] = _rate(sum(1 for row in rows if row.hit), len(rows))
        summary["signed"] = _rate(sum(1 for row in rows if row.signed), len(rows))
        reference = _reference_rate(rows, base)
        hit = summary["hit"]
        if reference and hit["rate"] is not None:
            summary["lift"] = round(hit["rate"] / reference, 3)
            summary["lift_ci"] = [
                round(hit["ci"][0] / reference, 3),
                round(hit["ci"][1] / reference, 3),
            ]
        else:
            summary["lift"] = None
            summary["lift_ci"] = None
        out[name] = summary
    return out


def _reference_rate(rows: Sequence[Sample], base: Mapping[str, Any]) -> float | None:
    """The base rate for the *same* direction as this bucket's rows.

    Comparing a bullish hit rate with the overall up rate is the only comparison
    that isolates the signal: in a market that rises 60% of the time, 60% is a
    coin flip, not an edge. For a mixed bucket (``signal`` holds both sides) the
    reference is the mean of the per-row references, so the lift still means
    "versus what happens anyway".
    """
    references = [
        base["up"]["rate"] if row.direction == analysis.BULLISH else base["down"]["rate"]
        for row in rows
        if row.direction in {analysis.BULLISH, analysis.BEARISH}
    ]
    references = [value for value in references if value]
    return statistics.fmean(references) if references else None


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def purged_split(count: int, *, test_fraction: float = 0.3, gap: int = 1) -> tuple[list[int], list[int]]:
    """Chronological train/test with the seam removed.

    Labels overlap in time even when the *samples* do not: a training observation
    whose horizon reaches into the test period has seen the future it is being
    tested on. ``gap`` samples between the two blocks kills that, and needs to be
    ``ceil(horizon_bars / stride)`` -- which is 1 for the default non-overlapping
    sampling, and is why the default is honest.
    """
    if count < 2:
        return ([], [])
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    cut = int(round(count * (1.0 - test_fraction)))
    cut = min(max(cut, 1), count - 1)
    train = list(range(0, max(0, cut - max(0, gap))))
    test = list(range(cut, count))
    return (train, test)


def assert_disjoint(train: Sequence[int], test: Sequence[int], *, stride: int, horizon: int) -> None:
    """Raise if any train label window reaches into the test block."""
    if not train or not test:
        return
    last_train_end = max(train) * stride + horizon
    first_test_moment = min(test) * stride
    if last_train_end > first_test_moment:
        raise ValueError(
            f"train label at {max(train)} reaches bar {last_train_end}, "
            f"past the first test sample at bar {first_test_moment}"
        )


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def table(store: Sequence[bars.Bar], spec: Spec | None = None) -> dict[str, Any]:
    """Base rate, conditional rates and lift, all from one pass over the store."""
    spec = spec or Spec()
    ladder = build_ladder(store, adx_period=spec.adx_period)
    samples = sample(ladder, spec)
    if not samples:
        raise ValueError(
            f"no usable samples: {len(store)} bars, warmup needs about "
            f"{_warmup_bars(ladder)} before the first 4h EMA200 has converged"
        )

    warnings: list[str] = []
    if spec.step < spec.horizon_bars:
        warnings.append(
            f"stride {spec.step} < horizon {spec.horizon_bars}: labels overlap, "
            "so the confidence intervals understate the real uncertainty"
        )
    if len(samples) < 100:
        warnings.append(f"only {len(samples)} non-overlapping samples; treat every rate as provisional")

    train, test = purged_split(len(samples), gap=max(1, math.ceil(spec.horizon_bars / spec.step)))
    assert_disjoint(train, test, stride=spec.step, horizon=spec.horizon_bars)

    first = store[0].open_time.isoformat()
    last = store[-1].open_time.isoformat()
    return {
        "product_bars": {"count": len(store), "from": first, "to": last},
        "spec": spec.to_dict(),
        "warmup_bars": _warmup_bars(ladder),
        "base": base_rate(samples),
        "buckets": buckets(samples),
        "split": {"train": len(train), "test": len(test), "gap": max(1, math.ceil(spec.horizon_bars / spec.step))},
        "warnings": warnings,
    }


def _warmup_bars(ladder: Ladder) -> int:
    return max(ladder.first_valid.get(timeframe, 0) for timeframe in TIMEFRAMES) * FACTORS["4h"]


def render(report: Mapping[str, Any]) -> str:
    """Plain text, because this is read once and judged, not charted."""
    product = report["product_bars"]
    spec = report["spec"]
    base = report["base"]
    lines: list[str] = []

    lines.append(
        f"bars       {product['count']:,}  {product['from'][:16]} -> {product['to'][:16]}"
    )
    lines.append(
        f"horizon    {spec['horizon_hours']:.1f}h   stride {spec['stride_bars']} bars"
        f"   event = |move| >= {spec['size_sigma']:g} sigma"
    )
    lines.append(f"warmup     {report['warmup_bars']:,} bars before the ladder converges")
    lines.append("")

    lines.append(f"base rate  n={base['n']:,} non-overlapping windows")
    for label in ("up", "down", "flat"):
        item = base[label]
        if item["rate"] is None:
            continue
        lines.append(
            f"  {label:<5}{item['rate']:>7.2f}%   [{item['ci'][0]:.1f}, {item['ci'][1]:.1f}]"
            f"   {item['count']:,}/{item['n']:,}"
        )
    lines.append(
        f"  mean forward {base['mean_forward_pct']:+.3f}%"
        f"  ({base['mean_forward_sigma']:+.2f} sigma)"
    )
    lines.append("")

    lines.append("by regime  (the agent's own label, computed the live way)")
    lines.append(
        f"  {'bucket':<10}{'n':>7}{'up%':>8}{'down%':>8}{'flat%':>8}{'hit%':>8}{'dir%':>8}{'lift':>8}"
    )
    for name, row in report["buckets"].items():
        if not row["n"]:
            lines.append(f"  {name:<10}{0:>7}")
            continue
        lift = "-" if row["lift"] is None else f"{row['lift']:.2f}x"
        signed = "-" if row["signed"]["rate"] is None else f"{row['signed']['rate']:.1f}"
        lines.append(
            f"  {name:<10}{row['n']:>7,}{row['up']['rate']:>8.1f}{row['down']['rate']:>8.1f}"
            f"{row['flat']['rate']:>8.1f}{row['hit']['rate']:>8.1f}{signed:>8}{lift:>8}"
        )
    lines.append("  hit% = moved the claimed way by >= the threshold; dir% = moved that way at all")
    lines.append("")

    signal = report["buckets"].get("signal") or {}
    if signal.get("n"):
        hit = signal["hit"]
        direction = signal["signed"]
        lift = "-" if signal["lift"] is None else f"{signal['lift']:.2f}x"
        if signal.get("lift_ci"):
            lift += f"  [{signal['lift_ci'][0]:.2f}, {signal['lift_ci'][1]:.2f}]"
        lines.append(
            f"signal     {signal['n']:,} of {base['n']:,} windows produced a direction"
            f" (3/3 agreement, out of chop)"
        )
        lines.append(
            f"           hit {hit['rate']:.1f}%  [{hit['ci'][0]:.1f}, {hit['ci'][1]:.1f}]"
            f"   direction {direction['rate']:.1f}%  [{direction['ci'][0]:.1f}, {direction['ci'][1]:.1f}]"
            f"   lift {lift}"
        )
        lines.append(
            "           lift 1.00x means the signal is worth exactly what happens anyway."
            " An interval containing 1.00x means this history cannot tell the difference."
        )
        lines.append(
            "           The credit gate (20% of width) is not modelled: it needs option history."
        )
    else:
        lines.append("signal     the gate never fired in this history")
    lines.append("")
    lines.append(
        f"split      train {report['split']['train']:,} / test {report['split']['test']:,}"
        f"   gap {report['split']['gap']} sample(s) removed at the seam"
    )
    for warning in report["warnings"]:
        lines.append(f"warning    {warning}")
    lines.append("")
    lines.append(
        "No predictor is fitted here. These are the rates any predictor has to beat,"
        " and until it does, every projection stays a diagnostic."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Base rates and conditional rates from stored bars. No model, no forecast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bars", default="out/bars/ETHUSDT_15m.csv", help="bar store to read")
    parser.add_argument("--backfill", type=int, default=0, help="fetch this many extra pages of 1000 bars first")
    parser.add_argument("--horizon", type=float, default=4.0, help="label horizon in hours")
    parser.add_argument("--stride", type=float, default=None, help="sampling step in hours (default: the horizon)")
    parser.add_argument("--sigma", type=float, default=0.5, help="event size, in trailing sigmas")
    parser.add_argument("--json", default=None, help="write the report here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.bars)

    if args.backfill:
        fetched = bars.backfill(path, pages=args.backfill)
        print(
            f"backfill   {fetched['appended']:,} new bars from {fetched['pages']} page(s);"
            f" store now {fetched.get('stored', 0):,}"
        )

    store = bars.load_bars(path)
    if not store:
        print(f"error: no bars in {path}; run with --backfill 20 or the main pipeline once")
        return 1

    horizon_bars = max(1, int(round(args.horizon * 60 / bars.INTERVAL_MINUTES["15m"])))
    stride = 0 if args.stride is None else max(1, int(round(args.stride * 60 / bars.INTERVAL_MINUTES["15m"])))
    spec = Spec(horizon_bars=horizon_bars, stride=stride, size_sigma=args.sigma)

    try:
        report = table(store, spec)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    print(render(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI only
    raise SystemExit(main())
