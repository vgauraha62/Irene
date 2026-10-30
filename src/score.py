"""Score past projections against what price actually did.

The headline number is **not** the error. It is the error compared with doing
nothing at all:

    skill = 1 - MAE(projection) / MAE(predict-the-current-price)

A drift extrapolation that cannot beat "assume price stays where it is" has no
skill, no matter how small its error looks. This module exists so that claim can
be checked instead of asserted.

Two honest limitations, encoded rather than documented:

* Observations are irregular. A projection is scored only against a record at or
  after its target time, and only if that record is close enough (``tolerance``,
  as a fraction of the horizon). Anything else is counted as *unscored* with a
  reason, so gaps stay visible.
* Prices come from polled snapshots, not candles. The realised price is a spot
  quote near the horizon, not a bar close.

Usage::

    python src/score.py out/projections.jsonl
    python src/score.py out/projections.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import journal

log = logging.getLogger("score")

#: Accept a realised observation up to this fraction of the horizon late.
#: 25% of a 4h projection is an hour; 25% of a 30m projection is 7.5 minutes.
DEFAULT_TOLERANCE = 0.25

UNSCORED_NO_DATA = "no later observation"
UNSCORED_TOO_LATE = "observation too late"


@dataclass(frozen=True)
class Scored:
    """One projection, one realised price, and the difference between them."""

    horizon_hours: float
    predicted_at: datetime
    realised_at: datetime
    gap_hours: float
    spot_at_prediction: float
    expected_price: float
    realised_price: float
    low: float
    high: float
    direction: str

    @property
    def error_pct(self) -> float:
        """Signed error of the projection, in % of the projected price."""
        return (self.realised_price / self.expected_price - 1.0) * 100.0

    @property
    def baseline_error_pct(self) -> float:
        """Signed error of the random-walk baseline (predict no change)."""
        return (self.realised_price / self.spot_at_prediction - 1.0) * 100.0

    @property
    def in_band(self) -> bool:
        return self.low <= self.realised_price <= self.high

    @property
    def directional(self) -> bool:
        return self.direction in ("bullish", "bearish")

    @property
    def direction_hit(self) -> bool | None:
        """Did price move the way the projection said? ``None`` if it said neutral."""
        if not self.directional:
            return None
        moved_up = self.realised_price > self.spot_at_prediction
        return moved_up if self.direction == "bullish" else not moved_up


@dataclass(frozen=True)
class HorizonScore:
    """Aggregate for one horizon. ``n`` is what was scored, not what was written."""

    horizon_hours: float
    n: int
    mae_pct: float
    baseline_mae_pct: float
    skill: float
    band_coverage: float
    directional_hit_rate: float | None
    n_directional: int
    mean_gap_hours: float
    max_gap_hours: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_hours": self.horizon_hours,
            "n": self.n,
            "mae_pct": round(self.mae_pct, 4),
            "baseline_mae_pct": round(self.baseline_mae_pct, 4),
            "skill": round(self.skill, 4),
            "band_coverage": round(self.band_coverage, 4),
            "directional_hit_rate": (
                None
                if self.directional_hit_rate is None
                else round(self.directional_hit_rate, 4)
            ),
            "n_directional": self.n_directional,
            "mean_gap_hours": round(self.mean_gap_hours, 4),
            "max_gap_hours": round(self.max_gap_hours, 4),
        }


@dataclass(frozen=True)
class ScoreCard:
    scored: list[Scored]
    unscored: dict[str, int]
    records: int
    duplicates: int
    symbol: str | None = None

    @property
    def horizons(self) -> list[HorizonScore]:
        return summarize(self.scored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "records": self.records,
            "duplicates": self.duplicates,
            "scored": len(self.scored),
            "unscored": dict(self.unscored),
            "horizons": [item.to_dict() for item in self.horizons],
        }


def observed_at(record: Mapping[str, Any]) -> datetime:
    """The record's UTC timestamp. Naive stamps are treated as UTC, not guessed."""
    stamp = datetime.fromisoformat(str(record["collected_at"]))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def symbols_of(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({str(record.get("symbol")) for record in records})


def _tolerance_for(horizon_hours: float, tolerance: float) -> float:
    return max(horizon_hours * tolerance, 0.5 / 60.0)


def _dedupe(records: Iterable[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    """Drop exact re-records of the same observation.

    A manual re-run minutes later is a new observation, not a duplicate; the
    same ``collected_at`` is a duplicate. Without this, re-running the CLI twice
    inflates ``n`` and makes the score look better than the data supports.
    """
    seen: set[tuple[Any, Any]] = set()
    unique: list[Mapping[str, Any]] = []
    duplicates = 0
    for record in sorted(records, key=observed_at):
        key = (record.get("symbol"), record.get("collected_at"))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, duplicates


def score_records(
    records: Sequence[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ScoreCard:
    """Pair every projection with the first observation of *the same symbol* at its horizon.

    Symbols must never be mixed. Pairing is by time, so one time array shared by
    an ETHUSD projection and a BTCUSD spot would "score" the ETH forecast against
    the wrong asset and report the result as skill. With ``symbol=None`` and more
    than one symbol present this raises instead of choosing for you.
    """
    present = symbols_of(records)
    if symbol is None:
        if len(present) > 1:
            raise ValueError(
                f"journal holds several symbols {present}; pass symbol= to score one"
            )
        symbol = present[0] if present else None

    selected = [record for record in records if record.get("symbol") == symbol]
    usable, duplicates = _dedupe(selected)
    times = [observed_at(record) for record in usable]
    stamps = [moment.timestamp() for moment in times]
    spots = [float(record["spot"]) for record in usable]

    scored: list[Scored] = []
    unscored: dict[str, int] = {}

    for index, record in enumerate(usable):
        predicted_at = times[index]
        for projection in record.get("projections") or []:
            horizon = float(projection["horizon_hours"])
            target = predicted_at.timestamp() + horizon * 3600.0
            position = bisect_left(stamps, target)

            if position >= len(usable):
                unscored[UNSCORED_NO_DATA] = unscored.get(UNSCORED_NO_DATA, 0) + 1
                continue

            realised_at = times[position]
            gap_hours = (realised_at.timestamp() - target) / 3600.0
            if gap_hours > _tolerance_for(horizon, tolerance):
                unscored[UNSCORED_TOO_LATE] = unscored.get(UNSCORED_TOO_LATE, 0) + 1
                continue

            scored.append(
                Scored(
                    horizon_hours=horizon,
                    predicted_at=predicted_at,
                    realised_at=realised_at,
                    gap_hours=gap_hours,
                    spot_at_prediction=spots[index],
                    expected_price=float(projection["expected_price"]),
                    realised_price=spots[position],
                    low=float(projection["low"]),
                    high=float(projection["high"]),
                    direction=str(projection.get("direction") or "neutral"),
                )
            )

    return ScoreCard(
        scored=scored,
        unscored=unscored,
        records=len(usable),
        duplicates=duplicates,
        symbol=symbol,
    )


def score_journal(
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[ScoreCard]:
    """One card per symbol in the journal, so a mixed journal stays honest."""
    return [
        score_records(records, symbol=symbol, tolerance=tolerance)
        for symbol in symbols_of(records)
    ]


def summarize(scored: Sequence[Scored]) -> list[HorizonScore]:
    """Per-horizon aggregates, including the baseline the skill is measured against."""
    by_horizon: dict[float, list[Scored]] = {}
    for item in scored:
        by_horizon.setdefault(item.horizon_hours, []).append(item)

    out: list[HorizonScore] = []
    for horizon in sorted(by_horizon):
        group = by_horizon[horizon]
        count = len(group)
        # Absolute errors: a mean signed error would let over- and under-shoots
        # cancel and flatter the score.
        mae = sum(abs(item.error_pct) for item in group) / count
        baseline = sum(abs(item.baseline_error_pct) for item in group) / count
        directional = [item for item in group if item.directional]
        hits = [item.direction_hit for item in directional]

        out.append(
            HorizonScore(
                horizon_hours=horizon,
                n=count,
                mae_pct=mae,
                baseline_mae_pct=baseline,
                skill=(1.0 - mae / baseline) if baseline else 0.0,
                band_coverage=sum(1 for item in group if item.in_band) / count,
                directional_hit_rate=(sum(1 for hit in hits if hit) / len(hits))
                if hits
                else None,
                n_directional=len(directional),
                mean_gap_hours=sum(item.gap_hours for item in group) / count,
                max_gap_hours=max(item.gap_hours for item in group),
            )
        )
    return out


def format_card(card: ScoreCard) -> str:
    """A plain-text scorecard. Deliberately blunt about small samples."""
    lines: list[str] = []
    label = f"{card.symbol}: " if card.symbol else ""
    lines.append(
        f"{label}records {card.records}   scored {len(card.scored)}   duplicates {card.duplicates}"
    )
    if card.unscored:
        dropped = ", ".join(f"{reason}: {count}" for reason, count in sorted(card.unscored.items()))
        lines.append(f"unscored {dropped}")

    if not card.scored:
        lines.append("")
        lines.append("nothing to score yet - a run needs a later observation at or past its horizon")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"  {'horizon':>8}{'n':>5}{'MAE %':>10}{'baseline %':>12}{'skill':>8}"
        f"{'in-band':>9}{'dir hit':>9}{'gap h':>8}"
    )
    for item in card.horizons:
        hit = "-" if item.directional_hit_rate is None else f"{item.directional_hit_rate:.0%}"
        lines.append(
            f"  {item.horizon_hours:>7g}h{item.n:>5}{item.mae_pct:>10.4f}"
            f"{item.baseline_mae_pct:>12.4f}{item.skill:>8.2f}"
            f"{item.band_coverage:>9.0%}{hit:>9}{item.max_gap_hours:>8.2f}"
        )

    lines.append("")
    lines.append("skill > 0 means the projection beat 'price does not move';")
    lines.append("skill <= 0 means it did not, and the drift carried no information.")
    if card.records < 30:
        lines.append(
            f"sample is {card.records} runs: consecutive projections overlap, so the "
            "effective sample is smaller still. Treat this as plumbing, not evidence."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The shadow gate, settled
# --------------------------------------------------------------------------- #
#
# A credit spread is not a directional bet, so scoring one against a direction is
# the wrong ruler. The right one is whether the market's implied probability of
# breaching the short strike was too high or too low: sell the wing often enough,
# and the realised breach rate either comes in under what was implied (the
# premium was worth collecting) or over it (it was not).
#
# Each journaled ladder row is one contract priced at a delta on an expiry. It
# becomes one observation when that expiry settles, which is why the ladder is
# journaled at three deltas: three contracts per expiry instead of one.


@dataclass(frozen=True)
class SettledSpread:
    """One contract, as it was priced and as it finished."""

    side: str
    target_delta: float
    expiry: datetime
    short_strike: float
    long_strike: float
    width: float
    credit: float
    sigma_rv: float
    sigma_window: str
    p_keep: float
    p_partial: float
    p_max_loss: float
    settlement: float
    breached: bool  # settled beyond the short strike
    max_loss_hit: bool  # settled beyond the hedge
    pnl: float  # per contract, USD
    near_strike: bool  # too close to call on a cross-venue settlement


def _settlement_index(bars: Sequence[Any]) -> tuple[list[datetime], list[float]]:
    ordered = sorted(bars, key=lambda bar: bar.open_time)
    return [bar.open_time for bar in ordered], [float(bar.close) for bar in ordered]


def _settlement_price(times: Sequence[datetime], closes: Sequence[float], expiry: datetime) -> float | None:
    """The last price printed before settlement.

    Deribit settles on its own index at 08:00 UTC; our bars are a different venue
    on a 15-minute grid. The closest honest answer is the close of the last bar
    that opened before settlement -- and any outcome within 0.5% of a strike is
    flagged, because on this data it is a coin toss whether it breached.
    """
    if not times:
        return None
    index = bisect_left(times, expiry)
    if index == 0:
        return None
    return closes[index - 1]


def _intrinsic(row: Mapping[str, Any], settlement: float) -> float:
    short = float(row["short_strike"])
    long_ = float(row["long_strike"])
    width = abs(short - long_)
    if row["side"] == "put":
        return min(max(short - settlement, 0.0), width)
    return min(max(settlement - short, 0.0), width)


def settle_pricing(
    records: Iterable[Mapping[str, Any]],
    bars: Sequence[Any],
    *,
    near_strike_band: float = 0.005,
) -> tuple[list[SettledSpread], dict[str, int]]:
    """Resolve every journaled ladder contract whose expiry has passed.

    One observation per ``(expiry, side, delta)``: the *first* time that contract
    was priced, i.e. as offered when it first appeared at ~30 days. A cron at a
    15-minute cadence would otherwise turn twelve contracts into thousands of
    correlated copies of twelve outcomes.
    """
    times, closes = _settlement_index(bars)
    chosen: dict[tuple[str, str, float], Mapping[str, Any]] = {}
    for record in sorted(records, key=observed_at):
        pricing_block = record.get("pricing") or {}
        if pricing_block.get("status") != "ok":
            continue
        for row in pricing_block.get("ladder") or []:
            if row.get("status") != "ok" or not row.get("expiry"):
                continue
            key = (row["expiry"], row["side"], float(row.get("target_delta") or 0.0))
            chosen.setdefault(key, row)

    settled: list[SettledSpread] = []
    accounting = {"pending": 0, "no_price": 0, "unusable": 0}
    now = datetime.now(timezone.utc)
    for (expiry_text, side, delta), row in sorted(chosen.items()):
        try:
            expiry = datetime.fromisoformat(expiry_text)
        except ValueError:
            accounting["unusable"] += 1
            continue
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry > now:
            accounting["pending"] += 1
            continue
        settlement = _settlement_price(times, closes, expiry)
        if settlement is None:
            accounting["no_price"] += 1
            continue

        intrinsic = _intrinsic(row, settlement)
        short = float(row["short_strike"])
        long_ = float(row["long_strike"])
        width = abs(short - long_)
        if side == "put":
            breached = settlement < short
            max_loss_hit = settlement < long_
        else:
            breached = settlement > short
            max_loss_hit = settlement > long_
        near = min(abs(settlement - short), abs(settlement - long_)) / max(short, 1.0) < near_strike_band

        settled.append(
            SettledSpread(
                side=side,
                target_delta=delta,
                expiry=expiry,
                short_strike=short,
                long_strike=long_,
                width=width,
                credit=float(row["credit"]),
                sigma_rv=float(row.get("sigma_rv") or 0.0),
                sigma_window=str(row.get("sigma_window") or ""),
                p_keep=float(row.get("p_keep") or 0.0),
                p_partial=float(row.get("p_partial") or 0.0),
                p_max_loss=float(row.get("p_max_loss") or 0.0),
                settlement=settlement,
                breached=breached,
                max_loss_hit=max_loss_hit,
                pnl=float(row["credit"]) - intrinsic,
                near_strike=near,
            )
        )
    return settled, accounting


def pricing_table(settled: Sequence[SettledSpread]) -> list[dict[str, Any]]:
    """Per side and delta: realised breach rate against the implied one, with lift."""
    from events import wilson_interval

    buckets: dict[tuple[str, float], list[SettledSpread]] = {}
    for row in settled:
        buckets.setdefault((row.side, row.target_delta), []).append(row)

    rows: list[dict[str, Any]] = []
    for (side, delta), group in sorted(buckets.items()):
        n = len(group)
        breaches = sum(1 for row in group if row.breached)
        realised = 100.0 * breaches / n
        implied = 100.0 * sum(1.0 - row.p_keep for row in group) / n
        low, high = wilson_interval(breaches, n)
        rows.append(
            {
                "side": side,
                "delta": delta,
                "n": n,
                "breached": breaches,
                "realised_breach_pct": round(realised, 1),
                "implied_breach_pct": round(implied, 1),
                "lift": round(realised / implied, 2) if implied > 0 else None,
                "low": round(low, 1),
                "high": round(high, 1),
                "verdict": _calibration_verdict(n, low, high, implied),
                "max_loss_hit": sum(1 for row in group if row.max_loss_hit),
                "mean_pnl": round(sum(row.pnl for row in group) / n, 2),
                "near_strike": sum(1 for row in group if row.near_strike),
            }
        )
    return rows


#: Below this many settled contracts, no verdict is available -- the interval is
#: wide enough to contain almost any plausible implied rate, and a ratio computed
#: from a handful of expiries is a rumour with a decimal point.
MIN_SETTLED = 20


def _calibration_verdict(n: int, low: float, high: float, implied: float) -> str:
    """What the interval allows you to say. Nothing, until it excludes 1.00x."""
    if implied <= 0:
        return "no implied rate"
    if n < MIN_SETTLED:
        return "too few observations"
    if low / implied > 1.0:
        return "sells too cheap: breaches exceed the price"
    if high / implied < 1.0:
        return "sells too rich: breaches below the price"
    return "no edge demonstrated"


def format_pricing(
    settled: Sequence[SettledSpread], accounting: Mapping[str, int]
) -> str:
    lines = ["shadow gate, settled at expiry", ""]
    rows = pricing_table(settled)
    if not rows:
        lines.append("  nothing settled yet")
        lines.append(
            f"  pending {accounting.get('pending', 0)}   "
            f"no bar at expiry {accounting.get('no_price', 0)}   "
            f"unusable {accounting.get('unusable', 0)}"
        )
        return "\n".join(lines)

    lines.append(
        f"  {'side':<5}{'delta':>7}{'n':>5}{'breach':>9}{'implied':>9}"
        f"{'lift':>7}{'95% interval':>18}{'pnl':>9}  verdict"
    )
    for row in rows:
        lift = "-" if row["lift"] is None else f"{row['lift']:.2f}"
        lines.append(
            f"  {row['side']:<5}{row['delta']:>7.2f}{row['n']:>5}"
            f"{row['realised_breach_pct']:>8.1f}%{row['implied_breach_pct']:>8.1f}%"
            f"{lift:>7}{f'[{row['low']:.1f}, {row['high']:.1f}]':>18}"
            f"{row['mean_pnl']:>9.2f}  {row['verdict']}"
        )
    lines.append("")
    lines.append(
        f"  pending {accounting.get('pending', 0)}   "
        f"no bar at expiry {accounting.get('no_price', 0)}   "
        f"unusable {accounting.get('unusable', 0)}"
    )
    lines.append(
        "  settlement is the last Binance ETHUSDT close before 08:00 UTC; the\n"
        "  contract settles on Deribit's index, so a result within 0.5% of a\n"
        "  strike is flagged rather than relied on."
    )
    return "\n".join(lines)


#: The bar store the pipeline writes. Anchored to this module, not the working
#: directory -- the same rule ``main.BARS_ROOT`` follows, and for the same reason.
DEFAULT_BARS_ROOT = Path(__file__).resolve().parent.parent / "out" / "bars"


def pricing_report(
    records: Iterable[Mapping[str, Any]], *, bars_root: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Settle the journal's ladder contracts and summarise them by side and delta.

    Returns ``(rows, accounting)``. ``accounting`` counts what could not be
    settled and why, so "n=0" is never mistaken for "no contracts were priced".
    """
    import bars as bar_store

    root = Path(bars_root) if bars_root is not None else DEFAULT_BARS_ROOT
    try:
        stored = bar_store.load_bars(bar_store.store_path(root))
    except Exception as exc:  # a missing store means nothing can be settled
        log.warning("no bar store at %s: %s", root, exc)
        stored = []

    settled, accounting = settle_pricing(records, stored)
    accounting = dict(accounting)
    accounting["bars"] = len(stored)
    return pricing_table(settled), accounting


def _main_pricing(records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> int:
    import bars as bar_store

    root = Path(args.bars_root) if args.bars_root else DEFAULT_BARS_ROOT
    try:
        stored = bar_store.load_bars(bar_store.store_path(root))
    except Exception as exc:
        print(f"no bar store at {root}: {exc}", file=sys.stderr)
        return 1

    settled, accounting = settle_pricing(records, stored)
    accounting["bars"] = len(stored)
    if args.json:
        print(
            json.dumps(
                {
                    "settled": len(settled),
                    "accounting": accounting,
                    "rows": pricing_table(settled),
                },
                indent=2,
            )
        )
    else:
        print(format_pricing(settled, accounting))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score journaled projections.")
    parser.add_argument("path", nargs="?", default=str(journal.DEFAULT_PATH))
    parser.add_argument(
        "--symbol",
        default=None,
        help="score only this symbol (default: every symbol, separately)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="accept an observation up to this fraction of the horizon late",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--pricing",
        action="store_true",
        help="settle the shadow gate's journaled contracts instead of the projections",
    )
    parser.add_argument(
        "--bars-root",
        default=None,
        help="where the bar store lives (default: the pipeline's out/bars)",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    path = Path(args.path)
    if not path.exists():
        print(f"no journal at {path} - run: python src/main.py --journal {path}")
        return 1

    records = journal.load(path)
    if not records:
        # Silence here reads as success. Every line was skipped (wrong schema or
        # corrupt), and the only useful thing to say is which.
        print(
            f"{path} holds no record this build can read "
            f"(journal schema {journal.SCHEMA_VERSION}); "
            "lines written by an older schema are skipped, not guessed at",
            file=sys.stderr,
        )
        return 1

    if args.pricing:
        return _main_pricing(records, args)

    cards = (
        [score_records(records, symbol=args.symbol, tolerance=args.tolerance)]
        if args.symbol
        else score_journal(records, tolerance=args.tolerance)
    )

    if args.json:
        # Always a list, one entry per symbol: a stable shape whether the
        # journal holds one symbol or ten.
        print(json.dumps([card.to_dict() for card in cards], indent=2))
    else:
        print("\n\n".join(format_card(card) for card in cards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
