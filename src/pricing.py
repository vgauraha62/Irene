"""A second opinion on the same chain, asked without the direction signal.

The direction gate answers "which way is price going". Measured over one year
that question has no edge: 44.9% accuracy, lift 0.91x [0.72, 1.14]. A credit
spread does not actually need an answer to it. What a credit spread needs is the
distribution of the settlement price relative to two strikes -- and the option
chain already prices that distribution.

So this module asks the question a seller of premium is paid to answer:

    is the credit larger than the expected payoff, measured against the
    volatility that has actually shown up?

    edge = credit - E[payoff | sigma = trailing realised vol]

Both terms are observable. ``credit`` is bid(short) - ask(long) from the live
ladder, so it is a price a seller receives and not a mid. ``E[payoff]`` is the
same spread marked at a flat sigma taken from the bar store, so it says what the
structure is worth under a vol you can measure rather than one the market
quotes. The difference is the bet.

**This gate never emits a trade.** It is recorded beside the direction gate so
that, once expiries resolve, there is a head-to-head with evidence on both
sides. Promotion is a separate, later decision that requires the calibration in
``score.py`` to exclude a lift of 1.00x. Until then the direction gate remains
the only thing that can produce a candidate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import analysis

log = logging.getLogger("pricing")

#: The delta the shadow trades at. One number, pre-registered: changing it after
#: seeing results is how a backtest turns into a curve fit.
DEFAULT_TARGET_DELTA = 0.30

#: Struck, priced and journaled so the calibration gets three observations per
#: expiry instead of one, without waiting three times as long.
LADDER_DELTAS: tuple[float, ...] = (0.15, 0.30, 0.45)

#: The vol window the edge is measured against, chosen before any result
#: existed. 21 trading days of 15m bars is ~30 calendar days, which is the
#: holding period of the spread it is compared with. The other windows are
#: carried for sensitivity and never enter a decision -- the IV/RV ratio flips
#: sign between 21 days and a year, so the window has to be named, not implied.
PRIMARY_WINDOW = "21d"
VOL_WINDOWS: dict[str, int] = {"2d": 192, "21d": 2016, "1y": 0}  # 0 == all bars

#: A spread whose credit is below this fraction of its width is refused. Kept
#: from the direction gate so the two are judged on the same rule.
DEFAULT_MIN_CREDIT_RATIO = 0.20

_SIDES = {"put": "bull_put_spread", "call": "bear_call_spread"}
_SIDE_OF_DIRECTION = {analysis.BULLISH: "put", analysis.BEARISH: "call"}


def annualised_vol(bars: Sequence[Any], window: str = PRIMARY_WINDOW) -> float | None:
    """Realised vol over ``window``, as a decimal (0.451 == 45.1%).

    Close-to-close log returns on the stored 15m bars, scaled by sqrt(time).
    ``bars.expected_move_pct`` already does that arithmetic against a horizon in
    hours, so an 8760-hour horizon returns exactly the annualised figure and
    there is one implementation of the formula rather than two.
    """
    if not bars:
        return None
    lookback = VOL_WINDOWS.get(window, VOL_WINDOWS[PRIMARY_WINDOW])
    import bars as bar_store

    move = bar_store.expected_move_pct(
        list(bars), 8760.0, lookback=lookback if lookback > 0 else len(bars)
    )
    if move is None:
        return None
    return move / 100.0


def volatility_profile(bars: Sequence[Any]) -> dict[str, Any]:
    """Every window we carry, with the primary one marked."""
    profile: dict[str, Any] = {
        "primary": PRIMARY_WINDOW,
        "bars": len(bars) if bars else 0,
        "windows": {},
    }
    for window in VOL_WINDOWS:
        value = annualised_vol(bars, window)
        profile["windows"][window] = None if value is None else round(value, 4)
    profile["value"] = profile["windows"].get(PRIMARY_WINDOW)
    return profile


@dataclass(frozen=True)
class Candidate:
    """One spread, measured, plus the verdict on it."""

    side: str  # "put" | "call"
    strategy: str
    short_strike: float
    long_strike: float
    width: float
    credit: float
    credit_mark: float
    credit_source: str
    credit_ratio: float
    #: Full credit is kept only outside the short strike; the rest sits between
    #: the strikes and is the band a "POP = 1 - delta" number hides.
    p_keep: float
    p_partial: float
    p_max_loss: float
    breakeven_p: float
    iv_short: float
    iv_long: float
    edge: float
    edge_pct_of_width: float
    sigma_rv: float
    sigma_window: str
    expiry: str | None
    dte_actual: float | None
    contracts: float = 1.0
    verdict: str = "NO TRADE"
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def is_trade(self) -> bool:
        return self.verdict == "TAKE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "strategy": self.strategy,
            "short_strike": round(self.short_strike, 2),
            "long_strike": round(self.long_strike, 2),
            "width": round(self.width, 2),
            "credit": round(self.credit, 4),
            "credit_mark": round(self.credit_mark, 4),
            "credit_source": self.credit_source,
            "credit_ratio": round(self.credit_ratio, 4),
            "breakeven_p": round(self.breakeven_p, 4),
            "p_keep": round(self.p_keep, 4),
            "p_partial": round(self.p_partial, 4),
            "p_max_loss": round(self.p_max_loss, 4),
            "iv_short": round(self.iv_short, 4),
            "iv_long": round(self.iv_long, 4),
            "edge": round(self.edge, 4),
            "edge_pct_of_width": round(self.edge_pct_of_width, 4),
            "sigma_rv": round(self.sigma_rv, 4),
            "sigma_window": self.sigma_window,
            "expiry": self.expiry,
            "dte_actual": None if self.dte_actual is None else round(self.dte_actual, 2),
            "verdict": self.verdict,
            "reason": self.reason,
            "notes": list(self.notes),
        }


def evaluate(
    side: str,
    spot: float,
    chain: Any,
    *,
    sigma_rv: float,
    sigma_window: str = PRIMARY_WINDOW,
    target_delta: float = DEFAULT_TARGET_DELTA,
    dte_days: int = 30,
    min_credit_ratio: float = DEFAULT_MIN_CREDIT_RATIO,
    anchor: str = "delta",
) -> Candidate | None:
    """Price one side and decide whether the credit beats its expected payoff.

    ``spread_plan`` is asked for the structure with the credit gate disabled, so
    a thin spread still yields numbers to journal; the gate is then applied here
    where its verdict can name which condition failed.

    ``anchor="delta"`` keeps the calibration ladder sweeping the short-leg delta;
    the shadow that judges the live candidate passes ``anchor="atm"`` so the
    second opinion prices the same strikes the candidate would sell.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be 'put' or 'call', got {side!r}")
    if sigma_rv <= 0:
        raise ValueError("sigma_rv must be positive")

    direction = analysis.BULLISH if side == "put" else analysis.BEARISH
    try:
        plan, _ = analysis.spread_plan(
            direction,
            spot,
            chain,
            target_delta=target_delta,
            dte_days=dte_days,
            min_credit_ratio=0.0,
            anchor=anchor,
        )
    except Exception as exc:  # a chain that cannot price is "no candidate", not a crash
        log.warning("could not price the %s side: %s", side, exc)
        return None
    if plan is None:
        return None

    years = (plan.dte_actual / 365.0) if plan.dte_actual else dte_days / 365.0
    fair_value = analysis.expected_intrinsic(
        side, spot, plan.short_strike, plan.long_strike, years, sigma_rv
    )
    edge = plan.credit - fair_value

    failed: list[str] = []
    if plan.credit_ratio < min_credit_ratio:
        failed.append(
            f"credit {plan.credit:.2f} is {plan.credit_ratio:.0%} of width "
            f"(floor {min_credit_ratio:.0%})"
        )
    if edge <= 0:
        failed.append(
            f"edge {edge:+.2f} at {sigma_rv:.1%} realised ({sigma_window})"
        )
    verdict = "NO TRADE" if failed else "TAKE"
    reason = "ok" if not failed else "; ".join(failed)

    notes = list(plan.notes)
    notes.append(
        f"edge = credit {plan.credit:.2f} - expected payoff {fair_value:.2f} "
        f"at a flat {sigma_rv:.1%}; the payoff is fat-tailed, so this is the "
        "optimistic end of the estimate"
    )

    return Candidate(
        side=side,
        strategy=plan.strategy,
        short_strike=plan.short_strike,
        long_strike=plan.long_strike,
        width=plan.width,
        credit=plan.credit,
        credit_mark=plan.credit_mark,
        credit_source=plan.credit_source,
        credit_ratio=plan.credit_ratio,
        p_keep=plan.pop,
        p_partial=plan.pop_partial,
        p_max_loss=plan.pop_max_loss,
        breakeven_p=1.0 - plan.credit_ratio,
        iv_short=plan.iv_short,
        iv_long=plan.iv_long,
        edge=edge,
        edge_pct_of_width=100.0 * edge / plan.width if plan.width else 0.0,
        sigma_rv=sigma_rv,
        sigma_window=sigma_window,
        expiry=plan.expiry,
        dte_actual=plan.dte_actual,
        verdict=verdict,
        reason=reason,
        notes=notes,
    )


def ladder(
    side: str,
    spot: float,
    chain: Any,
    *,
    sigma_rv: float,
    sigma_window: str = PRIMARY_WINDOW,
    deltas: Sequence[float] = LADDER_DELTAS,
    dte_days: int = 30,
    min_credit_ratio: float = DEFAULT_MIN_CREDIT_RATIO,
) -> list[dict[str, Any]]:
    """The same side at several deltas, for calibration breadth.

    One expiry per delta is one observation; three deltas is three. They are
    correlated, so this is not three independent trades -- but it is the
    difference between ~100 and ~300 samples a year, and the correlation is
    disclosed rather than corrected away.
    """
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        candidate = evaluate(
            side,
            spot,
            chain,
            sigma_rv=sigma_rv,
            sigma_window=sigma_window,
            target_delta=delta,
            dte_days=dte_days,
            min_credit_ratio=min_credit_ratio,
        )
        if candidate is None:
            rows.append({"side": side, "target_delta": delta, "status": "unavailable"})
            continue
        row = candidate.to_dict()
        row["target_delta"] = delta
        row["status"] = "ok"
        rows.append(row)
    return rows


def shadow(
    spot: float,
    chain: Any,
    *,
    profile: Mapping[str, Any] | None = None,
    bars: Sequence[Any] | None = None,
    dte_days: int = 30,
    target_delta: float = DEFAULT_TARGET_DELTA,
    min_credit_ratio: float = DEFAULT_MIN_CREDIT_RATIO,
    direction_side: str | None = None,
    anchor: str = "atm",
) -> dict[str, Any]:
    """The whole shadow block: both sides, a pick, and whether it agrees.

    ``profile`` is the realised-vol measurement from the bar store (see
    ``volatility_profile``). It is passed in rather than re-read so the shadow
    is unavailable exactly when the range block is -- one measurement, one
    provenance, no second silent read of a different store.

    ``direction_side`` is what the direction gate chose (``"put"``, ``"call"``,
    or ``None`` when it refused). Disagreement is the number worth watching: if
    the two gates rarely agree, one of them is not measuring what it claims.
    """
    profile = dict(profile) if profile else volatility_profile(bars or [])
    block: dict[str, Any] = {
        "mode": "shadow",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "sigma": profile,
        "gate": {
            "target_delta": target_delta,
            "ladder_deltas": list(LADDER_DELTAS),
            "dte_days": dte_days,
            "min_credit_ratio": min_credit_ratio,
            "rule": (
                "edge = credit - E[payoff | trailing realised vol] > 0 "
                "and credit >= 20% of width"
            ),
        },
        "note": (
            "shadow only: measured and journaled beside the direction gate, "
            "never emitted as a trade"
        ),
    }

    sigma_rv = profile.get("value")
    if not sigma_rv:
        block.update(
            {
                "status": "unavailable",
                "reason": "no realised vol from the bar store; without it there is no edge",
                "sides": {},
                "ladder": [],
                "verdict": "NO TRADE",
                "pick": None,
                "agrees_with_direction": None,
            }
        )
        return block

    sides: dict[str, Any] = {}
    ladder_rows: list[dict[str, Any]] = []
    for side in ("put", "call"):
        candidate = evaluate(
            side,
            spot,
            chain,
            sigma_rv=sigma_rv,
            target_delta=target_delta,
            dte_days=dte_days,
            min_credit_ratio=min_credit_ratio,
            anchor=anchor,
        )
        sides[side] = (
            candidate.to_dict()
            if candidate is not None
            else {"side": side, "status": "unavailable", "verdict": "NO TRADE"}
        )
        ladder_rows.extend(
            ladder(
                side,
                spot,
                chain,
                sigma_rv=sigma_rv,
                deltas=LADDER_DELTAS,
                dte_days=dte_days,
                min_credit_ratio=min_credit_ratio,
            )
        )

    tradable = [side for side, row in sides.items() if row.get("verdict") == "TAKE"]
    pick = max(tradable, key=lambda side: sides[side]["edge"]) if tradable else None

    if pick is None:
        failing = {
            side: row.get("reason") or row.get("status", "unavailable")
            for side, row in sides.items()
        }
        reason = "; ".join(f"{side}: {why}" for side, why in failing.items())
    else:
        reason = (
            f"{pick} side carries the larger edge "
            f"({sides[pick]['edge']:+.2f} at {sigma_rv:.1%} realised {PRIMARY_WINDOW})"
        )

    block.update(
        {
            "status": "ok",
            "sides": sides,
            "ladder": ladder_rows,
            "pick": pick,
            "verdict": "TAKE" if pick else "NO TRADE",
            "reason": reason,
            "direction_side": direction_side,
            "agrees_with_direction": (
                None if direction_side is None or pick is None else direction_side == pick
            ),
        }
    )
    return block


def describe(block: Mapping[str, Any]) -> str:
    """One line for the console, so the shadow is never silent."""
    if block.get("status") != "ok":
        return f"pricing  {block.get('status', 'unknown').upper()}  {block.get('reason', '')}"
    sigma = block.get("sigma") or {}
    window = sigma.get("primary", PRIMARY_WINDOW)
    value = sigma.get("value")
    head = f"pricing  {block.get('verdict')}"
    if value is not None:
        head += f"   edge measured at {value:.1%} realised ({window})"
    pick = block.get("pick")
    if pick:
        side = block["sides"][pick]
        head += f"   pick {side['strategy']} {side['short_strike']:.0f}/{side['long_strike']:.0f}"
    agree = block.get("agrees_with_direction")
    if agree is None:
        head += "   (no direction candidate to compare)"
    elif agree:
        head += "   agrees with the direction gate"
    else:
        head += (
            f"   DISAGREES: direction says {block.get('direction_side') or 'no trade'}"
        )
    return head


def load_store(root: str | Path) -> list[Any]:
    """Bars for the vol estimate. Missing store is an absent measurement."""
    import bars as bar_store

    path = bar_store.store_path(root)
    try:
        return bar_store.load_bars(path)
    except FileNotFoundError:
        return []
    except Exception as exc:  # a corrupt store must not sink a run
        log.warning("could not read %s: %s", path, exc)
        return []
