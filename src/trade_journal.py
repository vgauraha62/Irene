"""Append-only journal for virtual directional trade observations.

This is not a fill ledger. A start event records the selected side, deterministic
listed strikes, entry reference, expiry, and market features. A close records
whether price moved in the predicted direction from entry to exit. The prices
are used as automatic comparison points, not displayed as trade inputs or saved
as P&L. No premiums, size, fees, or P&L are requested or inferred.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("trade_journal")

SCHEMA_VERSION = 3
READABLE_SCHEMA_VERSIONS = {2, SCHEMA_VERSION}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "out" / "virtual_trades.jsonl"
SIDES = {"bull_put", "bear_call"}


class ObservationError(ValueError):
    """An observation cannot be started or closed as requested."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_events(path: str | Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    events: list[dict[str, Any]] = []
    skipped = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if (
            not isinstance(event, dict)
            or event.get("schema") not in READABLE_SCHEMA_VERSIONS
            or event.get("event") not in {"start", "close", "relabel"}
        ):
            skipped += 1
            continue
        events.append(event)
    if skipped:
        log.warning("%s: skipped %d malformed or unsupported event(s)", source.name, skipped)
    return events


def _append(path: str | Path, event: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def observations(events: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        observation_id = str(event.get("observation_id") or "")
        if not observation_id:
            continue
        if event.get("event") == "start":
            grouped.setdefault(
                observation_id,
                {"start": dict(event), "close": None, "relabel": None},
            )
        elif event.get("event") == "close" and observation_id in grouped:
            if grouped[observation_id]["close"] is None:
                grouped[observation_id]["close"] = dict(event)
        elif event.get("event") == "relabel" and observation_id in grouped:
            # Corrections append a relabel event; the original close time and
            # original classification remain intact for auditability.
            grouped[observation_id]["relabel"] = dict(event)
    return grouped


def active_by_side(events: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return every active observation grouped by strategy side."""
    active: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    for observation in observations(events).values():
        start = observation["start"]
        if observation["close"] is None:
            active.setdefault(start["side"], []).append(start)
    for rows in active.values():
        rows.sort(key=lambda row: row.get("time", ""))
    return active


def start_observation(
    *,
    side: str,
    short_strike: float,
    hedge_strike: float,
    entry_spot: float | None = None,
    expiry: str,
    entry_context: Mapping[str, Any],
    opened_at: str | None = None,
    path: str | Path = DEFAULT_PATH,
) -> dict[str, Any]:
    side = side.strip().lower()
    if side not in SIDES:
        raise ObservationError("side must be bull_put or bear_call")
    short = float(short_strike)
    hedge = float(hedge_strike)
    if not math.isfinite(short) or not math.isfinite(hedge) or short <= 0 or hedge <= 0:
        raise ObservationError("strikes must be positive")
    if side == "bull_put" and hedge >= short:
        raise ObservationError("bull put hedge must be below the short strike")
    if side == "bear_call" and hedge <= short:
        raise ObservationError("bear call hedge must be above the short strike")
    if not expiry:
        raise ObservationError("expiry is required")
    reference = float(entry_spot) if entry_spot is not None else short
    if not math.isfinite(reference) or reference <= 0:
        raise ObservationError("entry reference is unavailable")

    event = {
        "schema": SCHEMA_VERSION,
        "event": "start",
        "observation_id": str(uuid.uuid4()),
        "time": opened_at or _now(),
        "side": side,
        "short_strike": short,
        "hedge_strike": hedge,
        "entry_spot": reference,
        "entry_reference_source": "deribit_underlying" if entry_spot is not None else "legacy_atm_strike",
        "expiry": expiry,
        "entry_context": dict(entry_context),
    }
    _append(path, event)
    return event


def close_observation(
    *,
    side: str,
    exit_spot: float,
    closed_at: str | None = None,
    path: str | Path = DEFAULT_PATH,
) -> dict[str, Any]:
    """Close the oldest active observation for a side; persist directional outcome only.

    Bull put is a bullish prediction and succeeds when exit spot is above entry
    spot. Bear call is a bearish prediction and succeeds when exit spot is below
    entry spot. Exact ties are not directional wins. Exit spot is not persisted.
    """
    side = side.strip().lower()
    if side not in SIDES:
        raise ObservationError("side must be bull_put or bear_call")
    try:
        spot = float(exit_spot)
    except (TypeError, ValueError):
        raise ObservationError("current spot is unavailable") from None
    if not math.isfinite(spot) or spot <= 0:
        raise ObservationError("current spot is unavailable")
    active = active_by_side(load_events(path)).get(side, [])
    if not active:
        raise ObservationError(f"no open {side.replace('_', ' ')} observation")
    # Each start is a separate virtual position. A side-level Exit click closes
    # the oldest one, so repeated clicks can close a stack one observation at a time.
    start = active[0]
    reference = float(start.get("entry_spot") or start["short_strike"])
    success = spot > reference if side == "bull_put" else spot < reference
    event = {
        "schema": SCHEMA_VERSION,
        "event": "close",
        "observation_id": start["observation_id"],
        "time": closed_at or _now(),
        "success": success,
        "outcome_rule": "direction_from_entry_spot",
    }
    _append(path, event)
    return event


def summary(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = observations(events)
    opened: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    for observation_id, pair in grouped.items():
        start = pair["start"]
        exit_event = pair["close"]
        if exit_event is None:
            opened.append({**start, "observation_id": observation_id})
            continue
        classification = pair.get("relabel") or exit_event
        closed.append(
            {
                **start,
                "observation_id": observation_id,
                "closed_at": exit_event.get("time", ""),
                "success": bool(classification.get("success")),
                "outcome": (
                    "DIRECTIONAL SUCCESS"
                    if classification.get("success")
                    else "DIRECTIONAL FAILURE"
                ),
            }
        )
    opened.sort(key=lambda row: row.get("time", ""))
    closed.sort(key=lambda row: row.get("closed_at", ""))
    rates: dict[str, dict[str, Any]] = {}
    for side in sorted(SIDES):
        rows = [row for row in closed if row["side"] == side]
        successes = sum(row["success"] for row in rows)
        rates[side] = {
            "closed": len(rows),
            "successes": successes,
            "failures": len(rows) - successes,
            "success_rate_pct": 100.0 * successes / len(rows) if rows else None,
        }
    successes = sum(row["success"] for row in closed)
    return {
        "open": opened,
        "closed": closed,
        "total_closed": len(closed),
        "successes": successes,
        "failures": len(closed) - successes,
        "success_rate_pct": 100.0 * successes / len(closed) if closed else None,
        "by_side": rates,
    }
