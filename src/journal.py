"""Append-only journal of analysis runs, so projections can be scored later.

One JSON object per line, never rewritten. Append-only is the point: a change to
the scoring rules cannot silently rewrite what was predicted.

A line is a *record of one run*: when it happened, the spot, the per-timeframe
drift, and every projection the agent published. Scoring (``score.py``) only
reads. Nothing here computes returns -- that is a separate pass, by design.

Write failures are not fatal to the pipeline, but they are visible: the caller
is expected to surface them in the payload.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger("journal")

#: New writes use the current shape. The reader accepts prior versions where
#: the fields a reader needs are intact.
#: v2 added ``drift_split`` and chain provenance. v3 added every EMA rung,
#: versioned shadow weights, and a listed ATM reference for later resolution.
#: v4 adds the ``scenario`` block: the full regime read, the decision, the named
#: failing condition, and the proposed spread, as one historical helper record.
SCHEMA_VERSION = 4
READABLE_SCHEMA_VERSIONS = {1, 2, 3, SCHEMA_VERSION}

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "out" / "projections.jsonl"


def _as_drift(ladder_slope: Any) -> float | None:
    """Ladder slope -> clock-side drift. The negation *is* the sign fix."""
    if ladder_slope is None:
        return None
    return round(-float(ladder_slope), 5)


def _range_record(ranges: Any) -> dict[str, Any]:
    """The measured volatility at the moment of the projection, or its absence.

    Kept flat and small: per timeframe the ATR and the one-sigma move at each
    published horizon, plus the compression rank. ``status`` is carried because
    an unmeasured range must never be averaged together with a measured one.
    """
    if not isinstance(ranges, Mapping):
        return {"status": "unavailable"}
    measured = ranges.get("timeframes") or {}
    return {
        "status": ranges.get("status", "unknown"),
        "compression_percentile": (ranges.get("compression") or {}).get("percentile"),
        "atr_pct": {
            timeframe: (row or {}).get("atr_pct")
            for timeframe, row in measured.items()
        },
        "expected_move_pct": {
            timeframe: (row or {}).get("expected_move_pct")
            for timeframe, row in measured.items()
        },
    }


_LADDER_FIELDS: tuple[str, ...] = (
    "side",
    "target_delta",
    "expiry",
    "dte_actual",
    "short_strike",
    "long_strike",
    "width",
    "credit",
    "credit_mark",
    "credit_source",
    "credit_ratio",
    "breakeven_p",
    "p_keep",
    "p_partial",
    "p_max_loss",
    "iv_short",
    "iv_long",
    "edge",
    "sigma_rv",
    "sigma_window",
    "verdict",
    "reason",
)


def _pricing_record(block: Any) -> dict[str, Any]:
    """The shadow gate, flattened to one row per strike ladder.

    The ladder is the point: each (expiry, side, delta) is a contract whose
    outcome can be resolved from bars once the expiry passes, so this is the
    dataset ``score.py`` calibrates implied breach rates against. ``notes`` and
    the prose are dropped -- a journal line is not a report, and a field that
    no scoring pass reads is a field that will drift.

    Everything is recorded whether or not the gate would trade, because "would
    not trade" is itself an observation (and the one that costs nothing).
    """
    if not isinstance(block, Mapping):
        return {"status": "unavailable"}
    if block.get("status") != "ok":
        return {
            "status": block.get("status", "unavailable"),
            "reason": block.get("reason"),
        }
    sigma = block.get("sigma") or {}
    ladder = [
        {key: row.get(key) for key in _LADDER_FIELDS}
        | {"status": row.get("status", "ok")}
        for row in block.get("ladder") or []
        if isinstance(row, Mapping)
    ]
    return {
        "status": "ok",
        "mode": block.get("mode"),
        "verdict": block.get("verdict"),
        "pick": block.get("pick"),
        "agrees_with_direction": block.get("agrees_with_direction"),
        "sigma": {
            "window": sigma.get("primary"),
            "value": sigma.get("value"),
            "windows": sigma.get("windows"),
        },
        "ladder": ladder,
    }


def record_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a pipeline payload into one journal line.

    Only what a later scoring pass needs is kept: the observation time, the
    spot, the drift that produced each projection, and the projections
    themselves. ``decision`` is metadata for auditing, never a label.

    The ``scenario`` block is the historical helper: the regime read that
    produced the decision, the decision itself, the reason a spread was or was
    not proposed, and the proposed spread. It is the unit of record for "why
    we did / did not trade", and it is written whether or not a trade was
    taken.
    """
    market = payload["market"]
    import weight_model

    weights = weight_model.load_settings()
    shadow_model = weight_model.from_market(market, weights)
    return {
        "schema": SCHEMA_VERSION,
        "collected_at": payload["generated_at"],
        "symbol": payload["symbol"],
        "exchange": payload["exchange"],
        "spot": float(market["spot"]),
        "drift_pct_per_hour": {
            timeframe: row.get("drift_pct_per_hour")
            for timeframe, row in market["timeframes"].items()
        },
        # The ladder split is the leading edge of the same measurement: the
        # front rung turns before the median does. Recorded so a lead/lag
        # question can be asked of the history later (plan.md, H1).
        "drift_split": {
            timeframe: {
                "front": _as_drift((row.get("ladder_slope_pct_per_hour") or {}).get("EMA10")),
                "median": row.get("drift_pct_per_hour"),
                "back": _as_drift((row.get("ladder_slope_pct_per_hour") or {}).get("EMA200")),
            }
            for timeframe, row in market["timeframes"].items()
        },
        # Preserve every EMA rung so later weight experiments do not have to
        # reconstruct a different feature set from the old front/back summary.
        "ema_gradient_pct_per_hour": {
            timeframe: dict(row.get("ladder_slope_pct_per_hour") or {})
            for timeframe, row in market["timeframes"].items()
        },
        "weight_model": shadow_model,
        "atm_reference": payload.get("atm_reference"),
        "chain": payload.get("chain"),
        # Additive to schema 2: the band can only be judged against the volatility
        # that was on offer at the time. ``status`` says whether these numbers
        # were measured or substituted, so a later pass never pools the two.
        "range": _range_record(payload.get("range")),
        # Additive to schema 2: the second opinion on the same chain, with the
        # volatility the edge was measured against and the full strike ladder
        # so each contract can be resolved against bars at its own expiry.
        "pricing": _pricing_record(payload.get("pricing")),
        "agreement": (payload.get("analysis") or {}).get("agreement"),
        "decision": payload.get("decision"),
        # The historical helper: why the regime read resolved the way it did,
        # and what the credit gate said about the resulting spread.
        "scenario": {
            "regime": payload.get("analysis") or {},
            "decision": payload.get("decision"),
            "reason": payload.get("reason"),
            "plan": payload.get("plan"),
        },
        "projections": [
            {
                "horizon_hours": float(item["horizon_hours"]),
                "expected_price": float(item["expected_price"]),
                "low": float(item["low"]),
                "high": float(item["high"]),
                "direction": item.get("direction"),
            }
            for item in payload.get("projection") or []
        ],
    }


def append(path: str | Path, record: Mapping[str, Any], *, fsync: bool = False) -> Path:
    """Append one record as a single line. Returns the path written.

    One short ``O_APPEND`` write per run, so a crash leaves a truncated final
    line rather than a corrupted file, and ``load`` skips whatever it cannot
    parse. Callers should keep exactly one writer (the CLI or a cron job).
    """
    schema = record.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"refusing to journal schema {schema!r}; this build writes {SCHEMA_VERSION}"
        )
    if not record.get("collected_at"):
        raise ValueError("record needs a collected_at timestamp")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)

    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return destination


def load(path: str | Path) -> list[dict[str, Any]]:
    """Read every usable record. A missing file is an empty journal, not an error.

    Unparseable or wrong-schema lines are skipped and logged: losing one night
    of data must not make the rest unscoreable, but it must not be silent
    either.
    """
    source = Path(path)
    if not source.exists():
        return []

    records: list[dict[str, Any]] = []
    skipped = 0
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            if skipped <= 5:
                log.warning("%s:%d: not JSON, skipped", source.name, number)
            continue
        if (
            not isinstance(record, dict)
            or record.get("schema") not in READABLE_SCHEMA_VERSIONS
            or not record.get("collected_at")
            or not isinstance(record.get("symbol"), str)
            or not isinstance(record.get("projections"), list)
        ):
            skipped += 1
            if skipped <= 5:
                log.warning("%s:%d: unsupported schema or invalid record, skipped", source.name, number)
            continue
        records.append(record)

    if skipped:
        log.warning("%s: skipped %d unusable line(s)", source.name, skipped)
    return records
