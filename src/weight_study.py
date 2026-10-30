"""Resolve versioned EMA-weight signals against later listed-ATM outcomes."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

import bars
import events


def evaluate(
    records: Iterable[Mapping[str, Any]],
    candles: Sequence[bars.Bar],
    *,
    tolerance_minutes: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score one non-overlapping four-hour sample at a time.

    A sample is included only when its journal line froze a live Deribit ATM
    strike. Skipping overlapping horizons avoids presenting 16 near-identical
    15-minute predictions as 16 independent results.
    """
    ordered_records = sorted(
        (row for row in records if row.get("weight_model")),
        key=lambda row: _timestamp(row.get("collected_at")),
    )
    ordered_bars = sorted(candles, key=lambda row: row.open_time)
    close_times = [row.open_time + timedelta(minutes=15) for row in ordered_bars]
    accounting = {"eligible": 0, "pending": 0, "overlapping": 0, "unusable": 0}
    samples: list[dict[str, Any]] = []
    next_allowed: datetime | None = None
    tolerance = timedelta(minutes=tolerance_minutes)

    for record in ordered_records:
        model = record.get("weight_model") or {}
        reference = record.get("atm_reference") or {}
        if reference.get("status") != "ok" or reference.get("source") != "deribit":
            accounting["unusable"] += 1
            continue
        start = _timestamp(record.get("collected_at"))
        horizon = float(model.get("horizon_hours") or 4.0)
        target_time = start + timedelta(hours=horizon)
        if next_allowed is not None and start < next_allowed:
            accounting["overlapping"] += 1
            continue
        accounting["eligible"] += 1
        next_allowed = target_time

        bar_index = bisect_left(close_times, target_time)
        if bar_index >= len(ordered_bars):
            accounting["pending"] += 1
            continue
        realised_at = close_times[bar_index]
        if realised_at - target_time > tolerance:
            accounting["pending"] += 1
            continue

        signal = str(model.get("signal") or "neutral")
        strike = float(reference["strike"])
        start_spot = float(record["spot"])
        realised = float(ordered_bars[bar_index].close)
        actual = "above" if realised > strike else "below" if realised < strike else "at"
        correct = (
            signal == "bullish" and actual == "above"
        ) or (signal == "bearish" and actual == "below")
        expected_move = _expected_move_pct(record, horizon)
        realised_move_pct = 100.0 * abs(realised / start_spot - 1.0) if start_spot else None
        samples.append(
            {
                "version": model.get("weights_version", "unknown"),
                "weights": model.get("weights") or {},
                "signal": signal,
                "actual": actual,
                "correct": bool(correct),
                "strike": strike,
                "start_spot": start_spot,
                "realised_move_pct": realised_move_pct,
                "expected_move_pct": expected_move,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample["version"]), []).append(sample)

    rows: list[dict[str, Any]] = []
    for version, group in groups.items():
        directional = [row for row in group if row["signal"] in {"bullish", "bearish"} and row["actual"] != "at"]
        hits = sum(bool(row["correct"]) for row in directional)
        above = sum(row["actual"] == "above" for row in group)
        below = sum(row["actual"] == "below" for row in group)
        base = max(above, below) / (above + below) * 100 if above + below else None
        low, high = events.wilson_interval(hits, len(directional))
        expected = [row for row in group if row["expected_move_pct"] is not None and row["realised_move_pct"] is not None]
        vol_mae = (
            sum(abs(row["expected_move_pct"] - row["realised_move_pct"]) for row in expected) / len(expected)
            if expected else None
        )
        weights = group[0]["weights"]
        rows.append(
            {
                "weights version": version,
                "samples": len(group),
                "active predictions": len(directional),
                "coverage %": round(100 * len(directional) / len(group), 1) if group else 0.0,
                "hit rate %": round(100 * hits / len(directional), 1) if directional else None,
                "95% low %": round(low, 1) if directional else None,
                "95% high %": round(high, 1) if directional else None,
                "majority base %": round(base, 1) if base is not None else None,
                "lift": round((100 * hits / len(directional)) / base, 2) if directional and base else None,
                "4h vol MAE %": round(vol_mae, 3) if vol_mae is not None else None,
                "settings": _settings_label(weights),
            }
        )
    rows.sort(key=lambda row: (row["lift"] is not None, row["lift"] or -1), reverse=True)
    return rows, accounting


def evaluate_closed_observations(
    events_in: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report manual virtual entries separately, labelled at their Exit click."""
    from trade_journal import observations

    samples: list[dict[str, Any]] = []
    for pair in observations(events_in).values():
        start, close = pair["start"], pair["close"]
        if close is None:
            continue
        model = (start.get("entry_context") or {}).get("weight_model") or {}
        if not model:
            continue
        success = bool(close.get("success"))
        if start["side"] == "bull_put":
            actual = "above" if success else "below"
        else:
            actual = "below" if success else "above"
        signal = str(model.get("signal") or "neutral")
        samples.append(
            {
                "version": model.get("weights_version", "unknown"),
                "weights": model.get("weights") or {},
                "signal": signal,
                "actual": actual,
                "correct": signal == ("bullish" if actual == "above" else "bearish"),
            }
        )
    return _summarize(samples)


def _summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample["version"]), []).append(sample)
    rows: list[dict[str, Any]] = []
    for version, group in groups.items():
        directional = [row for row in group if row["signal"] in {"bullish", "bearish"} and row["actual"] != "at"]
        hits = sum(bool(row["correct"]) for row in directional)
        above = sum(row["actual"] == "above" for row in group)
        below = sum(row["actual"] == "below" for row in group)
        base = max(above, below) / (above + below) * 100 if above + below else None
        low, high = events.wilson_interval(hits, len(directional))
        weights = group[0]["weights"]
        rows.append(
            {
                "weights version": version,
                "samples": len(group),
                "active predictions": len(directional),
                "coverage %": round(100 * len(directional) / len(group), 1) if group else 0.0,
                "hit rate %": round(100 * hits / len(directional), 1) if directional else None,
                "95% low %": round(low, 1) if directional else None,
                "95% high %": round(high, 1) if directional else None,
                "majority base %": round(base, 1) if base is not None else None,
                "lift": round((100 * hits / len(directional)) / base, 2) if directional and base else None,
                "settings": _settings_label(weights),
            }
        )
    rows.sort(key=lambda row: (row["lift"] is not None, row["lift"] or -1), reverse=True)
    return rows


def _timestamp(value: Any) -> datetime:
    stamp = datetime.fromisoformat(str(value))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _expected_move_pct(record: Mapping[str, Any], horizon: float) -> float | None:
    if abs(horizon - 4.0) > 1e-6:
        return None
    timeframes = (record.get("range") or {}).get("expected_move_pct") or {}
    value = (timeframes.get("15m") or {}).get("4h")
    return None if value is None else float(value)


def _settings_label(weights: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for timeframe in ("15m", "30m", "4h"):
        row = weights.get(timeframe) or {}
        parts.append(
            f"{timeframe} F{row.get('fast', '—')} M{row.get('mid', '—')} S{row.get('slow', '—')}"
        )
    return " · ".join(parts)
