"""Versioned shadow weights for the EMA-gradient direction experiment.

This model is diagnostic only. It never changes the production regime or trade
decision. Each signal stores the weights and feature values that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import analysis

TIMEFRAMES = ("15m", "30m", "4h")
GROUPS = {
    "fast": ("EMA10", "EMA20"),
    "mid": ("EMA30", "EMA50"),
    "slow": ("EMA100", "EMA200"),
}
DEFAULT_WEIGHTS = {
    timeframe: {group: 1.0 for group in GROUPS}
    for timeframe in TIMEFRAMES
}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "out" / "weight_settings.json"
MODEL_VERSION = "ema-gradient-groups-v1"


def clean_weights(weights: Mapping[str, Any] | None = None) -> dict[str, dict[str, float]]:
    source = weights or DEFAULT_WEIGHTS
    cleaned: dict[str, dict[str, float]] = {}
    for timeframe in TIMEFRAMES:
        row = source.get(timeframe, {}) if isinstance(source, Mapping) else {}
        cleaned[timeframe] = {}
        for group in GROUPS:
            try:
                value = float(row.get(group, DEFAULT_WEIGHTS[timeframe][group]))
            except (TypeError, ValueError, AttributeError):
                value = DEFAULT_WEIGHTS[timeframe][group]
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                raise ValueError(f"weight {timeframe}.{group} must be between 0 and 2")
            cleaned[timeframe][group] = round(value, 4)
    if not any(value > 0 for row in cleaned.values() for value in row.values()):
        raise ValueError("at least one weight must be above zero")
    return cleaned


def version(weights: Mapping[str, Any]) -> str:
    canonical = json.dumps(clean_weights(weights), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{MODEL_VERSION}:{canonical}".encode()).hexdigest()[:12]


def load_settings(path: str | Path = DEFAULT_PATH) -> dict[str, dict[str, float]]:
    source = Path(path)
    if not source.exists():
        return clean_weights()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return clean_weights(payload.get("weights", payload))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read weight settings at {source}: {exc}") from None


def save_settings(
    weights: Mapping[str, Any], path: str | Path = DEFAULT_PATH
) -> dict[str, Any]:
    cleaned = clean_weights(weights)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_VERSION,
        "version": version(cleaned),
        "weights": cleaned,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return payload


def features_from_market(market: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Grouped clock-side drift features, in %/hour, from all six EMA rungs."""
    result: dict[str, dict[str, float]] = {}
    for timeframe in TIMEFRAMES:
        row = (market.get("timeframes") or {}).get(timeframe) or {}
        slopes = row.get("ladder_slope_pct_per_hour") or {}
        result[timeframe] = {}
        for group, keys in GROUPS.items():
            present = [float(slopes[key]) for key in keys if slopes.get(key) is not None]
            # Raw ladder slope runs opposite to clock-side price direction.
            result[timeframe][group] = -sum(present) / len(present) if present else 0.0
    return result


def predict(
    features: Mapping[str, Any], weights: Mapping[str, Any]
) -> dict[str, Any]:
    cleaned = clean_weights(weights)
    numerator = 0.0
    denominator = 0.0
    used: dict[str, dict[str, float]] = {}
    for timeframe in TIMEFRAMES:
        used[timeframe] = {}
        row = features.get(timeframe, {}) if isinstance(features, Mapping) else {}
        for group in GROUPS:
            weight = cleaned[timeframe][group]
            value = float(row.get(group, 0.0))
            used[timeframe][group] = round(value, 6)
            numerator += weight * value
            denominator += weight
    score = numerator / denominator if denominator else 0.0
    signal = "bullish" if score > analysis.DEFAULT_EPSILON else (
        "bearish" if score < -analysis.DEFAULT_EPSILON else "neutral"
    )
    return {
        "model": MODEL_VERSION,
        "weights_version": version(cleaned),
        "weights": cleaned,
        "features_pct_per_hour": used,
        "score_pct_per_hour": round(score, 6),
        "flat_band_pct_per_hour": analysis.DEFAULT_EPSILON,
        "signal": signal,
        "horizon_hours": 4.0,
    }


def from_market(
    market: Mapping[str, Any], weights: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    chosen = clean_weights(weights)
    return predict(features_from_market(market), chosen)
