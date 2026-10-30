"""Chart the multi-scale EMA term structure and its slope.

Import-safe and headless-safe: the backend is forced to ``Agg`` before pyplot is
imported, figure construction is a function, and the image is written with
``savefig``. Nothing runs at import time and nothing calls ``plt.show()`` unless
you explicitly ask for it with ``--show``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analysis  # noqa: E402

COLORS = {"15m": "#FFA500", "30m": "#26a69a", "4h": "#2962ff"}
LABELS = {
    "15m": "15m EMA curve (micro-flow)",
    "30m": "30m EMA curve (micro-flow)",
    "4h": "4h EMA curve (macro-inertia)",
}
NEUTRAL_COLOR = "#ef5350"

#: Two panels, same as before: the shape, then its slope.
FIGURE_SIZE = (16, 6)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "out"


def _colors(regime: analysis.Regime) -> tuple[str, str]:
    macro = COLORS["30m"] if regime.macro_dir == analysis.MACRO_BULL else NEUTRAL_COLOR
    if regime.macro_dir == analysis.MACRO_FLAT:
        macro = "#787b86"
    micro = COLORS["30m"] if regime.micro_shift == analysis.MICRO_BULL else NEUTRAL_COLOR
    if regime.micro_shift == analysis.MICRO_FLAT:
        micro = "#787b86"
    return micro, macro


def build_figure(
    snapshots: Mapping[str, Any],
    regime: analysis.Regime,
    *,
    title: str | None = None,
    ranges: Mapping[str, Any] | None = None,
):
    """Render the panels and return the Figure (no I/O).

    The third panel appears only when the run actually measured a range: an
    unmeasured width draws nothing rather than drawing a zero, for the same
    reason the payload prints ``UNMEASURED`` instead of ``0``.
    """
    series = ((ranges or {}).get("compression") or {}).get("series") or []
    panels = 3 if series else 2
    fig, axes = plt.subplots(
        1,
        panels,
        figsize=(FIGURE_SIZE[0] * panels / 2, FIGURE_SIZE[1]),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]} if panels == 3 else None,
    )

    # Numeric x, with EMA periods as tick labels. The old categorical x broke
    # `fill_between`, which needs real coordinates.
    x = np.arange(len(analysis.MA_KEYS))
    ladders = {tf: analysis.ema_values(snap) for tf, snap in snapshots.items()}
    price = analysis.spot(snapshots["15m"])

    ax = axes[0]
    for tf in ("15m", "30m", "4h"):
        if tf not in ladders:
            continue
        ax.plot(x, ladders[tf], "o-", color=COLORS[tf], label=LABELS[tf])

    ax.axhline(
        y=price,
        color=NEUTRAL_COLOR,
        linestyle="--",
        label=f"Spot ({price:,.2f})",
    )
    if "15m" in ladders and "4h" in ladders:
        ax.fill_between(
            x,
            ladders["15m"],
            ladders["4h"],
            color="gray",
            alpha=0.15,
            label="Cross-scale disequilibrium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(analysis.MA_KEYS)
    ax.set_title("EMA term structure: micro vs macro curves", fontsize=14)
    ax.set_xlabel("EMA horizon")
    ax.set_ylabel("Asset price (USD)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)

    # Slope panel: drift in %/hour, so 15m and 4h share one scale and the sign
    # matches the tables. `slopes` is the lookback-side derivative; a rising
    # market makes it negative, so it is negated here.
    ax = axes[1]
    width = 0.26
    offsets = {"15m": -width, "30m": 0.0, "4h": width}
    drift_arrays: list[np.ndarray] = []
    for tf in ("15m", "30m", "4h"):
        if tf not in snapshots:
            continue
        drift = -analysis.slopes(snapshots[tf])
        drift_arrays.append(drift)
        ax.bar(
            x + offsets[tf],
            drift,
            width=width,
            color=COLORS[tf],
            label=f"{tf} drift ({np.median(drift):+.3f} %/h median)",
        )
    ax.axhline(0, color="black", linewidth=1)

    # Reserve headroom so the annotations never sit on top of the bars.
    if drift_arrays:
        flat = np.concatenate(drift_arrays)
        low = min(0.0, float(flat.min()))
        high = max(0.0, float(flat.max()))
        span = max(high - low, 1e-9)
        ax.set_ylim(low - 0.1 * span, high + 0.5 * span)

    micro_color, macro_color = _colors(regime)
    ax.text(
        0.02,
        0.97,
        f"Micro shift: {regime.micro_shift}",
        transform=ax.transAxes,
        color=micro_color,
        fontweight="bold",
        fontsize=11,
        va="top",
    )
    ax.text(
        0.02,
        0.87,
        f"Macro trend: {regime.macro_dir}",
        transform=ax.transAxes,
        color=macro_color,
        fontweight="bold",
        fontsize=11,
        va="top",
    )
    ax.text(
        0.02,
        0.77,
        f"Direction: {regime.direction.upper()}  |  confidence {regime.confidence:.0%}",
        transform=ax.transAxes,
        color="#131722",
        fontsize=10,
        va="top",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(analysis.MA_KEYS)
    ax.set_title("Drift across EMA scale (positive = rising)", fontsize=14)
    ax.set_xlabel("EMA horizon")
    ax.set_ylabel("Drift (% of spot per hour)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)

    if panels == 3:
        _compression_panel(axes[2], ranges or {})

    headline = title or "ETHUSD multi-scale dynamical analysis"
    fig.suptitle(
        f"{headline}\n{regime.micro_shift} within {regime.macro_dir}"
        f"  |  as of {_stamp(regime.as_of)}",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def _compression_panel(ax, ranges: Mapping[str, Any]) -> None:
    """Keltner width over the last bars, against its own past distribution.

    A single percentile answers "is this tight?"; the line answers "is it getting
    tighter or has it already turned?", which is the question the planner has to
    act on. Deliberately drawn without an axis of price: this is a volatility
    panel, not a price forecast.
    """
    compression = ranges.get("compression") or {}
    values = list(compression.get("series") or [])
    if not values:
        return

    x = np.arange(len(values))
    ax.plot(x, values, color="#2962ff", linewidth=1.4)

    p10, p90 = compression.get("p10"), compression.get("p90")
    if p10 is not None and p90 is not None:
        ax.axhspan(p10, p90, color="#26a69a", alpha=0.12)
        ax.axhline(p10, color="#26a69a", linestyle="--", linewidth=0.8)
        ax.axhline(p90, color="#26a69a", linestyle="--", linewidth=0.8)
        ax.text(
            0,
            p90,
            f"90th {p90:.2f}%  ",
            color="#26a69a",
            fontsize=8,
            va="bottom",
            ha="left",
        )
        ax.text(
            0,
            p10,
            f"10th {p10:.2f}%  ",
            color="#26a69a",
            fontsize=8,
            va="top",
            ha="left",
        )

    rank = compression.get("percentile")
    read = compression.get("reads") or "unranked"
    label = f"now {values[-1]:.2f}%" + (
        f" · {rank:.0f}th pct · {read}" if rank is not None else ""
    )
    ax.scatter([len(values) - 1], [values[-1]], color="#ef5350", zorder=5, s=36)
    ax.annotate(
        label,
        xy=(len(values) - 1, values[-1]),
        xytext=(-8, 14),
        textcoords="offset points",
        color="#ef5350",
        fontsize=9,
        ha="right",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#ef5350", "alpha": 0.85},
    )

    ax.set_title("Keltner width — compression (low = coiled)", fontsize=14)
    ax.set_xlabel("last 15m bars (oldest left, newest right)")
    ax.set_ylabel("width (% of spot)")
    ax.grid(alpha=0.2)


def _stamp(value: str) -> str:
    """Trim an ISO timestamp to minutes; microseconds are noise on a chart."""
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return str(value)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _safe_stamp(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", value)[:15] or "unknown"


def save_figure(fig, destination: Path) -> Path:
    """Write the figure to disk, creating parents as needed."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return destination


def render(
    snapshots: Mapping[str, Any],
    regime: analysis.Regime,
    *,
    path: Path | str | None = None,
    title: str | None = None,
    ranges: Mapping[str, Any] | None = None,
) -> Path:
    """Build and save the chart, returning the written path.

    Callers own data acquisition (``main.py``); this module only draws. ``ranges``
    is the payload's ``range`` block: when it carries a measured width series the
    figure gains a compression panel, and when it does not the figure does not
    pretend otherwise.
    """
    fig = build_figure(snapshots, regime, title=title, ranges=ranges)
    if path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"chart_{_safe_stamp(regime.as_of)}.png"
    return save_figure(fig, Path(path))
