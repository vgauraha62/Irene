"""The chart gains a panel only when a range was actually measured.

The rule is the same one the payload follows: an unmeasured quantity is absent,
never drawn as a zero. A flat line at 0% width would read as "maximum
compression", which is the exact opposite of "we did not measure".
"""

from __future__ import annotations

import analysis
import plot
from test_analysis import book

BULLISH = {"15m": 0.10, "30m": 0.08, "4h": 0.06}


def _figure(ranges=None):
    snapshots = book(adx=30.0, **BULLISH)
    regime = analysis.regime(snapshots)
    return plot.build_figure(snapshots, regime, ranges=ranges)


MEASURED = {
    "compression": {
        "percentile": 8.0,
        "reads": "very compressed",
        "series": [0.62, 0.58, 0.51, 0.47, 0.44, 0.40],
        "p10": 0.41,
        "p90": 0.95,
    }
}


def test_without_a_range_the_chart_keeps_its_two_panels():
    figure = _figure()

    assert len(figure.axes) == 2
    plot.plt.close(figure)


def test_a_measured_width_adds_the_compression_panel():
    figure = _figure(MEASURED)

    assert len(figure.axes) == 3
    assert "compression" in figure.axes[2].get_title().lower()
    plot.plt.close(figure)


def test_the_panel_marks_the_current_value_and_its_band():
    figure = _figure(MEASURED)
    panel = figure.axes[2]

    # One line for the series; the reference band is drawn with axhspan, which
    # is a collection rather than a line.
    assert panel.collections, "the 10-90th percentile band must be shaded"
    annotations = [text.get_text() for text in panel.texts]
    assert any("8th pct" in text for text in annotations)
    plot.plt.close(figure)


def test_an_empty_series_does_not_draw_an_empty_panel():
    figure = _figure({"compression": {"series": [], "percentile": None}})

    assert len(figure.axes) == 2
    plot.plt.close(figure)


def test_render_writes_a_file_with_the_panel(tmp_path):
    snapshots = book(adx=30.0, **BULLISH)
    regime = analysis.regime(snapshots)
    destination = tmp_path / "chart.png"

    written = plot.render(snapshots, regime, path=destination, ranges=MEASURED)

    assert written == destination
    assert destination.stat().st_size > 1000
