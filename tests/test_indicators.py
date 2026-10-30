"""Tests for the data layer: CSV parsing, import hygiene, plot smoke test."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CSV = ROOT / "ethereum-data" / "Ethereum Historical Data.csv"

needs_csv = pytest.mark.skipif(not CSV.exists(), reason="bundled CSV not present")


@needs_csv
def test_csv_handles_bom_thousands_percent_and_volume_suffix():
    """The real file has a BOM, quoted ',' separators, '%' strings and 'K' volume."""
    indicators = pytest.importorskip("indicators")
    df = indicators.load_csv_history(CSV)

    for column in ("date", "open", "high", "low", "close", "volume", "change_pct"):
        assert column in df.columns, f"missing {column}: {list(df.columns)}"

    assert len(df) > 0
    assert df["date"].is_monotonic_increasing          # sorted ascending
    assert df["date"].dtype.kind == "M"                # real datetimes, not strings
    assert df["close"].dtype.kind == "f"               # numeric, not object

    # "2,314.98" must arrive as 2314.98, not a string or NaN.
    assert df["close"].iloc[0] == pytest.approx(2687.81)
    assert (df["close"] > 0).all()

    # "-0.41%" must arrive as -0.41.
    assert df["change_pct"].dtype.kind == "f"
    assert df["change_pct"].isna().sum() == 0
    assert df["change_pct"].abs().max() < 100

    # "193.65K" -> 193650.
    assert df["volume"].dtype.kind == "f"
    assert df["volume"].dropna().gt(1000).all()


@needs_csv
def test_csv_dates_are_day_first():
    indicators = pytest.importorskip("indicators")
    df = indicators.load_csv_history(CSV)
    # The raw file's newest row is "27-10-2026" -> DD-MM-YYYY, not MM-DD.
    assert df["date"].iloc[-1] == datetime(2026, 10, 27)


@needs_csv
def test_csv_is_too_short_for_ema200():
    """Documents a real limitation instead of quietly emitting a bad EMA200."""
    import analysis

    indicators = pytest.importorskip("indicators")
    df = indicators.load_csv_history(CSV)
    assert not analysis.warmup_sufficient(len(df), 200)
    assert analysis.warmup_sufficient(len(df), 5)


def test_missing_indicator_error_is_one_class():
    """The data layer and the domain layer must share one exception class."""
    import analysis

    indicators = pytest.importorskip("indicators")
    assert indicators.MissingIndicatorError is analysis.MissingIndicatorError


def test_interval_constants_are_real():
    indicators = pytest.importorskip("indicators")
    assert indicators.TIMEFRAMES == {"15m": "15m", "30m": "30m", "4h": "4h"}


def test_naive_scrape_clock_is_not_exchange_time():
    """tradingview_ta sets Analysis.time = datetime.now(); that is not exchange time."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    indicators = pytest.importorskip("indicators")
    fetched = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)

    naive = datetime(2026, 10, 26, 12, 0, 30)  # what the library actually sets
    as_of, source = indicators._resolve_as_of(SimpleNamespace(time=naive), fetched)
    assert source == "fetch-clock"
    assert as_of == fetched

    aware = datetime(2026, 10, 26, 11, 55, tzinfo=timezone.utc)
    as_of, source = indicators._resolve_as_of(SimpleNamespace(time=aware), fetched)
    assert source == "exchange"
    assert as_of == aware

    as_of, source = indicators._resolve_as_of(SimpleNamespace(time=1_793_000_000), fetched)
    assert source == "exchange"
    assert as_of.year == 2026


def test_importing_analysis_does_not_touch_the_network():
    """analysis must stay importable and testable with no data source present."""
    code = (
        "import sys, analysis; "
        "assert 'tradingview_ta' not in sys.modules, 'analysis imported the scraper'; "
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_plot_builds_and_saves_without_a_display(tmp_path):
    """The old plot.py raised NameError and called plt.show() at import."""
    pytest.importorskip("matplotlib")
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"

    import analysis
    import plot as plot_module
    from test_analysis import book

    snaps = book(**{"15m": 0.10, "30m": 0.08, "4h": 0.06})
    reg = analysis.regime(snaps)
    fig = plot_module.build_figure(snaps, reg)

    out = tmp_path / "chart.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000
