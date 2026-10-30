"""Tests for the orchestrator: one fetch, guard, evaluate, render, exit codes.

Every test stubs ``indicators.call_any_analysis``, so none of these touch the
network.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

import analysis
import bars
import indicators
import main
from test_analysis import Fake, book, ladder


def fresh_book(rates: dict[str, float], adx: float = 30.0) -> dict:
    """A snapshot book dated *now*, so the staleness guard passes."""
    snaps = book(adx=adx, **rates)
    now = datetime.now(timezone.utc)
    for snap in snaps.values():
        snap.as_of = now
        snap.as_of_source = "exchange"
    return snaps


@pytest.fixture
def stub_feed(monkeypatch):
    """Install a fake feed and return a setter for the next payload."""

    def install(snaps):
        monkeypatch.setattr(main.indicators, "call_any_analysis", lambda **kwargs: snaps)
        return snaps

    return install


BULLISH_RATES = {"15m": 0.10, "30m": 0.08, "4h": 0.06}
BEARISH_RATES = {"15m": -0.10, "30m": -0.08, "4h": -0.06}
MIXED_RATES = {"15m": -0.10, "30m": 0.08, "4h": 0.06}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_run_builds_a_complete_payload(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(
        main.build_parser().parse_args(["--no-chart", "--equity", "1000000"])
    )

    assert payload["symbol"] == "ETHUSD"
    assert payload["market"]["spot"] == 2300.0
    assert set(payload["market"]["timeframes"]) == {"15m", "30m", "4h"}
    assert payload["freshness"]["status"] == "verified"
    assert payload["analysis"]["direction"] == analysis.BULLISH
    assert payload["decision"] == "CANDIDATE"
    assert payload["plan"]["strategy"] == "bull_put_spread"
    assert payload["plan"]["contracts"] > 0
    assert payload["disclaimer"]


def test_payload_carries_drift_and_the_full_ladder_shape(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    # Clock-side trend rate: bullish rates must read positive, not inverted.
    for timeframe in ("15m", "30m", "4h"):
        assert payload["market"]["timeframes"][timeframe]["drift_pct_per_hour"] > 0
    assert payload["analysis"]["drift_pct_per_hour"]["4h"] == pytest.approx(0.06, abs=1e-3)

    # The whole ladder, per rung, so the shape is not collapsed to one median.
    rungs = payload["market"]["timeframes"]["4h"]["ladder_slope_pct_per_hour"]
    assert set(rungs) == set(analysis.MA_KEYS)
    assert all(value < 0 for value in rungs.values())  # rising => negative ladder slope


def test_payload_projects_the_expected_price(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    projections = payload["projection"]
    assert [item["horizon"] for item in projections] == ["30m", "4h"]
    for item in projections:
        assert item["low"] < item["expected_price"] < item["high"]
        assert item["expected_price"] > payload["market"]["spot"]
        assert item["invalidates"]
        assert any("never scored" in note for note in item["notes"])


def test_bearish_payload_projects_a_lower_price(stub_feed):
    stub_feed(fresh_book(BEARISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    assert payload["analysis"]["direction"] == analysis.BEARISH
    assert all(
        item["expected_price"] < payload["market"]["spot"] for item in payload["projection"]
    )


def test_run_declines_when_timeframes_disagree(stub_feed):
    stub_feed(fresh_book(MIXED_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    assert payload["decision"] == "NO TRADE"
    assert payload["plan"] is None
    assert "neutral" in payload["reason"]
    assert payload["analysis"]["agreement"] == 2


def test_bearish_regime_plans_a_bear_call(stub_feed):
    stub_feed(fresh_book(BEARISH_RATES))
    payload = main.run(
        main.build_parser().parse_args(["--no-chart", "--equity", "1000000"])
    )
    assert payload["decision"] == "CANDIDATE"
    assert payload["plan"]["strategy"] == "bear_call_spread"
    assert payload["plan"]["long_strike"] > payload["plan"]["short_strike"]


def test_adx_chop_blocks_the_trade(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES, adx=11.0))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))
    assert payload["decision"] == "NO TRADE"
    assert payload["analysis"]["adx"] == 11.0


def test_equity_is_optional(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))
    assert payload["plan"] is not None
    assert payload["plan"]["contracts"] == 0
    assert any("no account equity" in note for note in payload["plan"]["notes"])


# --------------------------------------------------------------------------- #
# Range measured from real bars
# --------------------------------------------------------------------------- #


def synthetic_rows(count: int = 700, *, interval_minutes: int = 15) -> list[list]:
    """A deterministic, gently trending series of Binance-shaped kline rows.

    Deliberately dull: the point is that the local EMAs and the feed's agree
    exactly, not that the market does anything interesting.
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 2000.0
    rows: list[list] = []
    for index in range(count):
        previous = price
        price = round(price * (1.0 + 0.0012 * math.sin(index / 9.0) + 0.0002), 2)
        moment = start + timedelta(minutes=interval_minutes * index)
        high = round(max(previous, price) * 1.0008, 2)
        low = round(min(previous, price) * 0.9992, 2)
        volume = 100.0 + index % 7
        rows.append([int(moment.timestamp() * 1000), str(previous), str(high), str(low), str(price), str(volume)])
    return rows


def feed_from_bars(candles: list[bars.Bar]) -> dict[str, Fake]:
    """Snapshots whose ladders are computed from the same candles.

    Where a period has too few bars to warm up (EMA200 on a 43-bar 4h series)
    the value is copied from the shortest ladder rather than left None: the
    cross-check skips it, but ``_market`` still needs a number to publish.
    """
    series = {
        "15m": candles,
        "30m": bars.resample(candles, 2, "15m"),
        "4h": bars.resample(candles, 16, "15m"),
    }
    out: dict[str, Fake] = {}
    for timeframe, rows in series.items():
        closes = [bar.close for bar in rows]
        values: dict[str, float] = {}
        for key in analysis.MA_KEYS:
            value = bars.ema_last(closes, int(key.removeprefix("EMA")))
            values[key] = closes[-1] if value is None else value
        values["close"] = closes[-1]
        values["ADX"] = 30.0
        out[timeframe] = Fake(
            timeframe=timeframe, indicators=values, as_of=datetime.now(timezone.utc)
        )
    return out


@pytest.fixture
def stubbed_bars(monkeypatch):
    rows = synthetic_rows()
    monkeypatch.setattr(bars, "_get", lambda *args, **kwargs: rows)
    return bars.parse_klines(rows)


def test_the_bar_store_never_resolves_against_the_working_directory():
    """Guards a real regression: a relative default created a second store.

    The dashboard launched from ``src/`` silently built ``src/out/bars`` with ten
    days of bars while the backfilled year sat in ``out/bars`` untouched. A
    default path is captured at import, so the check is on the signature, not on
    a call.
    """
    import inspect
    from pathlib import Path

    assert main.BARS_ROOT.is_absolute()
    default = inspect.signature(main.bars_block).parameters["root"].default
    assert Path(default).is_absolute()


def test_range_is_unmeasured_without_a_bar_feed(stub_feed, tmp_path):
    """The autouse fixture kills the fetch: the run must survive and admit it."""
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.analyse(use_bars=True, bars_root=tmp_path)

    assert payload["range"]["status"] == "unavailable"
    assert "offline" in payload["range"]["reason"]
    assert all(row["atr_pct"] is None for row in payload["market"]["timeframes"].values())
    assert "unmeasured" in main.summarize(payload).lower()
    assert payload["decision"] == "CANDIDATE"  # the read still happened


def test_no_bars_says_so_rather_than_looking_measured(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart", "--no-bars"]))

    assert payload["range"]["status"] == "unavailable"
    assert "--no-bars" in payload["range"]["reason"]


def test_range_is_measured_from_real_bars(stub_feed, stubbed_bars, tmp_path):
    stub_feed(feed_from_bars(stubbed_bars))
    payload = main.analyse(use_bars=True, bars_root=tmp_path)

    ranges = payload["range"]
    assert ranges["status"] == "ok"
    assert ranges["appended"] == len(stubbed_bars)
    assert ranges["stored"] == len(stubbed_bars)

    measured = ranges["timeframes"]["15m"]
    assert measured["bars"] == len(stubbed_bars)
    assert measured["atr_pct"] > 0
    assert measured["expected_move_pct"]["4h"] > measured["expected_move_pct"]["30m"]

    # The measurement has to reach the table the reader looks at.
    assert payload["market"]["timeframes"]["15m"]["atr_pct"] == measured["atr_pct"]
    text = main.summarize(payload)
    assert "ATR" in text and "1-sigma" in text


def test_the_bar_store_is_shared_between_runs(stub_feed, stubbed_bars, tmp_path):
    """Second run over the same window must not inflate the sample."""
    stub_feed(feed_from_bars(stubbed_bars))

    first = main.analyse(use_bars=True, bars_root=tmp_path)
    second = main.analyse(use_bars=True, bars_root=tmp_path)

    assert first["range"]["stored"] == second["range"]["stored"]
    assert second["range"]["appended"] == 0


def test_a_bar_feed_from_another_instrument_is_loud(stub_feed, stubbed_bars, tmp_path):
    """Local EMAs on a ~2,280 market against a feed claiming 3,500."""
    now = datetime.now(timezone.utc)
    feed = {}
    for timeframe, rate in BULLISH_RATES.items():
        snapshot = ladder(timeframe, 3500.0, rate)
        snapshot.indicators["ADX"] = 30.0
        snapshot.as_of = now
        feed[timeframe] = snapshot
    stub_feed(feed)

    payload = main.analyse(use_bars=True, bars_root=tmp_path)

    assert payload["range"]["status"] == "mismatch"
    assert "disagree with the feed" in payload["range"]["reason"]
    assert "MISMATCH" in main.summarize(payload)
    assert payload["decision"]  # still a decision: reconciliation is not a crash


# --------------------------------------------------------------------------- #
# Failure handling -> exit codes
# --------------------------------------------------------------------------- #


def test_stale_data_exits_two(stub_feed, capsys):
    snaps = fresh_book(BULLISH_RATES)
    old = datetime.now(timezone.utc) - timedelta(hours=6)
    for snap in snaps.values():
        snap.as_of = old
    stub_feed(snaps)

    assert main.main(["--no-chart"]) == main.EXIT_STALE
    captured = capsys.readouterr()
    assert "stale" in captured.err.lower()
    assert captured.out == ""  # no half-finished report is printed


def test_feed_failure_message_is_not_a_traceback(monkeypatch, capsys):
    def boom(**kwargs):
        raise indicators.DataFetchError("connection reset")

    monkeypatch.setattr(main.indicators, "call_any_analysis", boom)
    assert main.main(["--no-chart"]) == main.EXIT_ERROR
    captured = capsys.readouterr()
    assert "market data unavailable" in captured.err
    assert "Traceback" not in captured.err


def test_missing_indicator_exits_one(monkeypatch):
    def boom(**kwargs):
        raise indicators.MissingIndicatorError("15m: missing EMA200")

    monkeypatch.setattr(main.indicators, "call_any_analysis", boom)
    assert main.main(["--no-chart"]) == main.EXIT_ERROR


def test_unverified_freshness_is_reported(stub_feed):
    snaps = fresh_book(BULLISH_RATES)
    for snap in snaps.values():
        snap.as_of_source = "fetch-clock"
    stub_feed(snaps)

    payload = main.run(main.build_parser().parse_args(["--no-chart"]))
    assert payload["freshness"]["status"] == "unverified"
    assert "cannot be confirmed" in payload["freshness"]["note"]


# --------------------------------------------------------------------------- #
# Output side effects
# --------------------------------------------------------------------------- #


def test_chart_is_written(stub_feed, tmp_path):
    pytest.importorskip("matplotlib")
    stub_feed(fresh_book(BULLISH_RATES))
    target = tmp_path / "nested" / "chart.png"

    payload = main.run(main.build_parser().parse_args(["--chart", str(target)]))

    assert payload["chart"] == str(target)
    assert target.exists() and target.stat().st_size > 1000


def test_chart_failure_does_not_kill_the_run(stub_feed, monkeypatch, tmp_path):
    stub_feed(fresh_book(BULLISH_RATES))
    import plot as charting

    def boom(*args, **kwargs):
        raise RuntimeError("no display")

    monkeypatch.setattr(charting, "render", boom)
    payload = main.run(main.build_parser().parse_args(["--chart", str(tmp_path / "x.png")]))
    assert payload["chart"] is None
    assert payload["decision"] == "CANDIDATE"


def test_json_is_written(stub_feed, tmp_path):
    stub_feed(fresh_book(BULLISH_RATES))
    destination = tmp_path / "reports" / "run.json"

    assert main.main(["--no-chart", "--json", str(destination)]) == main.EXIT_OK

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["decision"] == "CANDIDATE"
    assert saved["market"]["spot"] == 2300.0


def test_summary_is_readable(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart", "--equity", "1000000"]))
    text = main.summarize(payload)

    assert "ETHUSD @ COINBASE" in text
    assert "drift %/h" in text
    assert "expected" in text and "30m" in text and "4h" in text
    assert "decision   CANDIDATE" in text
    assert "bull_put_spread" in text
    assert "max loss" in text
    assert "not investment advice" in text


def test_summary_reports_no_trade_cleanly(stub_feed):
    stub_feed(fresh_book(MIXED_RATES))
    text = main.summarize(main.run(main.build_parser().parse_args(["--no-chart"])))
    assert "decision   NO TRADE" in text
    assert "sell " not in text and "buy " not in text


# --------------------------------------------------------------------------- #
# What priced the spread
# --------------------------------------------------------------------------- #


def _chain_fixture():
    import json
    from pathlib import Path

    import iv

    path = Path(__file__).parent / "fixtures" / "deribit_eth_chain.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return iv.build_snapshot(payload, now=datetime.now(timezone.utc))


def test_the_payload_says_when_it_priced_off_the_stub(stub_feed):
    """A synthetic credit must never be mistaken for a market one."""
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    assert payload["chain"]["source"] == "synthetic"
    assert payload["chain"]["iv_atm"] == pytest.approx(0.60)
    assert "offline" in payload["chain"]["note"]


def test_synthetic_can_be_requested_and_the_note_admits_it(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart", "--chain", "synthetic"]))

    assert payload["chain"]["source"] == "synthetic"
    assert "--chain synthetic" in payload["chain"]["note"]


def test_a_live_chain_is_used_when_it_is_available(stub_feed, monkeypatch):
    import iv

    snapshot = _chain_fixture()
    monkeypatch.setattr(iv, "fetch_chain", lambda *a, **k: snapshot)
    stub_feed(fresh_book(BULLISH_RATES))

    payload = main.run(
        main.build_parser().parse_args(["--no-chart", "--equity", "1000000"])
    )

    assert payload["chain"]["source"] == "deribit"
    assert payload["chain"]["quotes"] == len(snapshot.quotes)
    assert 0.3 < payload["chain"]["iv_atm"] < 0.8
    # And the plan really was priced off it, not merely labelled. The plan
    # rounds to 4dp, and the 2098 strike sits just below the quoted wing, so
    # this clamps to 2100 -- a loose bound is the honest one here.
    assert payload["plan"]["iv_short"] == pytest.approx(
        iv.DeribitChain(snapshot, max_age_minutes=0).iv(
            "put", 30, payload["plan"]["short_strike"], payload["market"]["spot"]
        ),
        abs=1e-3,
    )


def test_a_dead_chain_costs_the_chain_not_the_run(stub_feed):
    """The market read must survive a broken option feed."""
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    assert payload["analysis"]["direction"] == analysis.BULLISH
    assert payload["plan"] is not None
    assert "unavailable" in payload["chain"]["note"]


def test_summary_reports_the_chain_and_its_vol(stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = main.run(main.build_parser().parse_args(["--no-chart"]))

    text = main.summarize(payload)

    assert "chain" in text
    assert "SYNTHETIC" in text.upper()
    assert "ATM IV" in text
