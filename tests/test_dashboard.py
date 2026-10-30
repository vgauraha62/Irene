"""Tests for the dashboard callback.

The callback is where all the logic lives, so it is tested directly: no server
is launched and no network is touched. Every test stubs ``pipeline.analyse``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

pytest.importorskip("gradio")

import dashboard  # noqa: E402
import journal  # noqa: E402
import main as pipeline  # noqa: E402
from test_analysis import book  # noqa: E402

BULLISH_RATES = {"15m": 0.10, "30m": 0.08, "4h": 0.06}
MIXED_RATES = {"15m": -0.10, "30m": 0.08, "4h": 0.06}

FORM = ("ETHUSD", "COINBASE", 0.30, 30, 1_000_000, 1.0, True)


@pytest.fixture(autouse=True)
def isolated_journal(tmp_path, monkeypatch):
    """Never read the user's real journal or base rate: tests must not depend on disk state."""
    path = tmp_path / "projections.jsonl"
    monkeypatch.setattr(dashboard, "JOURNAL_PATH", path)
    monkeypatch.setattr(dashboard, "BASE_RATE_PATH", tmp_path / "base_rate.json")
    return path


def make_payload(rates, *, equity=1_000_000, chart=None, source="exchange"):
    """A real payload from the real pipeline stages, without the network."""
    snaps = book(**rates)
    now = datetime.now(timezone.utc)
    for snap in snaps.values():
        snap.as_of = now
        snap.as_of_source = source
    regime, result = pipeline.evaluate(snaps, equity=equity)
    payload = pipeline.build_payload(
        symbol="ETHUSD",
        exchange="COINBASE",
        snapshots=snaps,
        regime=regime,
        result=result,
        max_age=3600.0,
        chart_path=chart,
    )
    return payload, snaps, regime


@pytest.fixture
def stub(monkeypatch):
    def install(payload):
        monkeypatch.setattr(dashboard.pipeline, "analyse", lambda **kwargs: payload)

    return install


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_refresh_returns_every_component(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    stub(payload)

    outputs = dashboard.refresh(*FORM)

    assert len(outputs) == dashboard.OUTPUTS
    header, decision, chart, timeframe_table, ladder, plan, projection = outputs[:7]
    pricing = outputs[7]
    verdict, scorecard, health = outputs[8:]

    assert "ETHUSD @ COINBASE" in header
    assert "2,300.00" in header
    assert "freshness verified" in header
    assert "CANDIDATE" in decision
    assert "bull put spread" in plan
    assert chart is None  # this payload has no chart file

    assert isinstance(timeframe_table, pd.DataFrame)
    assert timeframe_table.shape[0] == 3
    assert set(timeframe_table.columns) >= {"timeframe", "drift %/h", "ADX"}
    assert (timeframe_table["drift %/h"] > 0).all()

    assert isinstance(ladder, pd.DataFrame)
    assert ladder.shape[0] == 6
    assert list(ladder.columns) == ["EMA", "15m", "30m", "4h"]

    assert isinstance(projection, pd.DataFrame)
    assert list(projection["horizon"]) == ["30m", "4h"]
    assert set(projection.columns) >= {"horizon", "expected", "low", "high", "read"}


def test_refresh_reports_no_trade_without_a_plan(stub):
    payload, _, _ = make_payload(MIXED_RATES)
    stub(payload)

    _, decision, _, _, _, plan, projection, *_ = dashboard.refresh(*FORM)

    assert "NO TRADE" in decision
    assert "disagree" in decision
    assert "No spread proposed" in plan
    # The projection is the useful part precisely when there is no trade.
    assert len(projection) == 2


def test_unverified_freshness_is_surfaced(stub):
    payload, _, _ = make_payload(BULLISH_RATES, source="fetch-clock")
    stub(payload)

    header = dashboard.refresh(*FORM)[0]
    assert "UNVERIFIED" in header
    assert "cannot be confirmed" in header


def test_chart_is_loaded_as_an_image(stub, tmp_path):
    payload, snaps, regime = make_payload(BULLISH_RATES)
    import plot as charting

    path = charting.render(snaps, regime, path=tmp_path / "chart.png")
    payload["chart"] = str(path)
    stub(payload)

    chart = dashboard.refresh(*FORM)[2]
    assert chart is not None
    assert chart.size[0] > 500


# --------------------------------------------------------------------------- #
# Input validation and failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "symbol", ["", "E", "ETH USD", "ETHUSD<script>", "A" * 20, "ET-USD", "  "]
)
def test_invalid_symbol_never_reaches_the_feed(monkeypatch, symbol):
    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(dashboard.pipeline, "analyse", spy)

    outputs = dashboard.refresh(symbol, "COINBASE", 0.3, 30, 1000, 1.0, True)

    assert calls == []  # rejected before any fetch
    assert len(outputs) == dashboard.OUTPUTS
    assert "not a valid ticker" in outputs[1]


def test_valid_symbol_is_normalised_before_the_feed(monkeypatch):
    seen: dict = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return make_payload(BULLISH_RATES)[0]

    monkeypatch.setattr(dashboard.pipeline, "analyse", spy)

    dashboard.refresh("  ethusd  ", " coinbase ", 0.3, 30, 1000, 1.0, True)

    assert seen["symbol"] == "ETHUSD"
    assert seen["exchange"] == "COINBASE"
    assert seen["risk_pct"] == pytest.approx(0.01)  # percent -> fraction
    assert seen["chart"] is True


def test_empty_equity_means_no_sizing(monkeypatch):
    seen: dict = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return make_payload(BULLISH_RATES)[0]

    monkeypatch.setattr(dashboard.pipeline, "analyse", spy)

    dashboard.refresh("ETHUSD", "COINBASE", 0.3, 30, 0, 1.0, True)
    assert seen["equity"] is None


def test_invalid_exchange_is_rejected(monkeypatch):
    monkeypatch.setattr(dashboard.pipeline, "analyse", lambda **k: {})
    outputs = dashboard.refresh("ETHUSD", "COIN BASE", 0.3, 30, 1000, 1.0, True)
    assert "not valid" in outputs[1]


def test_stale_data_becomes_a_card(monkeypatch):
    import analysis

    def boom(**kwargs):
        raise analysis.StaleDataError("15m: data is 900s old")

    monkeypatch.setattr(dashboard.pipeline, "analyse", boom)

    outputs = dashboard.refresh(*FORM)
    assert len(outputs) == dashboard.OUTPUTS
    assert "Stale data" in outputs[1]
    assert outputs[2] is None


def test_feed_failure_becomes_a_card(monkeypatch):
    import indicators

    def boom(**kwargs):
        raise indicators.DataFetchError("connection reset")

    monkeypatch.setattr(dashboard.pipeline, "analyse", boom)

    outputs = dashboard.refresh(*FORM)
    assert "Market data unavailable" in outputs[1]


def test_unexpected_failure_is_contained(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("something odd")

    monkeypatch.setattr(dashboard.pipeline, "analyse", boom)

    outputs = dashboard.refresh(*FORM)
    assert len(outputs) == dashboard.OUTPUTS
    assert "Unexpected failure" in outputs[1]


# --------------------------------------------------------------------------- #
# Wiring and safety
# --------------------------------------------------------------------------- #


def test_ui_builds():
    demo = dashboard.build_ui()
    assert isinstance(demo, dashboard.gr.Blocks)


def test_defaults_to_loopback():
    args = dashboard.build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == dashboard.DEFAULT_PORT
    assert args.open is False


def test_share_is_never_exposed():
    """A public tunnel would let strangers use this box as a scraper."""
    flags = vars(dashboard.build_parser().parse_args([]))
    assert "share" not in flags


def test_card_escapes_markup():
    rendered = dashboard._card(
        "decision", dashboard.html.escape("<script>x</script>"), palette=("#000", "#fff")
    )
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_plan_card_escapes_notes(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["plan"]["notes"] = ["<img src=x onerror=alert(1)>"]
    stub(payload)

    plan = dashboard.refresh(*FORM)[5]
    assert "<img" not in plan
    assert "&lt;img" in plan


# --------------------------------------------------------------------------- #
# Score tab
# --------------------------------------------------------------------------- #


def _line(hours_ago: float, spot: float, symbol: str = "ETHUSD", *, horizon=1.0,
          expected=None, low=None, high=None, direction="bullish"):
    """One journal line, dated relative to now so staleness logic is exercised."""
    from datetime import timedelta

    moment = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    expected = spot if expected is None else expected
    return {
        "schema": journal.SCHEMA_VERSION,
        "collected_at": moment.isoformat(),
        "symbol": symbol,
        "exchange": "COINBASE",
        "spot": spot,
        "projections": [
            {
                "horizon_hours": horizon,
                "expected_price": expected,
                "low": 0.0 if low is None else low,
                "high": 1e9 if high is None else high,
                "direction": direction,
            }
        ],
    }


def test_score_tab_says_so_when_there_is_no_journal(isolated_journal):
    verdict, scorecard, health = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "No journal at" in verdict
    assert "*/15" in verdict  # the fix is handed to you, not just the problem
    assert scorecard.empty
    assert health.empty


def test_score_tab_reports_an_empty_journal(isolated_journal):
    isolated_journal.write_text("\n")
    verdict, _, _ = dashboard._score_panel(isolated_journal, "ETHUSD")
    assert "holds no usable line" in verdict


# --------------------------------------------------------------------------- #
# Base rate, pinned above the scorecard
# --------------------------------------------------------------------------- #


def _base_rate_report(lift=0.91, n=272, lift_ci=(0.72, 1.13)):
    return {
        "product_bars": {"count": 39_999, "from": "2025-09-06T13:30", "to": "2026-10-28T05:00"},
        "spec": {"horizon_hours": 4.0, "size_sigma": 0.5},
        "base": {
            "n": 1499,
            "up": {"rate": 24.55},
            "down": {"rate": 23.02},
            "flat": {"rate": 52.43},
        },
        "buckets": {
            "signal": {
                "n": n,
                "lift": lift,
                "lift_ci": list(lift_ci),
                "hit": {"rate": 21.7},
                "signed": {"rate": 44.9, "ci": [39.0, 50.8]},
            }
        },
    }


def test_the_base_rate_is_shown_next_to_the_scorecard(isolated_journal, monkeypatch):
    path = isolated_journal.parent / "base_rate.json"
    path.write_text(json.dumps(_base_rate_report()), encoding="utf-8")
    monkeypatch.setattr(dashboard, "BASE_RATE_PATH", path)
    journal.append(isolated_journal, _line(1.0, 100.0))

    verdict, _, _ = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "base rate" in verdict
    assert "0.91x" in verdict
    assert "indistinguishable from chance" in verdict
    assert "44.9%" in verdict


def test_a_lift_whose_interval_clears_one_is_called_an_edge(isolated_journal, monkeypatch):
    path = isolated_journal.parent / "base_rate.json"
    path.write_text(
        json.dumps(_base_rate_report(lift=1.34, lift_ci=(1.08, 1.62))), encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "BASE_RATE_PATH", path)
    journal.append(isolated_journal, _line(1.0, 100.0))

    verdict, _, _ = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "1.34x" in verdict
    assert "clears 1.00x" in verdict


def test_a_lift_whose_interval_sits_below_one_is_called_worse(isolated_journal, monkeypatch):
    path = isolated_journal.parent / "base_rate.json"
    path.write_text(
        json.dumps(_base_rate_report(lift=0.71, lift_ci=(0.58, 0.88))), encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "BASE_RATE_PATH", path)
    journal.append(isolated_journal, _line(1.0, 100.0))

    verdict, _, _ = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "significantly worse than chance" in verdict


def test_no_base_rate_file_means_no_claim(isolated_journal):
    journal.append(isolated_journal, _line(1.0, 100.0))

    verdict, _, _ = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "base rate" not in verdict.lower()


def test_a_corrupt_base_rate_file_is_contained(isolated_journal, monkeypatch):
    path = isolated_journal.parent / "base_rate.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(dashboard, "BASE_RATE_PATH", path)
    journal.append(isolated_journal, _line(1.0, 100.0))

    verdict, _, _ = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "base rate unreadable" in verdict


def test_a_feed_failure_still_shows_the_base_rate(isolated_journal, monkeypatch):
    path = isolated_journal.parent / "base_rate.json"
    path.write_text(json.dumps(_base_rate_report()), encoding="utf-8")
    monkeypatch.setattr(dashboard, "BASE_RATE_PATH", path)
    journal.append(isolated_journal, _line(1.0, 100.0))

    def boom(**kwargs):
        raise RuntimeError("feed down")

    monkeypatch.setattr(dashboard.pipeline, "analyse", boom)

    outputs = dashboard.refresh(*FORM)

    # live outputs come first (8 of them, the shadow gate last), then
    # verdict / scorecard / health -- indexed from the end so an inserted
    # live panel cannot silently shift the score tab's assertion.
    assert "0.91x" in outputs[-3]


def test_score_tab_renders_metrics_and_a_small_sample_banner(isolated_journal):
    journal.append(isolated_journal, _line(1.0, 100.0, expected=101.0))
    journal.append(isolated_journal, _line(0.0, 101.0, expected=101.0))

    verdict, scorecard, health = dashboard._score_panel(isolated_journal, "ETHUSD")

    assert "plumbing, not evidence" in verdict
    assert "skill" in scorecard.columns
    assert scorecard.loc[0, "n"] == 1
    assert scorecard.loc[0, "skill"] == pytest.approx(1.0)
    assert not health.empty


def test_score_tab_flags_a_dead_collector(isolated_journal):
    journal.append(isolated_journal, _line(3.0, 100.0, expected=101.0))
    journal.append(isolated_journal, _line(2.5, 101.0, expected=101.0))

    _, _, health = dashboard._score_panel(isolated_journal, "ETHUSD")
    writer = health.loc[health["metric"] == "writer", "value"].iloc[0]
    assert "STALE" in writer


def test_score_tab_does_not_borrow_another_symbols_numbers(isolated_journal):
    journal.append(isolated_journal, _line(1.0, 100.0))
    journal.append(isolated_journal, _line(0.0, 101.0))

    verdict, scorecard, health = dashboard._score_panel(isolated_journal, "BTCUSD")

    assert "none for BTCUSD" in verdict
    assert "ETHUSD" in verdict
    assert scorecard.empty
    assert not health.empty  # the journal itself is still reported


def test_score_tab_is_not_blanked_by_a_feed_failure(stub, monkeypatch, isolated_journal):
    journal.append(isolated_journal, _line(1.0, 100.0, expected=101.0))
    journal.append(isolated_journal, _line(0.0, 101.0))

    def boom(**kwargs):
        raise dashboard.indicators.DataFetchError("HTTP 429")

    monkeypatch.setattr(dashboard.pipeline, "analyse", boom)

    outputs = dashboard.refresh(*FORM)

    assert "Market data unavailable" in outputs[1]
    assert outputs[-3].strip()  # the verdict still renders
    assert not outputs[-2].empty  # and so does the scorecard


def test_page_load_reads_the_journal_and_never_scrapes(monkeypatch, isolated_journal):
    called: list = []

    def spy(**kwargs):
        called.append(kwargs)
        return {}

    monkeypatch.setattr(dashboard.pipeline, "analyse", spy)
    journal.append(isolated_journal, _line(0.0, 100.0))

    verdict, _, health = dashboard.load_scoreboard("ETHUSD")

    assert called == []  # no TradingView request on page load
    assert verdict.strip()
    assert not health.empty


# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #


def test_about_tab_explains_the_counterintuitive_bits():
    about = dashboard.ABOUT

    # The chart reads backwards on purpose: state it, do not leave it implied.
    assert "hours of EMA lookback" in about
    assert "means price has" in about
    # Drift sign, and that it is not the ladder slope.
    assert "Positive is up" in about or "Positive = rising" in about
    assert "opposite sign" in about
    # The scorecard rule.
    assert "not `MAE`" in about
    # The base rate is the number to beat, and lift is against chance, not 50%.
    assert "lift" in about.lower() and "1.00x" in about
    assert "base rate" in about.lower()
    assert "Wilson" in about
    # No-trade is a result.
    assert "NO TRADE" in about
    # The limits are on the page, not only in the code. The chain is live now, so
    # the honest statement is "market, or explicitly labelled stub" -- not "stub".
    assert "SYNTHETIC" in about and "Deribit" in about
    assert "UNVERIFIED" in about


def test_about_tab_is_wired_into_the_ui():
    demo = dashboard.build_ui()
    registered = {getattr(function, "fn", None) for function in demo.fns.values()}
    assert dashboard.load_scoreboard in registered
    assert dashboard.refresh in registered


def test_the_live_header_is_html_so_the_range_panel_keeps_its_styling():
    """Markdown sanitises inline styles, and the range panel is a styled table."""
    import gradio as gr

    demo = dashboard.build_ui()
    headers = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.HTML) and "Refresh" in str(block.value)
    ]

    assert headers, "the live header must be gr.HTML, not gr.Markdown"


def test_a_failure_does_not_show_raw_markdown_in_the_html_header():
    failed = dashboard._failure("boom")

    assert failed[0].startswith("<div")
    assert "Run failed" in failed[0]


# --------------------------------------------------------------------------- #
# Launch failures
# --------------------------------------------------------------------------- #


class _FakeDemo:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.launched = False

    def queue(self, **_kwargs: object) -> None:
        return None

    def launch(self, **_kwargs: object) -> None:
        self.launched = True
        raise self.error


def test_a_busy_port_explains_itself_instead_of_dumping_a_traceback(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        dashboard,
        "build_ui",
        lambda *a, **k: _FakeDemo(OSError("Cannot find empty port in range: 7860-7860.")),
    )

    assert dashboard.main(["--port", "7860"]) == 1

    err = capsys.readouterr().err
    assert "7860 is already in use" in err
    assert "kill <pid>" in err
    assert "--port 7861" in err  # the way out is handed over, not just the diagnosis


def test_an_unrelated_launch_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        dashboard, "build_ui", lambda *a, **k: _FakeDemo(OSError("cannot bind socket"))
    )
    with pytest.raises(OSError, match="cannot bind socket"):
        dashboard.main(["--port", "7860"])


def test_the_header_names_the_chain_that_priced_the_spread(stub):
    """A synthetic credit must be visible as such on screen, not only in JSON."""
    payload, _, _ = make_payload(BULLISH_RATES)
    stub(payload)

    header = dashboard.refresh(*FORM)[0]

    assert "SYNTHETIC chain" in header
    assert "ATM IV" in header


def test_the_header_marks_a_live_chain_differently(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["chain"] = {
        "source": "deribit",
        "iv_atm": 0.5042,
        "quotes": 38,
        "note": None,
    }
    stub(payload)

    header = dashboard.refresh(*FORM)[0]

    assert "DERIBIT chain" in header
    assert "50.4%" in header


# --------------------------------------------------------------------------- #
# Range: measured vs unmeasured must never look alike
# --------------------------------------------------------------------------- #


def test_an_unmeasured_range_is_stated_not_left_blank(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["range"] = {"status": "unavailable", "reason": "bar feed offline"}
    stub(payload)

    header = dashboard.refresh(*FORM)[0]

    assert "range UNMEASURED" in header
    assert "bar feed offline" in header


def test_a_measured_range_shows_atr_and_compression(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    for row in payload["market"]["timeframes"].values():
        row["atr_pct"] = 0.42
    payload["range"] = {
        "status": "ok",
        "timeframes": {"15m": {"atr_pct": 0.42}},
        "compression": {"percentile": 12.0, "reads": "compressed"},
    }
    stub(payload)

    header = dashboard.refresh(*FORM)[0]

    assert "0.42%" in header
    assert "compression <b>12th</b> percentile" in header
    assert "range UNMEASURED" not in header


def test_the_band_is_compared_with_a_measured_sigma_on_screen(stub):
    """The band-is-too-narrow finding has to be readable, not a paragraph."""
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["range"] = {
        "status": "ok",
        "timeframes": {"15m": {"atr_pct": 0.24, "expected_move_pct": {"30m": 0.28, "4h": 0.92}}},
        "compression": {"percentile": 4.0, "reads": "very compressed"},
    }
    for projection in payload["projection"]:
        projection["basis_timeframe"] = "15m"
        projection["horizon"] = "4h"
        projection["band_pct"] = 0.29
    stub(payload)

    header = dashboard.refresh(*FORM)[0]

    assert "vs measured 1σ 0.92%" in header
    assert "0.32x" in header
    assert "narrower than a random walk" in header


def test_a_cross_check_mismatch_is_flagged_on_screen(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["range"] = {
        "status": "mismatch",
        "timeframes": {"15m": {"atr_pct": 0.42}},
        "compression": {"percentile": None, "reads": "unranked"},
        "cross_check": {"15m": {"status": "mismatch", "reason": "8% apart"}},
    }
    stub(payload)

    header = dashboard.refresh(*FORM)[0]

    assert "DISAGREE" in header


def test_the_timeframe_table_carries_a_measured_atr(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["market"]["timeframes"]["15m"]["atr_pct"] = 0.42
    stub(payload)

    table = dashboard.refresh(*FORM)[3]

    assert "ATR %" in table.columns
    assert table[table["timeframe"] == "15m"]["ATR %"].iloc[0] == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# The shadow gate on screen
# --------------------------------------------------------------------------- #


def _shadow_payload(**overrides):
    payload, _, _ = make_payload(BULLISH_RATES)
    block = {
        "status": "ok",
        "mode": "shadow",
        "verdict": "NO TRADE",
        "pick": None,
        "agrees_with_direction": False,
        "direction_side": "put",
        "sigma": {"primary": "21d", "value": 0.4496, "windows": {"21d": 0.4496}},
        "reason": "put: edge -1.84; call: credit 19% of width",
        "sides": {
            "put": {
                "side": "put",
                "short_strike": 2200.0,
                "long_strike": 2150.0,
                "credit": 12.93,
                "credit_mark": 13.84,
                "credit_ratio": 0.2586,
                "p_keep": 0.6436,
                "p_partial": 0.0554,
                "p_max_loss": 0.301,
                "edge": -1.84,
                "verdict": "NO TRADE",
            },
            "call": {
                "side": "call",
                "short_strike": 2550.0,
                "long_strike": 2600.0,
                "credit": 9.40,
                "credit_mark": 12.15,
                "credit_ratio": 0.1881,
                "p_keep": 0.7405,
                "p_partial": 0.0424,
                "p_max_loss": 0.2171,
                "edge": -1.64,
                "verdict": "NO TRADE",
            },
        },
    }
    block.update(overrides)
    payload["pricing"] = block
    return payload


def test_the_shadow_gate_is_on_screen_with_its_vol_window():
    html_out = dashboard._pricing(_shadow_payload())

    assert "pricing (shadow)" in html_out.lower()
    assert "NO TRADE" in html_out
    assert "21d" in html_out
    assert "2,200" in html_out and "2,550" in html_out
    assert "-1.84" in html_out


def test_the_shadow_says_it_is_a_record_and_not_an_order():
    html_out = dashboard._pricing(_shadow_payload())

    assert "Shadow only" in html_out
    assert "never emitted" in html_out


def test_a_disagreement_with_the_direction_gate_is_shouted_not_hidden():
    html_out = dashboard._pricing(_shadow_payload())

    assert "DISAGREES" in html_out
    assert "it says put" in html_out


def test_an_unmeasured_shadow_says_so_rather_than_showing_zero():
    """A missing measurement must not render as an edge of zero."""
    html_out = dashboard._pricing(
        {"pricing": {"status": "unavailable", "reason": "no realised vol"}}
    )

    assert "not measured" in html_out
    assert "no realised vol" in html_out
    assert "-0.00" not in html_out


def test_a_take_is_green_and_names_the_side():
    payload = _shadow_payload(verdict="TAKE", pick="put", agrees_with_direction=True)
    html_out = dashboard._pricing(payload)

    assert "TAKE" in html_out
    assert "put side" in html_out
    assert "Agrees with the direction gate" in html_out


def test_the_shadow_panel_is_wired_into_the_refresh(stub):
    payload, _, _ = make_payload(BULLISH_RATES)
    stub(payload)

    outputs = dashboard.refresh(*FORM)

    assert len(outputs) == dashboard.OUTPUTS
    assert "not measured" in outputs[7]  # no pricing block in this payload


def test_the_plan_card_shows_where_the_credit_came_from():
    payload, _, _ = make_payload(BULLISH_RATES)
    payload["plan"]["credit_source"] = "book"
    payload["plan"]["credit_mark"] = payload["plan"]["credit"] + 1.0
    payload["plan"]["expiry"] = "2026-11-27"
    payload["plan"]["dte_actual"] = 30.1

    card = dashboard._plan(payload)

    assert "bid(short)" in card
    assert "2026-11-27" in card
