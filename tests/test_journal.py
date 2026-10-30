"""Tests for the append-only projection journal.

No network: every payload here is built from the stubbed pipeline stages.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import journal
import main as pipeline
from test_main import BULLISH_RATES, fresh_book


def make_payload(rates=None):
    snaps = fresh_book(rates or BULLISH_RATES)
    regime, result = pipeline.evaluate(snaps, equity=1_000_000)
    return pipeline.build_payload(
        symbol="ETHUSD",
        exchange="COINBASE",
        snapshots=snaps,
        regime=regime,
        result=result,
        max_age=3600.0,
    )


# --------------------------------------------------------------------------- #
# Record shape
# --------------------------------------------------------------------------- #


def test_record_from_payload_keeps_every_projection():
    payload = make_payload()
    record = journal.record_from_payload(payload)

    assert record["schema"] == journal.SCHEMA_VERSION
    assert record["symbol"] == "ETHUSD"
    assert record["spot"] == pytest.approx(payload["market"]["spot"])
    assert set(record["drift_pct_per_hour"]) == {"15m", "30m", "4h"}
    assert len(record["projections"]) == 2
    for item in record["projections"]:
        assert {"horizon_hours", "expected_price", "low", "high", "direction"} <= set(item)
        assert item["low"] < item["high"]


def test_record_is_json_serialisable():
    record = journal.record_from_payload(make_payload())
    assert json.loads(json.dumps(record))["schema"] == journal.SCHEMA_VERSION


def test_record_keeps_the_ladder_split_so_a_lead_lag_can_be_asked_later():
    """The front rung turns before the median; that is only askable if it is kept.

    v2 fields exist for plan.md H1: a reversal signature is *ordered*
    front -> median -> back, and the journal stored only the median.
    """
    payload = make_payload()
    split = journal.record_from_payload(payload)["drift_split"]

    assert set(split) == {"15m", "30m", "4h"}
    for timeframe, row in split.items():
        assert set(row) == {"front", "median", "back"}
        assert row["median"] == pytest.approx(payload["market"]["timeframes"][timeframe]["drift_pct_per_hour"])


def test_the_ladder_split_negates_the_ladder_slope():
    """front/back are clock-side drifts, not the lookback slopes they come from.

    Storing the raw ladder slope here would reintroduce the inverted sign in a
    field named "drift" -- invariant 8, one layer down.
    """
    payload = make_payload()
    row = payload["market"]["timeframes"]["15m"]
    split = journal.record_from_payload(payload)["drift_split"]["15m"]

    assert split["front"] == pytest.approx(-row["ladder_slope_pct_per_hour"]["EMA10"], abs=1e-5)
    assert split["back"] == pytest.approx(-row["ladder_slope_pct_per_hour"]["EMA200"], abs=1e-5)


def test_record_carries_what_the_spread_was_priced_off():
    """A synthetic credit and a market credit must be distinguishable in history."""
    record = journal.record_from_payload(make_payload())

    assert record["chain"]["source"] in {"deribit", "synthetic"}
    assert "iv_atm" in record["chain"]


def test_record_keeps_the_scenario_that_led_to_the_decision():
    """The historical helper: why the regime read resolved, and what the credit
    gate said about the resulting spread -- written whether or not a trade was
    proposed."""
    payload = make_payload()
    record = journal.record_from_payload(payload)

    scenario = record["scenario"]
    assert scenario["decision"] == payload["decision"]
    assert scenario["reason"] == payload["reason"]
    assert scenario["plan"] is None or isinstance(scenario["plan"], dict)
    regime = scenario["regime"]
    assert regime["direction"] in {"bullish", "bearish", "neutral"}
    assert isinstance(regime["agreement"], int)
    assert regime["drift_pct_per_hour"]["15m"] > 0  # bullish fixture: rising drift
    json.loads(json.dumps(scenario))  # must survive the JSONL round trip


def test_record_version_is_four_and_still_reads_v3():
    assert journal.SCHEMA_VERSION == 4
    assert 3 in journal.READABLE_SCHEMA_VERSIONS
    assert 4 in journal.READABLE_SCHEMA_VERSIONS


def test_record_says_when_the_range_was_not_measured():
    """An unmeasured band must not be pooled with a measured one later."""
    record = journal.record_from_payload(make_payload())

    assert record["range"]["status"] == "unavailable"
    assert record["range"]["atr_pct"] == {}
    assert record["range"]["compression_percentile"] is None


def test_record_keeps_the_volatility_that_the_band_was_compared_with():
    payload = make_payload()
    payload["range"] = {
        "status": "ok",
        "compression": {"percentile": 12.0, "reads": "compressed"},
        "timeframes": {
            "15m": {
                "atr_pct": 0.42,
                "expected_move_pct": {"30m": 0.29, "4h": 1.18},
            }
        },
    }

    recorded = journal.record_from_payload(payload)["range"]

    assert recorded["status"] == "ok"
    assert recorded["compression_percentile"] == 12.0
    assert recorded["atr_pct"]["15m"] == 0.42
    assert recorded["expected_move_pct"]["15m"]["4h"] == 1.18
    json.loads(json.dumps(recorded))  # must survive the JSONL round trip


# --------------------------------------------------------------------------- #
# Append / load
# --------------------------------------------------------------------------- #


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "projections.jsonl"
    record = journal.record_from_payload(make_payload())

    journal.append(path, record)  # creates the parent directory
    journal.append(path, {**record, "collected_at": "2026-10-29T00:00:00+00:00"})

    loaded = journal.load(path)
    assert len(loaded) == 2
    assert loaded[0]["symbol"] == "ETHUSD"
    assert [line.count("\n") for line in path.read_text().splitlines()] == [0, 0]


def test_load_missing_file_is_empty(tmp_path):
    assert journal.load(tmp_path / "absent.jsonl") == []


def test_corrupt_line_is_skipped_not_fatal(tmp_path, caplog):
    path = tmp_path / "j.jsonl"
    journal.append(path, journal.record_from_payload(make_payload()))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"schema": 1, "collected_at": "trunc\n')  # crash mid-write

    with caplog.at_level("WARNING"):
        loaded = journal.load(path)

    assert len(loaded) == 1
    assert "skipped" in caplog.text


def test_schema_mismatch_is_refused_on_write(tmp_path):
    record = journal.record_from_payload(make_payload())
    with pytest.raises(ValueError, match="schema"):
        journal.append(tmp_path / "j.jsonl", {**record, "schema": 999})
    assert not (tmp_path / "j.jsonl").exists()


def test_schema_mismatch_is_skipped_on_read(tmp_path, caplog):
    path = tmp_path / "j.jsonl"
    path.write_text(json.dumps({"schema": 999, "collected_at": "2026-10-28T00:00:00+00:00"}) + "\n")
    with caplog.at_level("WARNING"):
        assert journal.load(path) == []
    assert "schema" in caplog.text


def test_record_without_a_timestamp_is_refused(tmp_path):
    record = journal.record_from_payload(make_payload())
    record.pop("collected_at")
    with pytest.raises(ValueError, match="collected_at"):
        journal.append(tmp_path / "j.jsonl", record)


# --------------------------------------------------------------------------- #
# Pipeline integration
# --------------------------------------------------------------------------- #


def test_analyse_journals_when_asked(tmp_path, stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    path = tmp_path / "j.jsonl"

    payload = pipeline.run(
        pipeline.build_parser().parse_args(["--no-chart", "--journal", str(path)])
    )

    assert payload["journal"]["status"] == "ok"
    assert len(journal.load(path)) == 1


def test_analyse_does_not_journal_by_default(tmp_path, stub_feed):
    stub_feed(fresh_book(BULLISH_RATES))
    payload = pipeline.run(pipeline.build_parser().parse_args(["--no-chart"]))
    assert "journal" not in payload


def test_a_journal_failure_does_not_sink_the_run(tmp_path, stub_feed, caplog):
    stub_feed(fresh_book(BULLISH_RATES))

    class Exploding:
        DEFAULT_PATH = tmp_path

        @staticmethod
        def record_from_payload(payload):
            return {}

        @staticmethod
        def append(path, record):
            raise OSError("disk full")

    import sys

    sys.modules["journal"] = Exploding
    try:
        with caplog.at_level("WARNING"):
            payload = pipeline.run(
                pipeline.build_parser().parse_args(["--no-chart", "--journal", "x.jsonl"])
            )
    finally:
        sys.modules["journal"] = journal

    assert payload["journal"]["status"] == "error"
    assert "disk full" in payload["journal"]["error"]
    assert payload["decision"] == "CANDIDATE"  # the analysis still stands
    assert "journal could not be written" in caplog.text


@pytest.fixture
def stub_feed(monkeypatch):
    def install(snaps):
        monkeypatch.setattr(pipeline.indicators, "call_any_analysis", lambda **kwargs: snaps)
        return snaps

    return install


def test_timestamps_are_utc_isoformat():
    record = journal.record_from_payload(make_payload())
    stamp = datetime.fromisoformat(record["collected_at"])
    assert stamp.tzinfo is not None
    assert stamp.utcoffset() == timezone.utc.utcoffset(None)


# --------------------------------------------------------------------------- #
# The shadow gate
# --------------------------------------------------------------------------- #


def _shadow_block() -> dict:
    return {
        "status": "ok",
        "mode": "shadow",
        "verdict": "NO TRADE",
        "pick": None,
        "agrees_with_direction": None,
        "sigma": {"primary": "21d", "value": 0.4496, "windows": {"21d": 0.4496}},
        "sides": {"put": {}, "call": {}},
        "ladder": [
            {
                "side": "put",
                "target_delta": 0.3,
                "expiry": "2026-11-27",
                "dte_actual": 30.06,
                "short_strike": 2200.0,
                "long_strike": 2150.0,
                "width": 50.0,
                "credit": 12.93,
                "credit_mark": 13.84,
                "credit_source": "book",
                "credit_ratio": 0.2586,
                "breakeven_p": 0.7414,
                "p_keep": 0.6436,
                "p_partial": 0.0554,
                "p_max_loss": 0.301,
                "iv_short": 0.522,
                "iv_long": 0.532,
                "edge": -1.84,
                "sigma_rv": 0.4496,
                "sigma_window": "21d",
                "verdict": "NO TRADE",
                "reason": "edge -1.84",
                "status": "ok",
            }
        ],
        "reason": "put: edge -1.84",
    }


def test_the_shadow_gate_is_journaled_with_the_vol_it_was_measured_against():
    """Without the window, "IV is rich" is unfalsifiable: the ratio flips
    between 21 days and a year."""
    payload = make_payload()
    payload["pricing"] = _shadow_block()

    record = journal.record_from_payload(payload)

    assert record["pricing"]["status"] == "ok"
    assert record["pricing"]["sigma"]["window"] == "21d"
    assert record["pricing"]["sigma"]["value"] == pytest.approx(0.4496)


def test_the_ladder_is_journaled_so_each_contract_can_be_resolved_later():
    payload = make_payload()
    payload["pricing"] = _shadow_block()

    row = journal.record_from_payload(payload)["pricing"]["ladder"][0]

    # Every field a scoring pass needs to settle this contract against bars.
    assert {
        "side",
        "expiry",
        "short_strike",
        "long_strike",
        "width",
        "credit",
        "p_keep",
        "p_max_loss",
        "edge",
        "sigma_rv",
        "sigma_window",
        "verdict",
    } <= set(row)
    assert row["expiry"] == "2026-11-27"
    assert row["credit_source"] == "book"


def test_a_shadow_that_was_not_measured_is_journaled_as_such():
    """An unmeasured edge must never be pooled with a measured one."""
    record = journal.record_from_payload(make_payload())

    assert record["pricing"]["status"] == "unavailable"


def test_the_shadow_does_not_change_the_decision_it_is_recorded_beside():
    payload = make_payload()
    payload["pricing"] = _shadow_block()

    record = journal.record_from_payload(payload)

    assert record["decision"] == payload["decision"]
