"""Tests for the projection scorer.

The scorer is deliberately judged on one thing: does it refuse to flatter a bad
projection? Every test here is a case where a naive "mean error" would mislead.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import journal
import main as pipeline
import score
from test_main import BULLISH_RATES, fresh_book

BASE = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)


def stamp(hours: float) -> str:
    return (BASE + timedelta(hours=hours)).isoformat()


def record(hours, spot, projections, *, symbol="ETHUSD"):
    """One journal line. ``projections`` is a list of (horizon, expected, low, high, dir)."""
    return {
        "schema": journal.SCHEMA_VERSION,
        "collected_at": stamp(hours),
        "symbol": symbol,
        "exchange": "COINBASE",
        "spot": spot,
        "projections": [
            {
                "horizon_hours": horizon,
                "expected_price": expected,
                "low": low,
                "high": high,
                "direction": direction,
            }
            for horizon, expected, low, high, direction in projections
        ],
    }


# --------------------------------------------------------------------------- #
# The one number that matters: skill vs doing nothing
# --------------------------------------------------------------------------- #


def test_perfect_projection_beats_the_baseline():
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]),
            record(1, 101.0, []),
        ]
    )
    assert len(card.scored) == 1
    result = card.horizons[0]

    assert result.mae_pct == pytest.approx(0.0)
    assert result.baseline_mae_pct == pytest.approx(1.0)
    assert result.skill == pytest.approx(1.0)
    assert result.band_coverage == 1.0
    assert result.directional_hit_rate == 1.0


def test_a_wrong_projection_scores_negative_skill():
    """Predicting 110 when price stays at 101 is worse than predicting no move."""
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 110.0, 109.0, 111.0, "bullish")]),
            record(1, 101.0, []),
        ]
    )
    result = card.horizons[0]

    assert result.skill < 0
    assert result.mae_pct > result.baseline_mae_pct
    assert result.band_coverage == 0.0
    assert result.directional_hit_rate == 1.0  # it rose, just nowhere near enough


def test_a_flat_forecast_has_no_skill():
    """A projection that just says 'unchanged' cannot beat the same baseline."""
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 100.0, 99.0, 101.0, "neutral")]),
            record(1, 104.0, []),
        ]
    )
    result = card.horizons[0]
    assert result.skill == pytest.approx(0.0, abs=1e-9)
    assert result.directional_hit_rate is None


# --------------------------------------------------------------------------- #
# Irregular observations: refuse to score what cannot be scored honestly
# --------------------------------------------------------------------------- #


def test_no_later_observation_is_unscored_not_dropped():
    card = score.score_records([record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")])])

    assert card.scored == []
    assert card.unscored[score.UNSCORED_NO_DATA] == 1
    assert "nothing to score" in score.format_card(card)


def test_a_late_observation_is_refused():
    """A 1h projection must not be scored against a 2h-later price."""
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]),
            record(2, 101.0, []),
        ]
    )
    assert card.scored == []
    assert card.unscored[score.UNSCORED_TOO_LATE] == 1


def test_a_slightly_late_observation_is_accepted_and_recorded():
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]),
            record(1.1, 101.0, []),  # 6 minutes late, inside 25% of 1h
        ]
    )
    assert len(card.scored) == 1
    assert card.scored[0].gap_hours == pytest.approx(0.1, abs=1e-6)
    assert card.horizons[0].max_gap_hours == pytest.approx(0.1, abs=1e-6)


def test_each_projection_uses_the_first_observation_past_its_own_horizon():
    card = score.score_records(
        [
            record(0, 100.0, [(0.5, 100.5, 99.0, 102.0, "bullish"),
                              (4.0, 104.0, 102.0, 106.0, "bullish")]),
            record(0.5, 100.6, []),
            record(4.0, 104.2, []),
        ]
    )
    assert [item.horizon_hours for item in card.scored] == [0.5, 4.0]
    assert [item.realised_price for item in card.scored] == [100.6, 104.2]


# --------------------------------------------------------------------------- #
# Bookkeeping that protects the sample size
# --------------------------------------------------------------------------- #


def test_duplicate_observations_are_counted_not_double_scored():
    line = record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")])
    card = score.score_records([line, dict(line), record(1, 101.0, [])])

    assert card.duplicates == 1
    assert card.records == 2
    assert len(card.scored) == 1


def test_records_are_sorted_before_pairing():
    """A journal read out of order must not pair a projection with its own past."""
    card = score.score_records(
        [
            record(1, 101.0, []),
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]),
        ]
    )
    assert len(card.scored) == 1
    assert card.scored[0].realised_price == 101.0


def test_neutral_projections_are_excluded_from_the_hit_rate():
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]),  # right
            record(1, 101.0, [(1.0, 100.0, 99.0, 101.0, "bearish")]),  # wrong
            record(2, 102.0, [(1.0, 102.0, 101.0, 103.0, "neutral")]),  # not directional
            record(3, 103.0, []),
        ]
    )
    result = card.horizons[0]
    assert result.n == 3
    assert result.n_directional == 2
    assert result.directional_hit_rate == pytest.approx(0.5)


def test_horizons_are_summarised_separately():
    card = score.score_records(
        [
            record(0, 100.0, [(0.5, 100.5, 100.0, 101.0, "bullish"),
                              (4.0, 104.0, 103.0, 105.0, "bullish")]),
            record(0.5, 100.5, []),
            record(4.0, 104.5, []),
        ]
    )
    horizons = card.horizons
    assert [item.horizon_hours for item in horizons] == [0.5, 4.0]
    assert all(item.n == 1 for item in horizons)


# --------------------------------------------------------------------------- #
# End to end: a real payload must survive the journal round trip
# --------------------------------------------------------------------------- #


def test_a_real_pipeline_payload_is_scoreable(tmp_path):
    snaps = fresh_book(BULLISH_RATES)
    regime, result = pipeline.evaluate(snaps, equity=1_000_000)
    payload = pipeline.build_payload(
        symbol="ETHUSD",
        exchange="COINBASE",
        snapshots=snaps,
        regime=regime,
        result=result,
        max_age=3600.0,
    )

    path = tmp_path / "j.jsonl"
    journal.append(path, journal.record_from_payload(payload))

    later = journal.record_from_payload(payload)
    later["collected_at"] = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    later["spot"] = payload["market"]["spot"] * 1.01
    journal.append(path, later)

    card = score.score_records(journal.load(path))
    # The 30m projection lands on the later run and is scored; the 4h one has
    # no observation at its horizon yet, and the last run's own projections can
    # never be scored. Both show up as unscored rather than vanishing.
    assert [item.horizon_hours for item in card.scored] == [0.5]
    assert card.scored[0].realised_price == pytest.approx(payload["market"]["spot"] * 1.01)
    assert card.unscored[score.UNSCORED_NO_DATA] == 3


# --------------------------------------------------------------------------- #
# Symbols must never be mixed
# --------------------------------------------------------------------------- #


def test_an_eth_projection_is_never_scored_against_a_btc_price():
    """The bug this guards: pairing by time alone silently crosses assets."""
    card = score.score_records(
        [
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")], symbol="ETHUSD"),
            record(1, 60_000.0, [], symbol="BTCUSD"),
            record(1, 101.0, [], symbol="ETHUSD"),
        ],
        symbol="ETHUSD",
    )
    assert len(card.scored) == 1
    assert card.scored[0].realised_price == 101.0
    assert card.scored[0].error_pct == pytest.approx(0.0)


def test_a_mixed_journal_is_refused_without_a_symbol():
    with pytest.raises(ValueError, match="several symbols"):
        score.score_records(
            [
                record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")], symbol="ETHUSD"),
                record(1, 60_000.0, [], symbol="BTCUSD"),
            ]
        )


def test_a_mixed_journal_is_scored_per_symbol():
    cards = score.score_journal(
        [
            record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")], symbol="ETHUSD"),
            record(1, 101.0, [], symbol="ETHUSD"),
            record(0, 60_000.0, [(1.0, 61_200.0, 60_000.0, 62_000.0, "bullish")], symbol="BTCUSD"),
            record(1, 61_200.0, [], symbol="BTCUSD"),
        ]
    )
    assert [card.symbol for card in cards] == ["BTCUSD", "ETHUSD"]
    assert all(card.horizons[0].skill == pytest.approx(1.0) for card in cards)


def test_a_symbol_absent_from_the_journal_scores_nothing():
    card = score.score_records(
        [record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")], symbol="ETHUSD")],
        symbol="BTCUSD",
    )
    assert card.records == 0
    assert card.scored == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_reports_a_missing_journal(tmp_path, capsys):
    assert score.main([str(tmp_path / "absent.jsonl")]) == 1
    assert "no journal" in capsys.readouterr().out


def test_cli_prints_a_table(tmp_path, capsys):
    path = tmp_path / "j.jsonl"
    journal.append(path, record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]))
    journal.append(path, record(1, 101.0, []))

    assert score.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "skill" in out and "baseline" in out
    assert "plumbing, not evidence" in out  # small-sample warning


def test_cli_json_output_is_machine_readable(tmp_path, capsys):
    path = tmp_path / "j.jsonl"
    journal.append(path, record(0, 100.0, [(1.0, 101.0, 100.0, 102.0, "bullish")]))
    journal.append(path, record(1, 101.0, []))

    assert score.main([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["scored"] == 1
    assert payload[0]["horizons"][0]["skill"] == pytest.approx(1.0)


def test_a_journal_of_skipped_lines_says_so_instead_of_printing_nothing(tmp_path, capsys):
    """Silence on stderr used to look like a clean exit."""
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"schema": 0, "collected_at": "2026-01-01T00:00:00+00:00"}) + "\n")

    assert score.main([str(path)]) == 1

    err = capsys.readouterr().err
    assert "no record this build can read" in err
    assert f"schema {journal.SCHEMA_VERSION}" in err


# --------------------------------------------------------------------------- #
# The shadow gate, settled
# --------------------------------------------------------------------------- #
#
# A credit spread is scored on whether the short strike was breached, not on
# whether price went up. These tests pin the three things that make that
# measurement honest: the outcome arithmetic, one observation per contract, and
# the refusal to claim calibration from a handful of expiries.


def bar(when: datetime, close: float):
    import bars

    return bars.Bar(
        open_time=when, open=close, high=close, low=close, close=close, volume=1.0
    )


def ladder_row(
    side="put",
    short=2000.0,
    long=1950.0,
    delta=0.30,
    expiry="2026-10-20T08:00:00+00:00",
    credit=12.0,
    p_keep=0.65,
):
    return {
        "side": side,
        "target_delta": delta,
        "expiry": expiry,
        "short_strike": short,
        "long_strike": long,
        "width": abs(short - long),
        "credit": credit,
        "credit_mark": credit + 1.0,
        "credit_source": "book",
        "credit_ratio": credit / abs(short - long),
        "p_keep": p_keep,
        "p_partial": 0.05,
        "p_max_loss": 1.0 - p_keep - 0.05,
        "edge": -1.0,
        "sigma_rv": 0.45,
        "sigma_window": "21d",
        "verdict": "NO TRADE",
        "reason": "edge",
        "status": "ok",
    }


def shadow_record(when: str, rows, *, symbol="ETHUSD"):
    return {
        "schema": journal.SCHEMA_VERSION,
        "collected_at": when,
        "symbol": symbol,
        "exchange": "COINBASE",
        "spot": 2350.0,
        "pricing": {
            "status": "ok",
            "mode": "shadow",
            "verdict": "NO TRADE",
            "sigma": {"window": "21d", "value": 0.45},
            "ladder": list(rows),
        },
    }


def bars_around(price: float, *, when="2026-10-20T08:00:00+00:00") -> list:
    moment = datetime.fromisoformat(when)
    return [
        bar(moment - timedelta(minutes=30), price),
        bar(moment - timedelta(minutes=15), price),
        bar(moment, price),
        bar(moment + timedelta(minutes=15), price),
    ]


def test_a_kept_spread_is_scored_as_keeping_the_whole_credit():
    records = [shadow_record("2026-09-20T08:00:00+00:00", [ladder_row()])]
    settled, accounting = score.settle_pricing(records, bars_around(2300.0))

    assert accounting["pending"] == 0
    assert len(settled) == 1
    row = settled[0]
    assert row.breached is False
    assert row.max_loss_hit is False
    assert row.pnl == pytest.approx(12.0)


def test_a_breached_spread_pays_out_the_intrinsic_and_no_more():
    records = [shadow_record("2026-09-20T08:00:00+00:00", [ladder_row()])]
    settled, _ = score.settle_pricing(records, bars_around(1940.0))

    row = settled[0]
    assert row.breached is True
    assert row.max_loss_hit is True
    assert row.pnl == pytest.approx(-38.0)  # 12 credit - 50 payoff, capped at width


def test_a_partial_breach_keeps_part_of_the_credit():
    """Between the strikes the payoff is continuous, so the loss is too."""
    records = [shadow_record("2026-09-20T08:00:00+00:00", [ladder_row()])]
    settled, _ = score.settle_pricing(records, bars_around(1970.0))

    row = settled[0]
    assert row.breached is True
    assert row.max_loss_hit is False
    assert row.pnl == pytest.approx(12.0 - 30.0)


def test_the_settlement_price_is_the_last_print_before_the_expiry():
    """The bar opening exactly at expiry is after settlement, so it must not count."""
    records = [shadow_record("2026-09-20T08:00:00+00:00", [ladder_row()])]
    bars = [
        bar(datetime(2026, 10, 20, 7, 45, tzinfo=timezone.utc), 2300.0),
        bar(datetime(2026, 10, 20, 8, 0, tzinfo=timezone.utc), 1900.0),
    ]
    settled, _ = score.settle_pricing(records, bars)

    assert settled[0].settlement == pytest.approx(2300.0)


def test_one_observation_per_expiry_side_and_delta():
    """A 15-minute cron re-prices the same contract thousands of times; that is
    one outcome, not thousands, and pooling it would fake a large sample."""
    rows = [ladder_row()]
    records = [
        shadow_record("2026-09-20T08:00:00+00:00", rows),
        shadow_record("2026-09-21T08:00:00+00:00", rows),
        shadow_record("2026-09-22T08:00:00+00:00", rows),
    ]
    settled, _ = score.settle_pricing(records, bars_around(2300.0))

    assert len(settled) == 1


def test_three_deltas_on_one_expiry_are_three_observations():
    rows = [
        ladder_row(delta=0.15, short=1900.0, long=1850.0),
        ladder_row(delta=0.30, short=2000.0, long=1950.0),
        ladder_row(delta=0.45, short=2100.0, long=2050.0),
    ]
    settled, _ = score.settle_pricing(
        [shadow_record("2026-09-20T08:00:00+00:00", rows)], bars_around(2300.0)
    )

    assert len(settled) == 3


def test_a_future_expiry_is_pending_not_scored():
    rows = [ladder_row(expiry="2099-01-01T08:00:00+00:00")]
    settled, accounting = score.settle_pricing(
        [shadow_record("2026-09-20T08:00:00+00:00", rows)], bars_around(2300.0)
    )

    assert settled == []
    assert accounting["pending"] == 1


def test_an_expiry_before_the_bar_store_is_counted_not_guessed():
    rows = [ladder_row(expiry="2020-01-01T08:00:00+00:00")]
    settled, accounting = score.settle_pricing(
        [shadow_record("2019-12-01T08:00:00+00:00", rows)], bars_around(2300.0)
    )

    assert settled == []
    assert accounting["no_price"] == 1


def test_a_settlement_next_to_a_strike_is_flagged():
    """Our bars are Binance; the contract settles on Deribit's index. Within
    0.5% of a strike the two venues can disagree about the outcome."""
    rows = [ladder_row(short=2000.0, long=1950.0)]
    settled, _ = score.settle_pricing(
        [shadow_record("2026-09-20T08:00:00+00:00", rows)], bars_around(2004.0)
    )

    assert settled[0].near_strike is True


def test_the_realised_breach_rate_is_reported_against_the_implied_one():
    rows = [
        ladder_row(delta=0.30, short=2000.0, long=1950.0),
        ladder_row(delta=0.30, short=2000.0, long=1950.0, expiry="2026-10-21T08:00:00+00:00"),
        ladder_row(delta=0.30, short=2000.0, long=1950.0, expiry="2026-10-22T08:00:00+00:00"),
        ladder_row(delta=0.30, short=2000.0, long=1950.0, expiry="2026-10-23T08:00:00+00:00"),
    ]
    records = [shadow_record("2026-09-20T08:00:00+00:00", rows)]
    bars = (
        bars_around(2300.0, when="2026-10-20T08:00:00+00:00")
        + bars_around(2300.0, when="2026-10-21T08:00:00+00:00")
        + bars_around(1970.0, when="2026-10-22T08:00:00+00:00")
        + bars_around(1970.0, when="2026-10-23T08:00:00+00:00")
    )
    settled, _ = score.settle_pricing(records, bars)
    table = score.pricing_table(settled)

    assert len(table) == 1
    row = table[0]
    assert row["n"] == 4
    assert row["realised_breach_pct"] == pytest.approx(50.0)
    assert row["implied_breach_pct"] == pytest.approx(35.0)
    assert row["lift"] == pytest.approx(1.43, abs=0.01)


def test_a_small_sample_claims_no_edge_however_the_rate_looks():
    """Four observations is a rumour. The verdict is the interval's, not the ratio's."""
    rows = [
        ladder_row(delta=0.30, expiry=f"2026-10-2{day}T08:00:00+00:00")
        for day in range(0, 4)
    ]
    bars = []
    for day in range(0, 4):
        bars += bars_around(1970.0, when=f"2026-10-2{day}T08:00:00+00:00")
    settled, _ = score.settle_pricing([shadow_record("2026-09-20T08:00:00+00:00", rows)], bars)

    row = score.pricing_table(settled)[0]
    assert row["realised_breach_pct"] == pytest.approx(100.0)
    assert row["verdict"] == "too few observations"


def test_a_large_sample_that_excludes_the_implied_rate_is_allowed_to_say_so():
    rows = [
        ladder_row(delta=0.30, expiry=f"2026-09-{day:02d}T08:00:00+00:00")
        for day in range(1, 31)
    ]
    bars = []
    for day in range(1, 31):
        bars += bars_around(1970.0, when=f"2026-09-{day:02d}T08:00:00+00:00")
    settled, _ = score.settle_pricing([shadow_record("2026-09-20T08:00:00+00:00", rows)], bars)

    row = score.pricing_table(settled)[0]
    assert row["n"] == 30
    assert row["verdict"] == "sells too cheap: breaches exceed the price"


def test_the_report_says_nothing_settled_rather_than_showing_a_zero():
    text = score.format_pricing([], {"pending": 6, "no_price": 0, "unusable": 0})

    assert "nothing settled yet" in text
    assert "pending 6" in text
