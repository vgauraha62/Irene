"""Tests for the shadow pricing gate.

The gate exists to answer one question -- is the credit larger than the payoff
the measured volatility expects -- so most of these tests are about *units and
honesty* rather than about whether a particular number looks attractive:

* the credit must be the bid/ask, not the mark;
* the probability must be N(-d2), not 1 - delta;
* the vol window must be named, because the answer flips between 21 days and a
  year;
* and the gate must never turn into a trade.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import analysis
import iv
import pricing

NOW = datetime(2026, 10, 28, 6, 0, tzinfo=timezone.utc)
SPOT = 2300.0


def _bars(closes: list[float], *, step_minutes: int = 15) -> list:
    """A minimal bar series: only ``close`` and ``open_time`` are read."""
    import bars as bar_store

    start = NOW - timedelta(minutes=step_minutes * len(closes))
    return [
        bar_store.Bar(
            open_time=start + timedelta(minutes=step_minutes * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
        )
        for index, close in enumerate(closes)
    ]


def _flat_bars(count: int, price: float = 2300.0) -> list:
    return _bars([price] * count)


def _moving_bars(count: int = 2100, price: float = 2300.0) -> list:
    """A series with real realised vol: an alternating 1% wiggle."""
    # ~0.12% per bar, which annualises to roughly 45% -- the same order as the
    # realised vol this gate is meant to be measured against.
    return _bars([price * (1.0012 if index % 2 else 0.9988) for index in range(count)])


def _book(kind: str, strikes: list[float], *, iv_value: float = 0.55, touch: float = 0.0):
    expiry = NOW + timedelta(days=30)
    legs = []
    for strike in strikes:
        # Book prices are in the underlying, as Deribit quotes them.
        mark = analysis.bs_price(kind, SPOT, strike, 30 / 365, iv_value) / SPOT
        legs.append(
            iv.Quote(
                kind=kind,
                strike=strike,
                expiry=expiry,
                iv=iv_value,
                open_interest=100.0,
                bid=mark * (1.0 - touch),
                ask=mark * (1.0 + touch),
                mark=mark,
            )
        )
    return iv.Book(
        kind=kind,
        expiry=expiry,
        spot=SPOT,
        legs=tuple(sorted(legs, key=lambda leg: leg.strike)),
        observed_at=NOW,
    )


class FakeChain:
    """A chain that answers with one fixed ladder per side."""

    def __init__(self, *, touch: float = 0.0, iv_value: float = 0.55):
        self.touch = touch
        self.iv_value = iv_value
        # Wide enough to bracket a 0.05-delta put and a 0.45-delta call, which
        # the thin-credit and ladder tests both reach for.
        strikes = [1000.0 + 50.0 * step for step in range(49)]  # 1000 .. 3400
        self._books = {
            "put": _book("put", strikes, iv_value=iv_value, touch=touch),
            "call": _book("call", strikes, iv_value=iv_value, touch=touch),
        }

    contract_size = 1.0

    def iv(self, kind: str, dte_days: int, strike: float, spot: float) -> float:
        return self.iv_value

    def book(self, kind: str, dte_days: int, spot: float | None = None):
        return self._books[kind]


# --------------------------------------------------------------------------- #
# Realised vol
# --------------------------------------------------------------------------- #


def test_the_vol_window_is_named_because_the_answer_flips_with_it():
    profile = pricing.volatility_profile(_moving_bars())

    assert profile["primary"] == "21d"
    assert set(profile["windows"]) == set(pricing.VOL_WINDOWS)
    assert profile["value"] == profile["windows"]["21d"]


def test_a_flat_series_has_no_volatility():
    profile = pricing.volatility_profile(_flat_bars(2100))
    assert profile["value"] == pytest.approx(0.0, abs=1e-9)


def test_an_alternating_series_has_volatility():
    closes = [2300.0 + (5.0 if index % 2 else -5.0) for index in range(2100)]
    value = pricing.annualised_vol(_bars(closes), "21d")
    assert value is not None and value > 0.10


def test_no_bars_means_no_measurement_rather_than_zero():
    profile = pricing.volatility_profile([])
    assert profile["value"] is None
    assert profile["bars"] == 0


# --------------------------------------------------------------------------- #
# The edge
# --------------------------------------------------------------------------- #


def test_the_edge_is_the_credit_minus_the_payoff_at_realised_vol():
    chain = FakeChain()
    candidate = pricing.evaluate("put", SPOT, chain, sigma_rv=0.55)

    assert candidate is not None
    years = candidate.dte_actual / 365.0
    expected = candidate.credit - analysis.expected_intrinsic(
        "put", SPOT, candidate.short_strike, candidate.long_strike, years, 0.55
    )
    assert candidate.edge == pytest.approx(expected)


def test_a_lower_realised_vol_makes_the_same_credit_more_attractive():
    """The edge is a statement about vol, so it must move with vol."""
    chain = FakeChain()
    calm = pricing.evaluate("put", SPOT, chain, sigma_rv=0.30)
    wild = pricing.evaluate("put", SPOT, chain, sigma_rv=0.90)

    assert calm is not None and wild is not None
    assert calm.edge > wild.edge
    assert calm.verdict == "TAKE"
    assert wild.verdict == "NO TRADE"


def test_selling_below_the_mark_is_a_real_cost():
    """A wider touch is a worse credit at an identical mark, so a worse edge."""
    tight = pricing.evaluate("put", SPOT, FakeChain(touch=0.001), sigma_rv=0.40)
    wide = pricing.evaluate("put", SPOT, FakeChain(touch=0.02), sigma_rv=0.40)

    assert tight is not None and wide is not None
    assert wide.credit < tight.credit
    assert wide.credit_mark == pytest.approx(tight.credit_mark, rel=1e-9)
    assert wide.edge < tight.edge


def test_the_credit_is_not_the_mark():
    candidate = pricing.evaluate("put", SPOT, FakeChain(touch=0.01), sigma_rv=0.40)
    assert candidate is not None
    assert candidate.credit_source == "book"
    assert candidate.credit < candidate.credit_mark


# --------------------------------------------------------------------------- #
# Probability, not delta
# --------------------------------------------------------------------------- #


def test_the_probability_is_n_d2_and_not_one_minus_delta():
    candidate = pricing.evaluate("put", SPOT, FakeChain(), sigma_rv=0.55, target_delta=0.30)
    assert candidate is not None
    # A put's delta understates P(S_T < K); 1 - delta therefore overstates the
    # probability of keeping the full credit.
    assert candidate.p_keep < 0.70
    # For a put, keeping the credit means settling *above* the short strike.
    assert candidate.p_keep == pytest.approx(
        1.0
        - analysis.probability_below(
            SPOT, candidate.short_strike, candidate.dte_actual / 365.0, candidate.iv_short
        )
    )


def test_the_three_outcomes_partition_the_distribution():
    candidate = pricing.evaluate("put", SPOT, FakeChain(), sigma_rv=0.55)
    assert candidate is not None
    assert candidate.p_keep + candidate.p_partial + candidate.p_max_loss == pytest.approx(1.0)


def test_the_breakeven_rate_is_one_minus_the_credit_ratio():
    """A full win pays the credit, a total loss pays the width; that is the bar."""
    candidate = pricing.evaluate("put", SPOT, FakeChain(), sigma_rv=0.40)
    assert candidate is not None
    assert candidate.breakeven_p == pytest.approx(1.0 - candidate.credit_ratio)


# --------------------------------------------------------------------------- #
# Verdicts name the failing rule
# --------------------------------------------------------------------------- #


def test_a_thin_credit_is_refused_and_says_which_rule():
    candidate = pricing.evaluate(
        "put", SPOT, FakeChain(), sigma_rv=0.55, target_delta=0.05, min_credit_ratio=0.20
    )
    assert candidate is not None
    assert candidate.verdict == "NO TRADE"
    assert "credit" in candidate.reason


def test_a_negative_edge_is_refused_and_says_which_rule():
    candidate = pricing.evaluate("put", SPOT, FakeChain(), sigma_rv=0.90)
    assert candidate is not None
    assert candidate.verdict == "NO TRADE"
    assert "edge" in candidate.reason


def test_both_failing_rules_are_reported_not_just_the_first():
    candidate = pricing.evaluate(
        "put", SPOT, FakeChain(), sigma_rv=0.95, target_delta=0.05, min_credit_ratio=0.20
    )
    assert candidate is not None
    assert "credit" in candidate.reason and "edge" in candidate.reason


def test_an_unknown_side_is_a_programming_error():
    with pytest.raises(ValueError):
        pricing.evaluate("straddle", SPOT, FakeChain(), sigma_rv=0.4)


# --------------------------------------------------------------------------- #
# Strikes
# --------------------------------------------------------------------------- #


def test_both_legs_are_listed_strikes():
    chain = FakeChain()
    for side in ("put", "call"):
        candidate = pricing.evaluate(side, SPOT, chain, sigma_rv=0.55)
        assert candidate is not None
        book = chain.book(side, 30, SPOT)
        assert candidate.short_strike in book.strikes
        assert candidate.long_strike in book.strikes


def test_the_hedge_is_the_next_strike_and_the_expiry_is_the_real_one():
    candidate = pricing.evaluate("put", SPOT, FakeChain(), sigma_rv=0.55)
    assert candidate is not None
    assert candidate.width == pytest.approx(50.0)  # one step on this ladder
    assert candidate.expiry == (NOW + timedelta(days=30)).date().isoformat()
    assert candidate.dte_actual == pytest.approx(30.0, abs=0.1)


# --------------------------------------------------------------------------- #
# The whole shadow block
# --------------------------------------------------------------------------- #


def test_the_shadow_records_both_sides_and_a_ladder():
    block = pricing.shadow(SPOT, FakeChain(), profile=pricing.volatility_profile(_moving_bars()))

    assert block["status"] == "ok"
    assert set(block["sides"]) == {"put", "call"}
    assert len(block["ladder"]) == len(pricing.LADDER_DELTAS) * 2
    assert block["sigma"]["primary"] == "21d"


def test_zero_measured_vol_is_not_a_trade():
    """A flat series is not a free option: it means the measurement failed or
    the market is closed, and a zero sigma prices every spread as free money.
    That is exactly the shape of a bug, so it must refuse rather than fire."""
    block = pricing.shadow(SPOT, FakeChain(), profile=pricing.volatility_profile(_flat_bars(2100)))

    assert block["status"] == "unavailable"
    assert block["verdict"] == "NO TRADE"


def test_the_gate_picks_the_side_with_the_larger_edge():
    block = pricing.shadow(SPOT, FakeChain(), profile=pricing.volatility_profile(_moving_bars()))
    edges = {side: row["edge"] for side, row in block["sides"].items() if "edge" in row}

    assert block["pick"] == max(edges, key=edges.get)


def test_a_disagreement_is_recorded_rather_than_resolved():
    block = pricing.shadow(
        SPOT,
        FakeChain(),
        profile=pricing.volatility_profile(_moving_bars()),
        direction_side="call",
    )
    assert block["pick"] is not None
    assert block["direction_side"] == "call"
    assert block["agrees_with_direction"] in {True, False}


def test_no_direction_candidate_means_no_agreement_claim():
    block = pricing.shadow(
        SPOT, FakeChain(), profile=pricing.volatility_profile(_moving_bars()), direction_side=None
    )
    assert block["agrees_with_direction"] is None


def test_without_bars_there_is_no_edge_to_claim():
    block = pricing.shadow(SPOT, FakeChain(), profile=None, bars=[])

    assert block["status"] == "unavailable"
    assert block["verdict"] == "NO TRADE"
    assert block["pick"] is None
    assert "realised vol" in block["reason"]


def test_the_block_calls_itself_a_shadow():
    block = pricing.shadow(SPOT, FakeChain(), profile=pricing.volatility_profile(_moving_bars()))
    assert block["mode"] == "shadow"
    assert "never emitted" in block["note"]


def test_a_failing_chain_yields_no_candidate_rather_than_an_exception():
    class Broken:
        contract_size = 1.0

        def iv(self, kind, dte_days, strike, spot):
            raise RuntimeError("no chain")

        def book(self, kind, dte_days, spot=None):
            raise RuntimeError("no chain")

    assert pricing.evaluate("put", SPOT, Broken(), sigma_rv=0.4) is None


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def test_the_console_line_says_which_side_and_whether_it_agrees():
    block = pricing.shadow(
        SPOT,
        FakeChain(),
        profile=pricing.volatility_profile(_moving_bars()),
        direction_side="call",
    )
    line = pricing.describe(block)

    assert "pricing" in line
    assert "21d" in line
    assert "agrees" in line.lower() or "DISAGREES" in line


def test_the_console_line_admits_when_nothing_was_measured():
    line = pricing.describe({"status": "unavailable", "reason": "no bars"})
    assert "UNAVAILABLE" in line and "no bars" in line
