"""Offline tests for the live option chain.

Every test runs against a recorded Deribit response in
``tests/fixtures/deribit_eth_chain.json`` and a pinned clock. Nothing here may
touch the network: a pricing input that depends on someone else's uptime is not
testable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import analysis
import iv

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_eth_chain.json"

#: Pinned to the fixture, not to the wall clock: expiry choice and staleness
#: both reason about "now", so "now" is an input.
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def snapshot(recorded: dict) -> iv.ChainSnapshot:
    return iv.build_snapshot(recorded, currency="ETH", now=NOW)


@pytest.fixture
def chain(snapshot: iv.ChainSnapshot) -> iv.DeribitChain:
    return iv.DeribitChain(snapshot, now=NOW)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_instrument_names_are_parsed_without_a_second_endpoint():
    kind, expiry, strike = iv.parse_instrument("ETH-27DEC26-2400-P")

    assert kind == "put"
    assert strike == 2400.0
    assert expiry == datetime(2026, 12, 27, 8, 0, tzinfo=timezone.utc)


def test_calls_are_told_apart_from_puts():
    assert iv.parse_instrument("ETH-27DEC26-2400-C")[0] == "call"


@pytest.mark.parametrize("name", ["ETH-2400-P", "ETH-27DEC26-2400-X", "ETH-99XXX99-2400-P"])
def test_a_malformed_instrument_raises_rather_than_dropping_a_strike(name: str):
    with pytest.raises(iv.ChainError):
        iv.parse_instrument(name)


def test_vols_are_converted_from_percent_to_decimal():
    """Deribit publishes 61.6, Black-Scholes wants 0.616.

    Getting this backwards multiplies every credit by a hundred, so the
    conversion happens once, at the parse boundary, and is asserted here.
    """
    payload = {
        "result": [
            {
                "instrument_name": "ETH-27DEC26-2400-P",
                "mark_iv": 61.6,
                "underlying_price": 2361.5,
                "open_interest": 10,
            }
        ]
    }

    snapshot = iv.build_snapshot(payload, now=NOW)

    assert snapshot.quotes[0].iv == pytest.approx(0.616)
    assert 0.0 < snapshot.quotes[0].iv < 5.0


def test_a_one_sided_quote_is_not_a_price(recorded: dict):
    payload = {
        "result": [
            {
                "instrument_name": "ETH-27DEC26-2400-P",
                "mark_iv": None,
                "bid_iv": None,
                "ask_iv": 40.0,
                "underlying_price": 2361.5,
            },
            {
                "instrument_name": "ETH-27DEC26-2500-P",
                "mark_iv": None,
                "bid_iv": 40.0,
                "ask_iv": 44.0,
                "underlying_price": 2361.5,
            },
        ]
    }

    snapshot = iv.build_snapshot(payload, now=NOW)

    assert len(snapshot.quotes) == 1
    assert snapshot.quotes[0].strike == 2500.0
    assert snapshot.quotes[0].iv == pytest.approx(0.42)  # mid of 40 and 44, as a decimal


def test_an_implausible_vol_is_rejected():
    payload = {
        "result": [
            {
                "instrument_name": "ETH-27DEC26-2400-P",
                "mark_iv": 900.0,
                "underlying_price": 2361.5,
            }
        ]
    }

    with pytest.raises(iv.EmptyChainError):
        iv.build_snapshot(payload, now=NOW)


def test_an_empty_response_is_an_error_not_an_empty_chain():
    with pytest.raises(iv.EmptyChainError):
        iv.build_snapshot({"result": []}, now=NOW)
    with pytest.raises(iv.EmptyChainError):
        iv.build_snapshot({}, now=NOW)


def test_the_snapshot_reads_the_underlying_from_the_chain(snapshot: iv.ChainSnapshot):
    assert snapshot.currency == "ETH"
    assert snapshot.underlying == pytest.approx(2361.58, abs=0.01)
    assert len(snapshot.expiries) == 2


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #


def test_a_quoted_strike_is_returned_exactly(chain: iv.DeribitChain):
    """No smoothing where a real quote exists."""
    quote = chain.snapshot.quotes[0]
    assert chain.iv(quote.kind, 9, quote.strike, chain.snapshot.underlying) == pytest.approx(
        quote.iv, abs=1e-9
    )


def test_a_strike_between_nodes_is_between_them(chain: iv.DeribitChain):
    spot = chain.snapshot.underlying
    low, high = 2300.0, 2350.0
    middle = chain.iv("put", 30, (low + high) / 2, spot)
    edges = (chain.iv("put", 30, low, spot), chain.iv("put", 30, high, spot))

    assert min(edges) <= middle <= max(edges)


def test_a_strike_outside_the_quoted_range_clamps_to_the_wing(chain: iv.DeribitChain):
    """Extrapolating a skew is how a wing that never trades prices a real strike."""
    spot = chain.snapshot.underlying
    far_otm = 1500.0

    assert chain.iv("put", 30, far_otm, spot) == pytest.approx(chain.iv("put", 30, 2100.0, spot))


def test_the_expiry_nearest_the_requested_dte_is_used(chain: iv.DeribitChain):
    spot = chain.snapshot.underlying
    nine = chain.iv("put", 9, 2300.0, spot)
    thirty = chain.iv("put", 30, 2300.0, spot)

    expiries = chain.snapshot.expiries
    assert len(expiries) == 2
    assert nine != pytest.approx(thirty)  # the two tenors really do differ


def test_the_requested_side_selects_its_own_instrument(snapshot: iv.ChainSnapshot):
    """Deribit marks both legs alike, so the sides differ only if the chain does."""
    moment = snapshot.expiries[0]
    quotes = [
        iv.Quote("put", 2400.0, moment, 0.70),
        iv.Quote("call", 2400.0, moment, 0.50),
    ]
    chain = iv.DeribitChain(
        iv.ChainSnapshot("ETH", 2361.0, NOW, tuple(quotes)), now=NOW
    )

    assert chain.iv("put", 9, 2400.0, 2361.0) == pytest.approx(0.70)
    assert chain.iv("call", 9, 2400.0, 2361.0) == pytest.approx(0.50)


def test_a_missing_side_is_an_error_not_a_silent_substitution(snapshot: iv.ChainSnapshot):
    moment = snapshot.expiries[0]
    chain = iv.DeribitChain(
        iv.ChainSnapshot("ETH", 2361.0, NOW, (iv.Quote("put", 2400.0, moment, 0.6),)),
        now=NOW,
    )

    with pytest.raises(iv.EmptyChainError):
        chain.iv("call", 9, 2400.0, 2361.0)


def test_an_unknown_kind_is_rejected(chain: iv.DeribitChain):
    with pytest.raises(iv.ChainError):
        chain.iv("straddle", 30, 2400.0, 2361.0)


# --------------------------------------------------------------------------- #
# Staleness
# --------------------------------------------------------------------------- #


def test_a_stale_chain_is_refused_rather_than_priced(snapshot: iv.ChainSnapshot):
    late = NOW + timedelta(minutes=90)
    chain = iv.DeribitChain(snapshot, max_age_minutes=15, now=late)

    with pytest.raises(iv.StaleChainError):
        chain.iv("put", 30, 2400.0, 2361.0)


def test_a_fresh_chain_is_accepted(snapshot: iv.ChainSnapshot):
    chain = iv.DeribitChain(snapshot, max_age_minutes=15, now=NOW + timedelta(minutes=5))
    assert chain.iv("put", 30, 2400.0, 2361.0) > 0


# --------------------------------------------------------------------------- #
# Payload description
# --------------------------------------------------------------------------- #


def test_a_live_chain_is_labelled_live(chain: iv.DeribitChain):
    described = iv.describe(chain, dte_days=30, spot=chain.snapshot.underlying)

    assert described["source"] == "deribit"
    assert described["quotes"] == len(chain.snapshot.quotes)
    assert described["iv_atm"] == pytest.approx(0.4998, abs=0.05)
    assert "note" not in described


def test_a_synthetic_chain_is_labelled_and_explained():
    described = iv.describe(analysis.SyntheticChain(), dte_days=30, spot=2300.0)

    assert described["source"] == "synthetic"
    assert described["iv_atm"] == pytest.approx(0.60)
    assert "arithmetic" in described["note"]


def test_a_broken_chain_does_not_sink_the_description():
    class Broken:
        def iv(self, kind, dte_days, strike, spot):
            raise iv.ChainError("no quotes")

    described = iv.describe(Broken(), dte_days=30, spot=2300.0)

    assert described["iv_atm"] is None
    assert "no quotes" in described["iv_atm_error"]


# --------------------------------------------------------------------------- #
# The planner actually uses it
# --------------------------------------------------------------------------- #


def test_the_plan_is_priced_off_the_live_chain(chain: iv.DeribitChain):
    spot = chain.snapshot.underlying
    plan, reason = analysis.spread_plan(
        analysis.BULLISH, spot, chain, target_delta=0.30, dte_days=30, equity=1_000_000
    )

    assert reason == "ok"
    assert plan is not None
    # The short-leg vol in the plan is the chain's vol at that strike, not a stub.
    assert plan.iv_short == pytest.approx(chain.iv("put", 30, plan.short_strike, spot), abs=1e-6)
    assert plan.credit > 0


def test_a_live_chain_changes_the_credit_against_the_stub(chain: iv.DeribitChain):
    spot = chain.snapshot.underlying
    live, _ = analysis.spread_plan(analysis.BULLISH, spot, chain, dte_days=30)
    stub, _ = analysis.spread_plan(
        analysis.BULLISH, spot, analysis.SyntheticChain(), dte_days=30
    )

    assert live is not None and stub is not None
    assert live.credit != pytest.approx(stub.credit)


# --------------------------------------------------------------------------- #
# The ladder: what exists, and what it bids
# --------------------------------------------------------------------------- #


def test_quotes_carry_a_market_not_only_a_vol(recorded: dict):
    quotes, _ = iv.parse_quotes(recorded["result"])

    assert any(quote.two_sided for quote in quotes)
    for quote in quotes:
        assert quote.mark >= 0.0
        if quote.two_sided:
            assert 0.0 < quote.bid <= quote.ask


def test_a_missing_bid_stays_missing_instead_of_becoming_a_mid():
    quotes, _ = iv.parse_quotes(
        [
            {
                "instrument_name": "ETH-27DEC26-2400-P",
                "mark_iv": 60.0,
                "mark_price": 0.05,
            }
        ]
    )

    assert quotes[0].mark == pytest.approx(0.05)
    assert quotes[0].bid == 0.0
    assert not quotes[0].two_sided


def test_a_book_holds_one_expiry_and_only_strikes_that_exist(chain: iv.DeribitChain):
    book = chain.book("put", 30, chain.snapshot.underlying)

    assert book.strikes == sorted(book.strikes)
    assert len(set(book.strikes)) == len(book.strikes)
    assert all(quote.kind == "put" and quote.expiry == book.expiry for quote in book.legs)
    assert book.source == "deribit"


def test_the_book_reports_the_dte_it_actually_priced(chain: iv.DeribitChain):
    """A requested 30 days is rarely a listed expiry; the real one is reported."""
    book = chain.book("put", 30, chain.snapshot.underlying)

    assert 10.0 <= book.days_to_expiry() <= 60.0
    assert book.describe()["expiry"] == book.expiry.date().isoformat()


def test_a_stale_chain_refuses_to_supply_a_book(snapshot: iv.ChainSnapshot):
    late = iv.DeribitChain(snapshot, max_age_minutes=15, now=NOW + timedelta(minutes=90))

    with pytest.raises(iv.StaleChainError):
        late.book("put", 30, snapshot.underlying)


def test_the_plan_is_priced_at_the_touch_not_the_mark(chain: iv.DeribitChain):
    """The credit on screen must be one a seller could take, not a mid."""
    spot = chain.snapshot.underlying
    plan, reason = analysis.spread_plan(analysis.BULLISH, spot, chain, dte_days=30)

    assert reason == "ok" and plan is not None
    book = chain.book("put", 30, spot)
    short = book.at(plan.short_strike)
    long_ = book.at(plan.long_strike)

    assert short is not None and long_ is not None
    assert plan.credit_source == "book"
    expected = (short.bid - long_.ask) * spot
    assert plan.credit == pytest.approx(expected, rel=1e-9)
    assert plan.credit < plan.credit_mark


def test_the_plan_is_priced_in_the_chains_frame_not_the_callers(chain: iv.DeribitChain):
    """The chain's underlying is the frame its strikes mean.

    A caller's spot comes from a different venue at a different second; a 2.6%
    gap (as the fixture has) moves a 0.30-delta strike by a whole rung, so the
    plan must not silently choose strikes relative to it.
    """
    spot = chain.snapshot.underlying
    same, _ = analysis.spread_plan(analysis.BULLISH, spot, chain, dte_days=30)
    skewed, _ = analysis.spread_plan(analysis.BULLISH, spot * 0.974, chain, dte_days=30)

    assert same is not None and skewed is not None
    assert skewed.short_strike == same.short_strike
    assert any("chain underlying" in note for note in skewed.notes)
