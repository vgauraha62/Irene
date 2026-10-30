"""Offline tests for the analysis layer.

No network, no fixtures on disk: everything here is arithmetic that must hold.
The first test is the regression test for the bug that made every previous
signal bullish-biased.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import analysis
import iv
from analysis import (
    BEARISH,
    BULLISH,
    MA_KEYS,
    NEUTRAL,
    StaleDataError,
    SyntheticChain,
    TIMEFRAME_SECONDS,
)


@dataclass
class Fake:
    """Stand-in for indicators.Snapshot / a tradingview Analysis."""

    timeframe: str
    indicators: dict
    as_of: datetime = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    as_of_source: str = "exchange"

    def age_seconds(self, now=None):
        now = now or datetime.now(timezone.utc)
        return (now - self.as_of).total_seconds()


def ladder(timeframe: str, spot: float, pct_per_hour: float, *, close: float | None = None) -> Fake:
    """An EMA ladder for a market *trending* at ``pct_per_hour`` %/hour.

    A rising market puts the long-lookback EMAs **below** spot, because they
    average older, cheaper prices. So the ladder's %-of-spot value *decreases*
    with lookback and the raw ladder slope is the negative of the trend rate.
    ``analysis.drift_pct_per_hour`` is the function that returns
    ``pct_per_hour`` unchanged.

    Kept linear on purpose: a linear ladder has a constant gradient, so the
    compared-to-hours invariant is exact rather than approximate.
    """
    hours = analysis.ema_horizons_hours(timeframe)
    values = {k: spot * (1.0 - pct_per_hour / 100.0 * h) for k, h in zip(MA_KEYS, hours)}
    values["close"] = close if close is not None else spot
    return Fake(timeframe=timeframe, indicators=values)


def flat_ladder(timeframe: str, spot: float, adx: float | None = 30.0) -> Fake:
    values = {k: spot for k in MA_KEYS}
    values["close"] = spot
    if adx is not None:
        values["ADX"] = adx
    return Fake(timeframe=timeframe, indicators=values)


def book(*, adx: float | None = 30.0, **rates: float) -> dict[str, Fake]:
    """Build a 15m/30m/4h snapshot book from %/hour rates."""
    spot = 2300.0
    out = {}
    for tf in ("15m", "30m", "4h"):
        fake = ladder(tf, spot, rates[tf])
        if adx is not None:
            fake.indicators["ADX"] = adx
        out[tf] = fake
    return out


# --------------------------------------------------------------------------- #
# The unit bug
# --------------------------------------------------------------------------- #


def test_slopes_are_comparable_across_timeframes():
    """Equal %/hour must produce equal slopes at 15m and 4h.

    This is the regression test for the original unit defect: ``np.gradient``
    with the default spacing of one *bar* made a 4h slope 16x smaller than an
    identical 15m slope, so ``grad_15m > grad_4h`` was almost always true and
    the agent reported a permanent bullish shift.

    The ladder slope is the *negated* trend rate (see ``ladder``); the drift is
    the rate itself.
    """
    rate = 0.05  # %/hour of price trend
    slope_15m = analysis.slopes(ladder("15m", 2300.0, rate))
    slope_4h = analysis.slopes(ladder("4h", 2300.0, rate))

    assert np.allclose(slope_15m, -rate, atol=1e-9)
    assert np.allclose(slope_4h, -rate, atol=1e-9)
    assert np.allclose(slope_15m, slope_4h, atol=1e-9)

    assert analysis.drift_pct_per_hour(ladder("15m", 2300.0, rate)) == pytest.approx(rate)
    assert analysis.drift_pct_per_hour(ladder("4h", 2300.0, rate)) == pytest.approx(rate)


# --------------------------------------------------------------------------- #
# Sign: a ladder slope is not a trend direction
# --------------------------------------------------------------------------- #


def _ema_of_series(prices, period):
    alpha = 2.0 / (period + 1.0)
    value = prices[0]
    for price in prices[1:]:
        value = alpha * price + (1.0 - alpha) * value
    return value


def _trending_book(rate_pct_per_hour: float, bars: int = 1200, spot: float = 2300.0) -> dict:
    """A snapshot book from a real price series trending at ``rate_pct_per_hour``.

    Unlike ``ladder``, this runs the actual EMA recursion, so it is the honest
    check that the regime reads a genuine trend the right way round. The series
    is exponential, which is the only way a ±0.3 %/hour trend stays positive
    over a 1200-bar window.
    """
    out = {}
    for timeframe in ("15m", "30m", "4h"):
        seconds = TIMEFRAME_SECONDS[timeframe]
        hours = np.arange(bars) * seconds / 3600.0
        prices = spot * np.exp(rate_pct_per_hour / 100.0 * hours)
        indicators = {
            key: _ema_of_series(prices, period)
            for key, period in zip(MA_KEYS, analysis.MA_PERIODS)
        }
        indicators["close"] = prices[-1]
        indicators["ADX"] = 35.0
        out[timeframe] = Fake(timeframe=timeframe, indicators=indicators)
    return out


def test_ladder_slope_and_drift_have_opposite_signs():
    """The trap that inverted every call: a rise has a *negative* ladder slope."""
    rising = ladder("15m", 2300.0, 0.20)
    assert np.median(analysis.slopes(rising)) < 0
    assert analysis.drift_pct_per_hour(rising) > 0


def test_drift_agrees_with_the_spectral_spread():
    """Two independent readings of one ladder must not contradict each other.

    Before the sign fix the payload reported a drift of +0.010 %/h next to a
    spectral spread of -5.39%: an EMA10 above EMA200 described as a rising
    trend. Comparing these two numbers is a free consistency check.
    """
    for rate in (0.25, -0.25):
        fake = ladder("4h", 2300.0, rate)
        spread = analysis.term_structure(fake)["spectral_spread_pct"]
        assert np.sign(analysis.drift_pct_per_hour(fake)) == np.sign(spread)


def test_rising_market_is_bullish():
    reg = analysis.regime(_trending_book(+0.05))
    assert reg.direction == BULLISH, reg.reasons
    assert reg.macro_dir == analysis.MACRO_BULL
    assert reg.agreement == 3


def test_falling_market_is_bearish():
    reg = analysis.regime(_trending_book(-0.05))
    assert reg.direction == BEARISH, reg.reasons
    assert reg.macro_dir == analysis.MACRO_BEAR
    assert reg.agreement == 3


def test_rung_slopes_expose_the_shape():
    slopes = analysis.rung_slopes(ladder("4h", 2300.0, 0.10))
    assert set(slopes) == set(MA_KEYS)
    # A linear ladder has a flat gradient: same slope at the front and the back.
    assert slopes["EMA10"] == pytest.approx(slopes["EMA200"], rel=1e-6)


def test_per_bar_gradient_was_biased_by_bar_length():
    """Documents the size of the old defect: a 16x spurious difference."""
    rate = -0.05
    naive_15m = np.median(np.gradient(analysis.ema_values(ladder("15m", 2300.0, rate))))
    naive_4h = np.median(np.gradient(analysis.ema_values(ladder("4h", 2300.0, rate))))

    assert TIMEFRAME_SECONDS["4h"] / TIMEFRAME_SECONDS["15m"] == 16
    assert naive_4h / naive_15m == pytest.approx(16.0, rel=1e-6)
    # Same market, same rate. Only the horizon differs, so the fixed version agrees.
    assert analysis.slope_summary(ladder("4h", 2300.0, rate))["median"] == pytest.approx(
        analysis.slope_summary(ladder("15m", 2300.0, rate))["median"], rel=1e-9
    )


def test_missing_indicator_is_named():
    fake = Fake("15m", {"close": 2300.0, "EMA10": 2300.0})
    with pytest.raises(analysis.MissingIndicatorError, match="EMA20"):
        analysis.ema_values(fake)


def test_non_positive_spot_rejected():
    fake = Fake("15m", {**{k: 0.0 for k in MA_KEYS}, "close": 0.0})
    with pytest.raises(ValueError):
        analysis.slopes(fake)


def test_unknown_timeframe_rejected():
    with pytest.raises(KeyError):
        analysis.ema_horizons_hours("7m")


def test_term_structure_is_scale_free():
    small = analysis.term_structure(ladder("15m", 200.0, 0.1))
    large = analysis.term_structure(ladder("15m", 2000.0, 0.1))
    assert small["spectral_spread_pct"] == pytest.approx(large["spectral_spread_pct"])


# --------------------------------------------------------------------------- #
# Velocity / acceleration
# --------------------------------------------------------------------------- #


def test_velocity_of_linear_series_is_constant_and_acceleration_zero():
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    velocity, acceleration = analysis.velocity_and_acceleration(prices, 3600.0)
    assert velocity == pytest.approx(1.0, rel=1e-6)  # %/hour
    assert acceleration == pytest.approx(0.0, abs=1e-9)


def test_acceleration_of_convex_series_is_positive():
    prices = [100.0, 101.0, 103.0, 106.0, 110.0]
    velocity, acceleration = analysis.velocity_and_acceleration(prices, 3600.0)
    assert velocity > 0
    assert acceleration > 0


def test_velocity_needs_enough_points():
    with pytest.raises(ValueError):
        analysis.velocity_and_acceleration([100.0, 101.0], 3600.0)


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #


def test_flat_market_is_neutral():
    snaps = {tf: flat_ladder(tf, 2300.0) for tf in ("15m", "30m", "4h")}
    reg = analysis.regime(snaps)
    assert reg.direction == NEUTRAL
    assert reg.macro_dir == analysis.MACRO_FLAT
    assert reg.confidence == pytest.approx(0.0)


def test_full_agreement_is_directional():
    reg = analysis.regime(book(**{"15m": -0.10, "30m": -0.08, "4h": -0.06}))
    assert reg.direction == BEARISH
    assert reg.agreement == 3
    assert reg.macro_dir == analysis.MACRO_BEAR
    assert reg.confidence > 0.5

    reg = analysis.regime(book(**{"15m": 0.10, "30m": 0.08, "4h": 0.06}))
    assert reg.direction == BULLISH


def test_disagreement_stands_down():
    """Macro up, micro down. The old `or` condition still fired a bullish label."""
    reg = analysis.regime(book(**{"15m": -0.10, "30m": 0.08, "4h": 0.06}))
    assert reg.direction == NEUTRAL
    assert reg.agreement == 2
    assert any("disagree" in reason for reason in reg.reasons)


def test_chop_filter_blocks_direction():
    reg = analysis.regime(book(adx=12.0, **{"15m": 0.10, "30m": 0.08, "4h": 0.06}))
    assert reg.direction == NEUTRAL
    assert any("ADX" in reason for reason in reg.reasons)


def test_chop_filter_can_be_disabled():
    reg = analysis.regime(
        book(adx=12.0, **{"15m": 0.10, "30m": 0.08, "4h": 0.06}), use_chop_filter=False
    )
    assert reg.direction == BULLISH


def test_regime_requires_all_timeframes():
    with pytest.raises(KeyError):
        analysis.regime({"15m": flat_ladder("15m", 2300.0)})


# --------------------------------------------------------------------------- #
# Staleness
# --------------------------------------------------------------------------- #


def test_stale_data_refused():
    old = Fake("15m", {**{k: 2300.0 for k in MA_KEYS}, "close": 2300.0},
               as_of=datetime.now(timezone.utc) - timedelta(hours=5))
    with pytest.raises(StaleDataError):
        analysis.guard({"15m": old}, max_age_seconds=3600)


def test_fresh_data_passes_guard():
    now = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    fresh = Fake("4h", {**{k: 2300.0 for k in MA_KEYS}, "close": 2300.0},
                 as_of=now - timedelta(minutes=5))
    analysis.guard({"4h": fresh}, max_age_seconds=3600, now=now)


# --------------------------------------------------------------------------- #
# Expected price
# --------------------------------------------------------------------------- #


def test_projection_points_up_in_a_rising_market():
    projection = analysis.project_price(
        book(**{"15m": 0.50, "30m": 0.50, "4h": 0.50}), horizon_hours=4.0
    )
    assert projection.direction == BULLISH
    assert projection.expected_price > projection.spot
    assert projection.low < projection.expected_price < projection.high
    assert projection.timeframe == "4h"


def test_projection_points_down_in_a_falling_market():
    projection = analysis.project_price(
        book(**{"15m": -0.50, "30m": -0.50, "4h": -0.50}), horizon_hours=4.0
    )
    assert projection.direction == BEARISH
    assert projection.expected_price < projection.spot


def test_projection_refuses_a_direction_inside_the_band():
    """A move smaller than the band is not a direction, it is noise."""
    projection = analysis.project_price(
        book(**{"15m": 0.02, "30m": 0.02, "4h": 0.02}), horizon_hours=0.5
    )
    assert projection.direction == NEUTRAL
    assert projection.expected_price > projection.spot  # the number still moves
    assert "straddles spot" in projection.invalidates


def test_projection_horizon_picks_the_matching_ladder():
    book_ = _trending_book(+0.10)
    assert analysis.project_price(book_, horizon_hours=0.5).timeframe == "15m"
    assert analysis.project_price(book_, horizon_hours=2.0).timeframe == "30m"
    assert analysis.project_price(book_, horizon_hours=4.0).timeframe == "4h"


def test_flat_market_projection_is_neutral():
    snaps = {tf: flat_ladder(tf, 2300.0) for tf in ("15m", "30m", "4h")}
    projection = analysis.project_price(snaps, horizon_hours=4.0)
    assert projection.direction == NEUTRAL
    assert projection.expected_price == pytest.approx(2300.0)
    assert "straddles spot" in projection.invalidates


def test_projection_band_widens_in_chop():
    book_ = _trending_book(+0.10)
    calm = analysis.project_price(book_, horizon_hours=4.0)
    for snapshot in book_.values():
        snapshot.indicators["ADX"] = 8.0
    choppy = analysis.project_price(book_, horizon_hours=4.0)

    assert choppy.band_pct > calm.band_pct
    assert any("chop" in note for note in choppy.notes)


def test_projection_acceleration_bends_the_estimate():
    flat = analysis.project_price(_trending_book(+0.10), horizon_hours=4.0)
    bent = analysis.project_price(
        _trending_book(+0.10), horizon_hours=4.0, acceleration_pct_per_hour2=0.05
    )
    assert bent.expected_price > flat.expected_price
    assert bent.acceleration_pct_per_hour2 == pytest.approx(0.05)
    assert any("first-order" in note for note in flat.notes)


def test_every_projection_is_labelled_a_diagnostic():
    """AGENTS.md invariant 7: never present an unscored estimate as a forecast."""
    for projection in analysis.project_all(_trending_book(+0.10)):
        assert any("never scored" in note for note in projection.notes)
        assert projection.invalidates


def test_project_all_covers_the_published_horizons():
    projections = analysis.project_all(_trending_book(+0.10))
    assert [p.horizon_label for p in projections] == ["30m", "4h"]
    assert all(p.to_dict()["expected_price"] > 0 for p in projections)


def test_projection_rejects_a_bad_horizon():
    with pytest.raises(ValueError):
        analysis.project_price(_trending_book(+0.10), horizon_hours=0)


def test_projection_requires_all_timeframes():
    with pytest.raises(KeyError):
        analysis.project_price({"15m": flat_ladder("15m", 2300.0)}, horizon_hours=4.0)


# --------------------------------------------------------------------------- #
# Spread planner
# --------------------------------------------------------------------------- #


def test_bull_put_spread_is_defined_risk_and_credit():
    # A 1% risk budget of 1m equity affords a couple of 50-wide contracts.
    # Delta-anchored: this test exercises the risk/credit structure, not the ATM pick.
    plan, reason = analysis.spread_plan(
        BULLISH, 2300.0, SyntheticChain(), target_delta=0.30, dte_days=30, equity=1_000_000, anchor="delta"
    )
    assert reason == "ok"
    assert plan is not None
    assert plan.strategy == "bull_put_spread"
    assert plan.short_strike < 2300.0                      # short put is OTM
    assert plan.long_strike == pytest.approx(plan.short_strike - plan.width)
    assert 0 < plan.credit < plan.width                    # a credit, less than width
    assert plan.max_loss_per_contract == pytest.approx(plan.width - plan.credit)
    assert plan.breakeven < plan.short_strike              # put breakeven is below the short strike
    # POP is N(-d2) at the short strike, not 1 - delta: the middle band is a
    # real outcome and it costs part of the credit, so the three zones are
    # reported separately and they partition the distribution.
    assert plan.pop < 0.70                                 # strictly below 1 - target_delta
    assert plan.pop == pytest.approx(1.0 - plan.pop_partial - plan.pop_max_loss)
    assert 0.30 < plan.pop_max_loss < 0.42                 # below the hedge is a real tail
    assert plan.contracts > 0
    assert plan.total_max_loss <= 1_000_000 * 0.01 + 1e-9  # risk budget respected


def test_bear_call_spread_is_a_credit_above_spot():
    plan, reason = analysis.spread_plan(
        BEARISH, 2300.0, SyntheticChain(), target_delta=0.30, dte_days=30
    )
    assert reason == "ok"
    assert plan is not None
    assert plan.strategy == "bear_call_spread"
    assert plan.short_strike > 2300.0
    assert plan.long_strike == pytest.approx(plan.short_strike + plan.width)
    assert plan.breakeven > plan.short_strike
    assert plan.credit > 0


def test_neutral_regime_never_produces_a_trade():
    plan, reason = analysis.spread_plan(NEUTRAL, 2300.0, SyntheticChain())
    assert plan is None
    assert "neutral" in reason


def test_thin_credit_is_refused():
    # Thin credit is a property of a far-OTM leg: test it on the delta anchor.
    plan, reason = analysis.spread_plan(
        BULLISH, 2300.0, SyntheticChain(), target_delta=0.02, dte_days=7, anchor="delta"
    )
    assert plan is None
    assert "credit" in reason.lower()


def test_the_atm_anchored_candidate_is_not_thin():
    """The production candidate sells the listed strike nearest the spot, so its
    credit must clear the floor at a sane delta -- a far-OTM leg is what made it thin."""
    plan, reason = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(), dte_days=30)
    assert reason == "ok"
    assert plan is not None
    assert plan.short_strike < 2300.0
    assert plan.credit > 0
    assert plan.credit_ratio > analysis.DEFAULT_MIN_CREDIT_RATIO


def test_high_vol_widens_strikes():
    calm, _ = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(base_iv=0.30), anchor="delta")
    wild, _ = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(base_iv=0.90), anchor="delta")
    assert wild is not None and calm is not None
    assert wild.short_strike < calm.short_strike  # further OTM for the same delta


def test_the_atm_anchored_candidate_sells_near_the_money():
    """The candidate sells the listed strike nearest the spot; the hedge is one
    step further out, so the width stays one step and the credit is the one that
    matters for the credit gate."""
    bear, reason = analysis.spread_plan(BEARISH, 2300.0, SyntheticChain(), dte_days=30)
    assert reason == "ok" and bear is not None
    assert bear.short_strike > 2300.0                      # short call is OTM
    assert bear.long_strike == pytest.approx(bear.short_strike + bear.width)

    bull, reason = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(), dte_days=30)
    assert reason == "ok" and bull is not None
    assert bull.short_strike < 2300.0                      # short put is OTM
    assert bull.long_strike == pytest.approx(bull.short_strike - bull.width)


def test_an_unknown_strike_anchor_is_refused():
    with pytest.raises(ValueError):
        analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(), anchor="median")


def test_position_size_respects_budget():
    # width 50, credit 10 -> max loss 4000/contract.
    assert analysis.position_size(10_000, 0.01, 50, 10) == 0
    assert analysis.position_size(1_000_000, 0.01, 50, 10) == 2


def test_position_size_rejects_credit_above_width():
    with pytest.raises(ValueError):
        analysis.position_size(10_000, 0.01, 10, 12)


def test_black_scholes_put_call_parity():
    from analysis import bs_price

    S, K, T, sigma, r = 2300.0, 2300.0, 30 / 365, 0.6, 0.0
    call = bs_price("call", S, K, T, sigma, r)
    put = bs_price("put", S, K, T, sigma, r)
    assert call - put == pytest.approx(S - K, abs=1e-6)


def test_strike_for_delta_recovers_delta():
    from analysis import bs_price, strike_for_delta

    S, T, sigma = 2300.0, 30 / 365, 0.6
    strike = strike_for_delta("put", 0.30, S, T, sigma)
    assert strike < S
    # Black-Scholes put delta via finite difference of the price.
    h = 0.01
    delta = (bs_price("put", S + h, strike, T, sigma) - bs_price("put", S - h, strike, T, sigma)) / (2 * h)
    assert abs(delta) == pytest.approx(0.30, abs=1e-3)


def test_trade_scenario_shape():
    reg = analysis.regime(book(**{"15m": 0.10, "30m": 0.08, "4h": 0.06}))
    result = analysis.trade_scenario(reg, 2300.0, SyntheticChain(), equity=1_000_000)
    assert result["decision"] == "CANDIDATE"
    assert result["plan"]["strategy"] == "bull_put_spread"
    assert result["plan"]["contracts"] > 0
    assert "regime" in result and "disclaimer" in result


def test_unaffordable_plan_is_not_a_candidate():
    """A structure the risk budget cannot size is a no-trade, not a candidate."""
    plan, reason = analysis.spread_plan(
        BULLISH, 2300.0, SyntheticChain(), equity=2_000, max_risk_pct=0.01
    )
    assert plan is None
    assert "budget" in reason


# --------------------------------------------------------------------------- #
# Warm-up guard
# --------------------------------------------------------------------------- #


def test_warmup_guard():
    assert analysis.warmup_sufficient(30, 5)
    assert not analysis.warmup_sufficient(30, 200)
    with pytest.raises(ValueError):
        analysis.warmup_sufficient(30, 0)


# --------------------------------------------------------------------------- #
# Where the credit actually comes from
# --------------------------------------------------------------------------- #


def test_the_short_leg_is_a_listed_strike_not_an_interpolation():
    """A 2211 put cannot be sold; the ladder only has 2200 and 2250."""
    chain = SyntheticChain()
    plan, reason = analysis.spread_plan(BULLISH, 2300.0, chain, target_delta=0.30)

    assert reason == "ok" and plan is not None
    book = chain.book("put", 30, 2300.0)
    assert plan.short_strike in book.strikes
    assert plan.long_strike in book.strikes


def test_the_credit_is_what_a_seller_receives_not_the_mark():
    plan, _ = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain())

    assert plan is not None
    assert plan.credit_source == "book"
    assert plan.credit < plan.credit_mark  # sell the bid, buy the ask


def test_a_one_sided_market_is_priced_as_a_mark_and_says_so():
    class OneSided(SyntheticChain):
        def book(self, kind, dte_days, spot=None):
            book = super().book(kind, dte_days, spot)
            return iv.Book(
                kind=book.kind,
                expiry=book.expiry,
                spot=book.spot,
                legs=tuple(
                    iv.Quote(
                        kind=leg.kind,
                        strike=leg.strike,
                        expiry=leg.expiry,
                        iv=leg.iv,
                        open_interest=leg.open_interest,
                        bid=0.0,
                        ask=0.0,
                        mark=leg.mark,
                    )
                    for leg in book.legs
                ),
                observed_at=book.observed_at,
                source=book.source,
            )

    plan, reason = analysis.spread_plan(BULLISH, 2300.0, OneSided())

    assert reason == "ok" and plan is not None
    assert plan.credit_source == "mark"
    assert any("two-sided" in note for note in plan.notes)


def test_the_expiry_actually_priced_travels_with_the_plan():
    """The request is 30 days; the chain rarely has one exactly there."""
    plan, _ = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(), dte_days=30)

    assert plan is not None
    assert plan.expiry is not None
    assert plan.dte_actual == pytest.approx(30.0, abs=0.1)


def test_a_crypto_contract_is_one_unit_not_one_hundred():
    """Deribit ETH options are 1 ETH. The 100x equity multiplier was a 100x error."""
    plan, _ = analysis.spread_plan(BULLISH, 2300.0, SyntheticChain(), equity=1_000_000)

    assert plan is not None
    assert plan.contract_size == 1.0
    assert plan.total_max_loss == pytest.approx(
        plan.max_loss_per_contract * plan.contracts, rel=1e-9
    )
    assert plan.total_max_loss <= 1_000_000 * 0.01 + 1e-6


def test_the_zones_partition_the_distribution_for_both_sides():
    for kind in ("put", "call"):
        short = 2200.0 if kind == "put" else 2400.0
        long_ = 2150.0 if kind == "put" else 2450.0
        keep, partial, lose = analysis.zone_probabilities(
            kind, 2300.0, short, long_, 30 / 365, 0.55
        )
        assert keep + partial + lose == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in (keep, partial, lose))


def test_the_expected_payoff_is_the_short_leg_minus_the_long_leg():
    put = analysis.expected_intrinsic("put", 2300.0, 2200.0, 2150.0, 30 / 365, 0.5)
    call = analysis.expected_intrinsic("call", 2300.0, 2400.0, 2450.0, 30 / 365, 0.5)

    assert put > 0 and put < 50.0
    assert call > 0 and call < 50.0
