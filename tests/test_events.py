"""Offline tests for the label and base-rate layer.

The two properties worth defending here are the ones that make the numbers
believable at all: **no lookahead**, and **an interval that admits how little
sample there is**. Everything else is arithmetic.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

import analysis
import bars
import events


def series(count: int = 700, *, drift: float = 0.0, wobble: float = 0.0015) -> list[bars.Bar]:
    """A deterministic 15m series: optional drift, deterministic wobble."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 2000.0
    out: list[bars.Bar] = []
    for index in range(count):
        previous = price
        price = round(price * (1.0 + drift + wobble * math.sin(index / 5.0)), 4)
        moment = start + timedelta(minutes=15 * index)
        out.append(
            bars.Bar(
                open_time=moment,
                open=previous,
                high=round(max(previous, price) * 1.0008, 4),
                low=round(min(previous, price) * 0.9992, 4),
                close=price,
                volume=100.0 + index % 5,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# No lookahead
# --------------------------------------------------------------------------- #


def test_state_at_cannot_see_the_future():
    """Change everything after index i; the state at i must not move by a tick."""
    store = series(700)
    honest = events.build_ladder(store)

    tampered_tail = [
        bars.Bar(
            open_time=bar.open_time,
            open=bar.open * 2,
            high=bar.high * 2,
            low=bar.low * 2,
            close=bar.close * 2,
            volume=bar.volume,
        )
        for bar in store[301:]
    ]
    tampered = events.build_ladder(store[:301] + tampered_tail)

    before = honest.snapshots_at(300)
    after = tampered.snapshots_at(300)

    assert before == after


def test_sigma_is_ex_ante(monkeypatch):
    """The threshold must not become easier because the future turned out calm."""
    store = series(400)
    spec = events.Spec(horizon_bars=16, vol_lookback=200)

    honest = events.sigma_pct(store, 300, spec)
    tampered = events.sigma_pct(store[:301] + series(99, drift=0.01), 300, spec)

    assert honest is not None
    assert tampered == honest


def test_forward_return_spans_exactly_the_horizon():
    store = series(64)
    value = events.forward_return_pct(store, 10, 16)

    expected = 100.0 * math.log(store[26].close / store[10].close)
    assert value == pytest.approx(expected)
    assert events.forward_return_pct(store, 60, 16) is None  # runs off the end


def test_a_rising_series_is_labelled_up():
    store = series(400, drift=0.004)
    spec = events.Spec()

    forward = events.forward_return_pct(store, 300, spec.horizon_bars)
    sigma = events.sigma_pct(store, 300, spec)

    assert forward is not None and sigma is not None
    assert events.classify(forward, sigma, spec) == "up"


def test_classify_respects_the_sigma_multiple():
    spec = events.Spec(size_sigma=0.5)

    assert events.classify(0.49, 1.0, spec) == "flat"
    assert events.classify(0.50, 1.0, spec) == "up"
    assert events.classify(-0.51, 1.0, spec) == "down"


# --------------------------------------------------------------------------- #
# State comes from the live maths
# --------------------------------------------------------------------------- #


def test_the_state_is_the_live_regime_not_a_second_implementation(long_store):
    ladder = events.build_ladder(long_store)
    index = len(long_store) - 20

    snapshots = ladder.snapshots_at(index)
    state = ladder.state_at(index)

    assert set(snapshots) == {"15m", "30m", "4h"}
    assert snapshots["15m"]["indicators"]["close"] == ladder.series["15m"][index].close
    assert state is not None
    direct = analysis.regime(snapshots)
    assert (state.direction, state.agreement) == (direct.direction, direct.agreement)
    assert state.as_of


def test_a_ladder_without_enough_history_returns_no_state():
    ladder = events.build_ladder(series(400))

    assert ladder.state_at(350) is None


# --------------------------------------------------------------------------- #
# Rates and intervals
# --------------------------------------------------------------------------- #


def test_wilson_interval_brackets_the_point_estimate():
    low, high = events.wilson_interval(30, 100)

    assert low < 30 < high
    assert 20 < low < 24
    assert 37 < high < 41


def test_a_small_sample_gets_a_wide_interval():
    """8.3% of 12 is not 8.3% -- the interval has to say so."""
    low, high = events.wilson_interval(1, 12)

    assert high - low > 20
    assert low < 2


def test_an_empty_bucket_does_not_invent_a_rate():
    assert events.wilson_interval(0, 0) == (0.0, 100.0)
    assert events._rate(0, 0)["rate"] is None


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_rates_sum_to_one_hundred(long_store):
    ladder = events.build_ladder(long_store)
    samples = events.sample(ladder, events.Spec(adx_period=14))

    assert samples
    base = events.base_rate(samples)
    total = base["up"]["rate"] + base["down"]["rate"] + base["flat"]["rate"]

    assert total == pytest.approx(100.0, abs=0.05)
    assert base["up"]["n"] == base["down"]["n"] == base["flat"]["n"] == len(samples)


def test_overlapping_samples_are_flagged(stub_store):
    report = events.table(stub_store, events.Spec(horizon_bars=16, stride=4))

    assert any("overlap" in warning for warning in report["warnings"])
    assert report["spec"]["overlapping"] is True


def test_short_history_fails_with_an_actionable_message():
    with pytest.raises(ValueError, match="warmup needs"):
        events.table(series(300))


def test_the_split_is_chronological_and_gapless():
    train, test = events.purged_split(100, test_fraction=0.3, gap=2)

    assert train and test
    assert max(train) < min(test)
    assert min(test) == 70
    assert max(train) == 67  # two samples removed at the seam
    assert not set(train) & set(test)


def test_a_split_that_leaks_fails_loudly():
    """Without the gap, a training label's horizon reaches into the test block."""
    with pytest.raises(ValueError, match="past the first test sample"):
        events.assert_disjoint(range(0, 71), range(70, 100), stride=16, horizon=16)

    # Adjacent is fine: the label window ends exactly where the test begins.
    events.assert_disjoint(range(0, 70), range(70, 100), stride=16, horizon=16)


# --------------------------------------------------------------------------- #
# Fixtures and rendering
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def long_store() -> list[bars.Bar]:
    """~20,000 bars: enough for the 4h EMA200 to converge and produce samples.

    A gentle drift so the agreement gate actually fires: a perfectly flat series
    produces no directional state, and a test that only ever exercises the empty
    branch proves nothing about the branch that matters.
    """
    return series(20_500, drift=0.0002)


@pytest.fixture(scope="module")
def stub_store(long_store) -> list[bars.Bar]:
    return long_store


def test_the_table_reports_a_base_rate_and_a_signal_bucket(stub_store):
    report = events.table(stub_store, events.Spec(horizon_bars=16))

    assert report["base"]["n"] > 0
    assert set(report["buckets"]) == {"bullish", "bearish", "neutral", "signal"}
    assert report["warmup_bars"] > 0
    assert report["spec"]["horizon_hours"] == 4.0


def test_rendering_says_what_the_lift_means(stub_store):
    text = events.render(events.table(stub_store, events.Spec(horizon_bars=16)))

    assert "base rate" in text
    assert "means the signal is worth exactly what happens anyway" in text
    assert "No predictor is fitted here" in text


def test_the_report_is_json_serialisable(stub_store):
    import json

    report = events.table(stub_store, events.Spec(horizon_bars=16))

    assert json.loads(json.dumps(report))["base"]["n"] == report["base"]["n"]
