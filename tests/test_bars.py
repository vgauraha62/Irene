"""Offline tests for the bar store and the derived range measures.

Everything runs against a recorded Binance response. The properties that matter
here are mechanical: append-only, dedupe, resample alignment, and the arithmetic
of ATR and realised vol.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import bars

FIXTURE = Path(__file__).parent / "fixtures" / "binance_ethusdt_15m.json"


@pytest.fixture(scope="module")
def raw() -> list:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def recorded(raw: list) -> list[bars.Bar]:
    return bars.parse_klines(raw)


def make(start: datetime, closes: list[float], *, spread: float = 1.0) -> list[bars.Bar]:
    """A deterministic series: open == previous close, high/low symmetric."""
    out: list[bars.Bar] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        moment = start + timedelta(minutes=15 * index)
        out.append(
            bars.Bar(
                open_time=moment,
                open=previous,
                high=max(previous, close) + spread,
                low=min(previous, close) - spread,
                close=close,
                volume=1.0,
            )
        )
        previous = close
    return out


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_klines_are_parsed_into_bars(recorded: list[bars.Bar]):
    first = recorded[0]

    assert len(recorded) == 160
    assert first.open_time.tzinfo is not None
    assert first.high >= first.low > 0
    assert all(later.open_time > earlier.open_time for earlier, later in zip(recorded, recorded[1:]))


def test_a_malformed_row_fails_instead_of_leaving_a_hole(raw: list):
    broken = [list(row) for row in raw[:5]]
    broken[2] = broken[2][:4]

    with pytest.raises(bars.BarError, match="not a kline"):
        bars.parse_klines(broken)


def test_a_non_positive_price_fails(raw: list):
    broken = [list(row) for row in raw[:3]]
    broken[1][4] = "0"

    with pytest.raises(bars.BarError, match="non-positive"):
        bars.parse_klines(broken)


def test_a_duplicate_open_time_fails(raw: list):
    broken = [list(row) for row in raw[:3]]
    broken.append(list(broken[-1]))

    with pytest.raises(bars.BarError, match="duplicate"):
        bars.parse_klines(broken)


def test_the_forming_candle_is_dropped(monkeypatch):
    """The last kline is incomplete; ATR must not depend on when you polled."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=45)
    rows = []
    for index in range(3):
        moment = start + timedelta(minutes=15 * index)
        rows.append(
            [int(moment.timestamp() * 1000), "1", "2", "0.5", "1.5", "10"]
        )
    # The third row opens *now*, so it cannot be a closed 15m bar.
    rows[-1][0] = int(now.timestamp() * 1000)

    monkeypatch.setattr(bars, "_get", lambda *a, **k: rows)
    kept = bars.fetch_klines(limit=3)

    assert len(kept) == 2
    assert kept[-1].open_time == datetime.fromtimestamp(rows[1][0] / 1000.0, tz=timezone.utc)


def test_settled_false_keeps_the_forming_candle(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [[int(now.timestamp() * 1000), "1", "2", "0.5", "1.5", "10"]]

    monkeypatch.setattr(bars, "_get", lambda *a, **k: rows)

    assert len(bars.fetch_klines(limit=1)) == 0
    assert len(bars.fetch_klines(limit=1, settled=False)) == 1


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def test_append_is_idempotent(tmp_path, recorded: list[bars.Bar]):
    path = tmp_path / "ETHUSDT_15m.csv"

    assert bars.append_bars(path, recorded[:10]) == 10
    assert bars.append_bars(path, recorded[:10]) == 0  # same window twice: no new rows
    assert bars.append_bars(path, recorded[:15]) == 5  # overlap: only the 5 new ones
    assert len(bars.load_bars(path)) == 15


def test_append_keeps_the_store_readable(tmp_path, recorded: list[bars.Bar]):
    path = tmp_path / "ETHUSDT_15m.csv"
    bars.append_bars(path, recorded[:20])
    stored = bars.load_bars(path)

    assert stored[0].open_time == recorded[0].open_time
    assert stored[-1].close == pytest.approx(recorded[19].close)
    assert stored[0].volume == pytest.approx(recorded[0].volume)


def test_a_missing_store_is_empty_not_an_error(tmp_path):
    assert bars.load_bars(tmp_path / "nothing.csv") == []


def test_backfilling_older_bars_extends_history_without_duplicating_it(
    tmp_path, recorded: list[bars.Bar]
):
    """A backfill appends bars *older* than everything on disk, unsorted."""
    path = tmp_path / "ETHUSDT_15m.csv"

    assert bars.append_bars(path, recorded[-20:]) == 20
    assert bars.append_bars(path, recorded[-40:-20]) == 20  # older, appended after
    assert bars.append_bars(path, recorded[-40:]) == 0  # the whole window is known

    stored = bars.load_bars(path)
    assert len(stored) == 40
    assert stored == sorted(stored, key=lambda bar: bar.open_time)
    assert len({bar.open_time for bar in stored}) == 40


def test_a_duplicate_line_on_disk_is_read_once(tmp_path, recorded: list[bars.Bar]):
    path = tmp_path / "ETHUSDT_15m.csv"
    bars.append_bars(path, recorded[:5])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(path.read_text(encoding="utf-8").splitlines()[1] + "\n")

    assert len(bars.load_bars(path)) == 5


def test_backfill_walks_backwards_and_stops_at_the_start_of_history(tmp_path, monkeypatch):
    """45 bars, 20 per page: two full pages, then a short one ends the walk."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        [
            int((start + timedelta(minutes=15 * index)).timestamp() * 1000),
            "100",
            "101",
            "99",
            str(100 + index),
            "1",
        ]
        for index in range(45)
    ]

    def fake_get(url, params, **kwargs):
        end = params.get("endTime")
        limit = int(params["limit"])
        if end is None:
            return rows[-limit:]
        older = [row for row in rows if row[0] < end]
        return older[-limit:]

    monkeypatch.setattr(bars, "_get", fake_get)
    result = bars.backfill(
        tmp_path / "store.csv", pages=5, limit=20, pause=0
    )

    assert result["stored"] == 45
    assert result["appended"] == 45
    assert result["pages"] == 3  # two full pages, then a short one ends it


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def test_resampling_aggregates_the_right_fields(recorded: list[bars.Bar]):
    two = bars.resample(recorded, 2, "15m")

    # Groups are clock-aligned to :00/:30, so the leading :15 bar forms a
    # one-bar bucket and is correctly dropped rather than paired across the
    # boundary. Every emitted group must aggregate its own two source bars.
    assert two
    assert all(group.open_time.minute % 30 == 0 for group in two)
    for group in two:
        members = [
            bar
            for bar in recorded
            if group.open_time <= bar.open_time < group.open_time + timedelta(minutes=30)
        ]
        assert len(members) == 2
        assert group.open == pytest.approx(members[0].open)
        assert group.high == pytest.approx(max(bar.high for bar in members))
        assert group.low == pytest.approx(min(bar.low for bar in members))
        assert group.close == pytest.approx(members[-1].close)
        assert group.volume == pytest.approx(sum(bar.volume for bar in members))


def test_an_incomplete_tail_group_is_dropped(recorded: list[bars.Bar]):
    """A half-built 4h bar is not a 4h bar."""
    odd = recorded[:7]  # 7 bars: three complete pairs, one left over
    assert len(bars.resample(odd, 2, "15m")) == 3


def test_resampling_keeps_clock_alignment_across_a_gap(recorded: list[bars.Bar]):
    """Grouping by clock, not by index: a missing bar must not shift everything."""
    with_gap = list(recorded[:8]) + list(recorded[9:16])
    groups = bars.resample(with_gap, 2, "15m")

    for group in groups:
        offset = group.open_time.minute
        assert offset % 30 == 0


# --------------------------------------------------------------------------- #
# Range measures
# --------------------------------------------------------------------------- #


def test_atr_matches_a_hand_worked_value():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    series = make(start, [100.0] * 20, spread=1.0)
    # Every bar is 100 -> 100 with a +/-1 range: true range is 2.0 throughout.
    assert bars.atr(series, period=14) == pytest.approx(2.0)


def test_atr_pct_is_scale_free():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cheap = make(start, [100.0] * 20, spread=1.0)
    rich = make(start, [1000.0] * 20, spread=10.0)

    assert bars.atr_pct(cheap, 14) == pytest.approx(2.0)
    assert bars.atr_pct(rich, 14) == pytest.approx(2.0)


def test_atr_is_none_until_it_has_warmed_up():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    series = make(start, [100.0] * 10, spread=1.0)

    assert bars.atr(series, period=14) is None
    assert bars.atr_pct(series, 14) is None


def test_a_real_series_gives_a_plausible_atr(recorded: list[bars.Bar]):
    value = bars.atr_pct(recorded, 14)

    assert value is not None
    assert 0.02 < value < 3.0  # 15m ETH ATR as % of price


def test_keltner_width_is_atr_scaled(recorded: list[bars.Bar]):
    width = bars.keltner_width_pct(recorded, ema_period=20, atr_period=10, mult=2.0)
    expected = 2.0 * 2.0 * bars.atr(recorded, 10) / recorded[-1].close * 100.0

    assert width == pytest.approx(expected)


def test_compression_is_ranked_against_its_own_history(recorded: list[bars.Bar]):
    series = bars.rolling_keltner_width_pct(recorded, window=80)
    current = bars.keltner_width_pct(recorded, ema_period=20, atr_period=10)

    assert series
    rank = bars.percentile_rank(series, current)
    assert rank is not None and 0.0 <= rank <= 100.0


def test_a_tight_series_ranks_low_and_a_wide_one_high():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    calm = [100.0 + 0.02 * index for index in range(60)]
    wide = [100.0 + (5.0 if index % 2 else -5.0) for index in range(60)]

    calm_width = bars.keltner_width_pct(make(start, calm, spread=0.05))
    wide_width = bars.keltner_width_pct(make(start, wide, spread=0.05))

    assert calm_width < wide_width


def test_expected_move_scales_with_root_time(recorded: list[bars.Bar]):
    """A 4h move is 4x a 15m move, not 16x: variance adds, not sigma."""
    short = bars.expected_move_pct(recorded, 0.25)
    long = bars.expected_move_pct(recorded, 4.0)

    assert short is not None and long is not None
    assert long / short == pytest.approx(math.sqrt(16.0), rel=0.02)


def test_expected_move_rejects_a_negative_horizon(recorded: list[bars.Bar]):
    with pytest.raises(bars.BarError):
        bars.expected_move_pct(recorded, -1.0)


def test_the_atr_series_ends_at_the_scalar(recorded: list[bars.Bar]):
    series = bars.atr_series(recorded, 14)

    assert series[-1] == pytest.approx(bars.atr(recorded, 14))
    assert all(value is None for value in series[:14])


def test_adx_is_high_in_a_clean_trend_and_low_when_nothing_directional_happens():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trending = make(start, [100.0 + index for index in range(120)], spread=0.5)

    # Contracting inside bars: every high is lower than the last and every low
    # is higher, so there is no directional movement in either direction.
    narrowing = [
        bars.Bar(
            open_time=start + timedelta(minutes=15 * index),
            open=100.0,
            high=100.0 + 2.0 / (1 + index * 0.05),
            low=100.0 - 2.0 / (1 + index * 0.05),
            close=100.0,
            volume=1.0,
        )
        for index in range(120)
    ]

    strong = bars.adx(trending, 14)
    directionless = bars.adx(narrowing, 14)

    assert strong is not None and directionless is not None
    assert strong > 60
    assert directionless < 5


def test_adx_is_none_until_it_has_warmed_up(recorded: list[bars.Bar]):
    assert bars.adx_series(recorded[:20], 14)[-1] is None


def test_the_adx_series_cannot_see_the_future(recorded: list[bars.Bar]):
    """Index i must not change when the bars after i do."""
    honest = bars.adx_series(recorded, 14)
    tampered = bars.adx_series(recorded[:60] + recorded[60:][::-1], 14)

    assert honest[59] == pytest.approx(tampered[59])
    assert honest[59] is not None


# --------------------------------------------------------------------------- #
# Cross-check against the feed
# --------------------------------------------------------------------------- #


def test_an_agreeing_series_passes_the_cross_check(recorded: list[bars.Bar]):
    ours = {
        "EMA20": bars.ema_last([bar.close for bar in recorded], 20),
        "EMA50": bars.ema_last([bar.close for bar in recorded], 50),
    }

    result = bars.cross_check(recorded, ours, periods={"EMA20": 20, "EMA50": 50})

    assert result["status"] == "ok"
    assert result["compared"] == 2
    assert result["max_rel_error"] == pytest.approx(0.0)


def test_a_different_series_fails_loudly(recorded: list[bars.Bar]):
    """Five percent is not a seed difference; it is the wrong instrument."""
    wrong = {"EMA20": 1.0}

    with pytest.raises(bars.BarError, match="not the same series"):
        bars.cross_check(recorded, wrong, periods={"EMA20": 20})


def test_a_period_without_enough_bars_is_reported_not_invented(recorded: list[bars.Bar]):
    result = bars.cross_check(
        recorded[:30], {"EMA200": 2400.0}, periods={"EMA200": 200}
    )

    assert result["status"] == "unchecked"
    assert result["emas"]["EMA200"]["status"] == "not enough bars"
