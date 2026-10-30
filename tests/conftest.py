import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import iv  # noqa: E402
import bars  # noqa: E402


@pytest.fixture(autouse=True)
def no_live_option_chain(monkeypatch):
    """No test may reach Deribit.

    The CLI prices off the live chain by default, which is right in production
    and wrong in a test suite: it makes runs slow, flaky and dependent on
    someone else's uptime. Patching the *fetch* rather than ``default_chain``
    keeps the fallback path -- the one that decides what a degraded run looks
    like -- under test in every run. Tests that want a live chain say so.
    """

    def offline(*_args, **_kwargs):
        raise iv.ChainError("test suite is offline")

    monkeypatch.setattr(iv, "fetch_chain", offline)


@pytest.fixture(autouse=True)
def no_live_bars(monkeypatch):
    """No test may reach Binance, and none may write into the real bar store.

    Failing the fetch (rather than stubbing a series) exercises the fail-soft
    path every run: the payload must still carry a decision and must say the
    range is unmeasured. Tests that want measured bars patch this back.
    """

    def offline(*_args, **_kwargs):
        raise bars.BarError("test suite is offline")

    # Patched at the HTTP seam, not at ``fetch_klines``: the parsing and the
    # settled-candle rule stay under test, and a test that wants a recorded
    # page can patch the same seam later without fighting this fixture.
    monkeypatch.setattr(bars, "_get", offline)
