"""Live option implied volatility, so the credit is priced off a market.

The planner's arithmetic was always real; its *inputs* were not. Everything
down to the ``credit >= 20% of width`` gate was computed from
``analysis.SyntheticChain``, a constant 60% vol with a hand-drawn skew, so the
gate was a gate on an invented number.

Deribit publishes its whole ETH option chain in one unauthenticated request:

    GET /api/v2/public/get_book_summary_by_currency?currency=ETH&kind=option

Each row carries ``mark_iv``, ``bid_iv``, ``ask_iv``, ``underlying_price``,
``open_interest`` and an instrument name like ``ETH-27DEC26-2400-P``. One call,
no key, no quota of the kind the TradingView scraper has.

Two honesty rules are built in, because a wrong IV is worse than a missing one:

* an empty, unparseable or stale chain raises instead of returning a guess;
* interpolation is local and clamped, and says when it clamped.

Nothing here writes, caches or mutates state. ``fetch_chain`` is the only
function that touches the network, and it is never called at import.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

log = logging.getLogger("iv")

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
DEFAULT_CURRENCY = "ETH"

#: A chain older than this is not a price, it is a memory.
DEFAULT_MAX_AGE_MINUTES = 15.0

#: Deribit settles options at 08:00 UTC. Expiry parsing uses the date; the hour
#: only matters when picking the *nearest* expiry, where a few hours is noise.
DERIBIT_EXPIRY_HOUR_UTC = 8

#: Refuse absurd vols rather than interpolating through them. Deribit has quoted
#: four-figure mark_iv on deep wings that never trade.
MAX_PLAUSIBLE_IV = 5.0

_KINDS = {"C": "call", "P": "put"}


class ChainError(RuntimeError):
    """Base class for option-chain failures."""


class EmptyChainError(ChainError):
    """The chain holds nothing usable for the requested kind."""


class StaleChainError(ChainError):
    """The chain is older than the caller is willing to price off."""


@dataclass(frozen=True)
class Quote:
    """One instrument: a single strike, expiry and side.

    ``bid``/``ask``/``mark`` are in the **underlying** (ETH per contract), which
    is how Deribit quotes options -- not in USD. Multiply by the underlying to
    get dollars. They default to 0.0 because a quote that never traded has no
    market, and "no bid" must stay distinguishable from "bid of zero".
    """

    kind: str  # "put" | "call"
    strike: float
    expiry: datetime  # tz-aware UTC
    iv: float  # decimal, 0.62 == 62%
    open_interest: float = 0.0
    bid: float = 0.0  # underlying units
    ask: float = 0.0
    mark: float = 0.0

    @property
    def two_sided(self) -> bool:
        """True only when both sides of the market exist and are ordered."""
        return 0.0 < self.bid <= self.ask


@dataclass(frozen=True)
class ChainSnapshot:
    """One observation of a whole option chain."""

    currency: str
    underlying: float
    fetched_at: datetime
    quotes: tuple[Quote, ...]

    @property
    def expiries(self) -> list[datetime]:
        return sorted({quote.expiry for quote in self.quotes})

    def describe(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "underlying": round(self.underlying, 2),
            "fetched_at": self.fetched_at.isoformat(),
            "quotes": len(self.quotes),
            "expiries": len(self.expiries),
        }


@dataclass(frozen=True)
class Book:
    """One expiry, one side, every listed strike -- the ladder you can trade.

    Deliberately separate from :meth:`DeribitChain.iv`. That method interpolates
    in log-moneyness so the planner can ask about *any* strike; a book answers
    only about strikes that exist, and carries the bid/ask that turns a mark
    into a price you could actually get. The distinction matters because a
    credit computed at an interpolated 2211 is arithmetic about an instrument
    nobody can sell.
    """

    kind: str
    expiry: datetime
    spot: float
    legs: tuple[Quote, ...]
    observed_at: datetime
    source: str = "deribit"

    @property
    def strikes(self) -> list[float]:
        return [leg.strike for leg in self.legs]

    def at(self, strike: float, *, tol: float = 1e-6) -> Quote | None:
        for leg in self.legs:
            if abs(leg.strike - strike) <= tol:
                return leg
        return None

    def nearest(self, strike: float) -> Quote | None:
        if not self.legs:
            return None
        return min(self.legs, key=lambda leg: abs(leg.strike - strike))

    def days_to_expiry(self) -> float:
        return max((self.expiry - self.observed_at).total_seconds() / 86400.0, 0.0)

    def describe(self) -> dict[str, Any]:
        quoted = sum(1 for leg in self.legs if leg.two_sided)
        return {
            "kind": self.kind,
            "expiry": self.expiry.date().isoformat(),
            "dte_actual": round(self.days_to_expiry(), 2),
            "strikes": len(self.legs),
            "two_sided": quoted,
            "source": self.source,
        }


def parse_instrument(name: str) -> tuple[str, datetime, float]:
    """``ETH-27DEC26-2400-P`` -> (``"put"``, expiry, 2400.0).

    Deribit's naming is stable and unambiguous, so it is parsed rather than
    trusted to a second endpoint. A name that does not fit is a schema change,
    and we want it to raise rather than silently drop a strike.
    """
    parts = name.split("-")
    if len(parts) != 4:
        raise ChainError(f"unexpected instrument name {name!r}")
    _, expiry_token, strike_token, kind_token = parts
    kind = _KINDS.get(kind_token.upper())
    if kind is None:
        raise ChainError(f"unexpected option kind in {name!r}")
    try:
        expiry = datetime.strptime(expiry_token.upper(), "%d%b%y").replace(
            hour=DERIBIT_EXPIRY_HOUR_UTC, tzinfo=timezone.utc
        )
        strike = float(strike_token)
    except ValueError as exc:
        raise ChainError(f"cannot parse {name!r}: {exc}") from None
    if strike <= 0:
        raise ChainError(f"non-positive strike in {name!r}")
    return kind, expiry, strike


def _quote_iv(row: Mapping[str, Any]) -> float | None:
    """The row's implied vol as a **decimal** (0.616, not 61.6).

    Deribit publishes ``mark_iv``/``bid_iv``/``ask_iv`` in percent. Converting
    at the parse boundary is the only place the distinction is allowed to
    exist: everything downstream -- Black-Scholes, the credit ratio, the
    payload -- speaks decimals, and 61.6 as a sigma is not a price.

    ``mark_iv`` is Deribit's own surface mark. If it is missing, a bid/ask mid
    is a fair substitute. A one-sided quote is not: half a market is not a
    price.
    """
    mark = row.get("mark_iv")
    if isinstance(mark, (int, float)) and mark > 0:
        return float(mark) / 100.0
    bid, ask = row.get("bid_iv"), row.get("ask_iv")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
        return (float(bid) + float(ask)) / 200.0
    return None


def _quote_price(value: Any) -> float:
    """A book summary price, already in the underlying. Missing means 0.0.

    A mark exists on almost every row; a bid does not. Both are kept in the
    same units and the same field shape so that "the market is one-sided" is
    expressible, rather than being rounded away into a mid.
    """
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return 0.0


def parse_quotes(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Quote], list[str]]:
    """Rows -> usable quotes, plus one complaint per dropped instrument."""
    quotes: list[Quote] = []
    problems: list[str] = []
    for row in rows:
        name = str(row.get("instrument_name") or "")
        try:
            kind, expiry, strike = parse_instrument(name)
        except ChainError as exc:
            problems.append(str(exc))
            continue
        iv = _quote_iv(row)
        if iv is None:
            problems.append(f"{name}: no mark_iv and no two-sided quote")
            continue
        if iv > MAX_PLAUSIBLE_IV:
            problems.append(f"{name}: mark_iv {iv:.2f} is not a vol")
            continue
        oi = row.get("open_interest")
        quotes.append(
            Quote(
                kind=kind,
                strike=strike,
                expiry=expiry,
                iv=float(iv),
                open_interest=float(oi) if isinstance(oi, (int, float)) else 0.0,
                bid=_quote_price(row.get("bid_price")),
                ask=_quote_price(row.get("ask_price")),
                mark=_quote_price(row.get("mark_price")),
            )
        )
    return quotes, problems


def build_snapshot(
    payload: Mapping[str, Any], *, currency: str = DEFAULT_CURRENCY, now: datetime | None = None
) -> ChainSnapshot:
    """Turn one ``get_book_summary_by_currency`` response into a snapshot.

    Kept separate from the HTTP call so the parser is testable against a
    recorded fixture, with no network and no clock.
    """
    rows = payload.get("result")
    if not isinstance(rows, list) or not rows:
        raise EmptyChainError("chain response has no 'result' rows")

    quotes, problems = parse_quotes(rows)
    for problem in problems[:5]:
        log.warning("%s: %s", currency, problem)
    if problems:
        log.warning("%s: dropped %d unusable instrument(s)", currency, len(problems))
    if not quotes:
        raise EmptyChainError(f"no usable {currency} option quotes")

    underlyings = [
        float(row["underlying_price"])
        for row in rows
        if isinstance(row.get("underlying_price"), (int, float))
        and float(row["underlying_price"]) > 0
    ]
    if not underlyings:
        raise EmptyChainError(f"no underlying price in the {currency} chain")

    return ChainSnapshot(
        currency=currency,
        underlying=statistics.median(underlyings),
        fetched_at=now or datetime.now(timezone.utc),
        quotes=tuple(quotes),
    )


def _get(url: str, params: Mapping[str, Any], *, timeout: float, retries: int, backoff: float) -> Any:
    """One GET with a hard timeout and bounded retries. Makes no promises."""
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "trading-agent/1.0", "Accept": "application/json"}
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            if isinstance(body, Mapping) and body.get("error"):
                raise ChainError(str(body["error"]))
            return body
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ChainError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise ChainError(f"{params}: {last}")


def fetch_chain(
    currency: str = DEFAULT_CURRENCY,
    *,
    timeout: float = 8.0,
    retries: int = 2,
    backoff: float = 1.5,
    now: datetime | None = None,
) -> ChainSnapshot:
    """Fetch the live chain. The only function here that touches the network."""
    payload = _get(
        f"{DERIBIT_BASE}/get_book_summary_by_currency",
        {"currency": currency.upper(), "kind": "option"},
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )
    return build_snapshot(payload, currency=currency.upper(), now=now)


@dataclass(frozen=True)
class DeribitChain:
    """A live chain, viewed through the planner's one-method interface.

    ``iv`` picks the expiry nearest the requested DTE, then interpolates
    linearly in log-moneyness across that expiry's strikes. Outside the quoted
    range it clamps to the nearest node and warns: extrapolating a skew is how
    a 0.30-delta strike ends up priced off a wing that never trades.
    """

    snapshot: ChainSnapshot
    max_age_minutes: float = DEFAULT_MAX_AGE_MINUTES
    now: datetime | None = None  # injectable for tests

    @property
    def contract_size(self) -> float:
        """Underlying per contract. Deribit ETH options are 1 ETH.

        Not 100, which is the US-equity convention the planner used to assume:
        sizing off 100 shrank every position by two orders of magnitude.
        """
        return 1.0

    # -- interface -------------------------------------------------------- #

    def iv(self, kind: str, dte_days: int, strike: float, spot: float) -> float:
        if kind not in ("put", "call"):
            raise ChainError(f"kind must be 'put' or 'call', got {kind!r}")
        if spot <= 0 or strike <= 0:
            raise ChainError("spot and strike must be positive")
        self._check_fresh()

        nodes = self._nodes(kind, dte_days, spot)
        x = math.log(strike / spot)
        if len(nodes) == 1:
            return nodes[0][1]

        xs = [node[0] for node in nodes]
        if x <= xs[0]:
            self._warn_clamp(kind, strike, spot, "below")
            return nodes[0][1]
        if x >= xs[-1]:
            self._warn_clamp(kind, strike, spot, "above")
            return nodes[-1][1]

        right = bisect.bisect_left(xs, x)
        x0, y0 = nodes[right - 1]
        x1, y1 = nodes[right]
        # Linear in log-moneyness: vol is closer to linear there than in strike,
        # which is what keeps a 2% wing from bending the whole smile.
        weight = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return y0 + weight * (y1 - y0)

    # -- internals -------------------------------------------------------- #

    def book(self, kind: str, dte_days: int, spot: float | None = None) -> Book:
        """The whole listed ladder for the expiry nearest ``dte_days``.

        No interpolation: every leg is a strike that exists, with whatever
        bid/ask the exchange published for it.

        ``spot`` is accepted for interface parity and deliberately ignored:
        the prices in this book were quoted off the exchange's own underlying,
        so that is the frame they mean. Using a caller's spot -- a different
        venue, a different minute -- to convert them, or to pick a
        delta-strike, is how the plan drifts from the instrument.
        """
        if kind not in ("put", "call"):
            raise ChainError(f"kind must be 'put' or 'call', got {kind!r}")
        self._check_fresh()
        expiry = self._expiry(kind, dte_days)
        legs = tuple(
            sorted(
                (
                    quote
                    for quote in self.snapshot.quotes
                    if quote.kind == kind and quote.expiry == expiry
                ),
                key=lambda quote: quote.strike,
            )
        )
        if not legs:
            raise EmptyChainError(f"no {kind} quotes for expiry {expiry:%Y-%m-%d}")
        return Book(
            kind=kind,
            expiry=expiry,
            spot=self.snapshot.underlying,
            legs=legs,
            observed_at=self.now or datetime.now(timezone.utc),
        )

    def _check_fresh(self) -> None:
        if self.max_age_minutes <= 0:
            return
        moment = self.now or datetime.now(timezone.utc)
        age = (moment - self.snapshot.fetched_at).total_seconds() / 60.0
        if age > self.max_age_minutes:
            raise StaleChainError(
                f"chain is {age:.0f} min old (limit {self.max_age_minutes:.0f}); "
                "refusing to price off it"
            )

    def _expiry(self, kind: str, dte_days: int) -> datetime:
        moments = sorted({quote.expiry for quote in self.snapshot.quotes if quote.kind == kind})
        if not moments:
            raise EmptyChainError(f"no {kind} quotes in the {self.snapshot.currency} chain")
        reference = (self.now or datetime.now(timezone.utc)) + timedelta(days=dte_days)
        return min(moments, key=lambda moment: abs((moment - reference).total_seconds()))

    def _nodes(self, kind: str, dte_days: int, spot: float) -> list[tuple[float, float]]:
        expiry = self._expiry(kind, dte_days)
        nodes = sorted(
            (math.log(quote.strike / spot), quote.iv)
            for quote in self.snapshot.quotes
            if quote.kind == kind and quote.expiry == expiry
        )
        if not nodes:
            raise EmptyChainError(f"no {kind} quotes for expiry {expiry:%Y-%m-%d}")
        return nodes

    def _warn_clamp(self, kind: str, strike: float, spot: float, side: str) -> None:
        log.warning(
            "%s %.0f is %s the quoted %s range for spot %.0f; clamping to the nearest wing",
            kind,
            strike,
            side,
            kind,
            spot,
        )

    def describe(self) -> dict[str, Any]:
        info = self.snapshot.describe()
        info["oldest_quote_age_minutes"] = round(
            ((self.now or datetime.now(timezone.utc)) - self.snapshot.fetched_at).total_seconds()
            / 60.0,
            1,
        )
        return info


def describe(chain: Any, *, dte_days: int, spot: float, note: str | None = None) -> dict[str, Any]:
    """What the planner is actually pricing off, for the payload.

    A synthetic credit and a market credit must never look the same on screen,
    so every payload carries this beside the plan.
    """
    snapshot = getattr(chain, "snapshot", None)
    if isinstance(snapshot, ChainSnapshot):
        info: dict[str, Any] = {"source": "deribit", **chain.describe()}
    else:
        info = {"source": "synthetic"}
        note = note or "synthetic Black-Scholes stub; credit is arithmetic, not a market"

    try:
        info["iv_atm"] = round(float(chain.iv("put", dte_days, spot, spot)), 4)
    except Exception as exc:  # a broken chain must not sink the payload
        info["iv_atm"] = None
        info["iv_atm_error"] = str(exc)

    if note:
        info["note"] = note
    return info
