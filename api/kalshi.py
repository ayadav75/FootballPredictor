"""Kalshi EPL match-winner markets (public data, no auth).

Verified live against the API (2026-09-03):
- Base URL is api.elections.kalshi.com (trading-api.kalshi.com is retired
  and redirects there).
- Series KXEPLGAME holds match-winner events, ticker pattern
  KXEPLGAME-{YY}{MON}{DD}{HOME3}{AWAY3} (e.g. KXEPLGAME-26SEP04IPSLFC),
  each with three independent binary markets: one per team plus "-TIE".
- Prices come back as dollar strings ("0.1800"); a market's yes price IS
  the implied probability of that outcome. The three mid-prices summed to
  ~0.995 on live data, so no de-vig step is applied — we report raw mids
  plus their sum so callers can see any drift.

Responses are cached in-process for 60 seconds (module-level timestamped
cache): prices move, but not fast enough to justify hammering the API on
every request.
"""

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests

from data import canonical_team_name

KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXEPLGAME"
CACHE_TTL_SECONDS = 60

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

_cache: dict = {"at": 0.0, "payload": None}


class KalshiUnavailable(Exception):
    pass


@dataclass
class KalshiMarket:
    """One matched 3-way market for a fixture, prices as probabilities."""

    event_ticker: str
    home: float
    draw: float
    away: float
    mid_sum: float  # how close the three raw mids are to 1
    fetched_at: datetime


def _fetch_open_markets() -> tuple[list[dict], datetime]:
    now = time.monotonic()
    if _cache["payload"] is not None and now - _cache["at"] < CACHE_TTL_SECONDS:
        return _cache["payload"]

    markets, cursor = [], None
    try:
        while True:
            params = {"series_ticker": SERIES_TICKER, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(f"{KALSHI_API_URL}/markets", params=params, timeout=15)
            resp.raise_for_status()
            body = resp.json()
            markets.extend(body.get("markets", []))
            cursor = body.get("cursor")
            if not cursor:
                break
    except requests.RequestException as exc:
        raise KalshiUnavailable(f"Kalshi unreachable: {exc}") from exc

    payload = (markets, datetime.now(timezone.utc))
    _cache.update(at=now, payload=payload)
    return payload


def parse_event_date(event_ticker: str) -> date | None:
    """KXEPLGAME-26SEP04IPSLFC -> date(2026, 9, 4)."""
    try:
        seg = event_ticker.split("-")[1]
        return date(2000 + int(seg[:2]), _MONTHS[seg[2:5]], int(seg[5:7]))
    except (IndexError, KeyError, ValueError):
        return None


def _mid_price(market: dict) -> float | None:
    try:
        bid = float(market["yes_bid_dollars"])
        ask = float(market["yes_ask_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    if ask <= 0:  # empty book — fall back to last trade if there is one
        last = float(market.get("last_price_dollars") or 0)
        return last or None
    return (bid + ask) / 2


def build_market_index(markets: list[dict], known_teams: list[str]) -> dict:
    """Group markets by event and key them by (team-pair, event date).

    Keyed on a frozenset of canonical team names because Kalshi's home/away
    order isn't knowable from the markets payload alone — the caller
    reassigns outcomes by matching each price to the fixture's actual home
    and away team.
    """
    by_event: dict[str, dict] = {}
    for m in markets:
        by_event.setdefault(m["event_ticker"], {})[m.get("yes_sub_title", "")] = m

    index = {}
    for event_ticker, outcome_markets in by_event.items():
        event_date = parse_event_date(event_ticker)
        if event_date is None:
            continue
        prices, teams = {}, set()
        for sub_title, market in outcome_markets.items():
            mid = _mid_price(market)
            if mid is None:
                continue
            if sub_title.strip().lower() == "tie":
                prices["TIE"] = mid
            else:
                canon = canonical_team_name(sub_title, known_teams)
                if canon:
                    prices[canon] = mid
                    teams.add(canon)
        if len(teams) == 2 and "TIE" in prices:
            index[(frozenset(teams), event_date)] = {
                "event_ticker": event_ticker,
                "prices": prices,
            }
    return index


def get_market_index(known_teams: list[str]) -> tuple[dict, datetime]:
    markets, fetched_at = _fetch_open_markets()
    return build_market_index(markets, known_teams), fetched_at


def market_for_fixture(index: dict, home: str, away: str,
                       kickoff_utc: datetime, fetched_at: datetime) -> KalshiMarket | None:
    """Find the market for a fixture; None when Kalshi has no contract for
    it (an expected case, not an error). Matches on the team pair and the
    event date +/- 1 day (kickoff timezone vs ticker-date drift)."""
    pair = frozenset({home, away})
    for day_offset in (0, -1, 1):
        key = (pair, kickoff_utc.date() + timedelta(days=day_offset))
        entry = index.get(key)
        if entry:
            prices = entry["prices"]
            return KalshiMarket(
                event_ticker=entry["event_ticker"],
                home=prices[home],
                draw=prices["TIE"],
                away=prices[away],
                mid_sum=round(prices[home] + prices["TIE"] + prices[away], 4),
                fetched_at=fetched_at,
            )
    return None
