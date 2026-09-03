"""Upcoming Premier League fixtures from football-data.org (v4 API).

Different site from our historical source (football-data.co.uk) — this one
has scheduled matches. Free tier; needs an API key in the
FOOTBALL_DATA_API_KEY environment variable (X-Auth-Token header).

Responses are cached for 5 minutes: the free tier allows 10 requests/min
and fixture schedules don't move minute to minute.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4"
# SCHEDULED = date known, time TBC; TIMED = kickoff confirmed.
UPCOMING_STATUSES = {"SCHEDULED", "TIMED"}
CACHE_TTL_SECONDS = 300

_cache: dict = {"at": 0.0, "days": None, "fixtures": None}


class FixturesUnavailable(Exception):
    """Raised when fixtures can't be fetched (missing key, upstream error)."""


@dataclass
class Fixture:
    kickoff_utc: datetime
    home_name: str  # source naming, e.g. "Arsenal FC"
    away_name: str
    matchday: int | None


def fetch_upcoming_fixtures(days: int = 7) -> list[Fixture]:
    now = time.monotonic()
    if _cache["fixtures"] is not None and _cache["days"] == days \
            and now - _cache["at"] < CACHE_TTL_SECONDS:
        return _cache["fixtures"]

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise FixturesUnavailable(
            "FOOTBALL_DATA_API_KEY is not set — get a free key at "
            "football-data.org/client/register"
        )
    today = datetime.now(timezone.utc).date()
    try:
        resp = requests.get(
            f"{FOOTBALL_DATA_API_URL}/competitions/PL/matches",
            params={"dateFrom": today.isoformat(),
                    "dateTo": (today + timedelta(days=days)).isoformat()},
            headers={"X-Auth-Token": api_key},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise FixturesUnavailable(f"football-data.org unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise FixturesUnavailable(
            f"football-data.org returned {resp.status_code}: {resp.text[:200]}"
        )

    fixtures = [
        Fixture(
            kickoff_utc=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
            home_name=m["homeTeam"]["name"],
            away_name=m["awayTeam"]["name"],
            matchday=m.get("matchday"),
        )
        for m in resp.json().get("matches", [])
        if m.get("status") in UPCOMING_STATUSES
    ]
    fixtures.sort(key=lambda f: f.kickoff_utc)
    _cache.update(at=now, days=days, fixtures=fixtures)
    return fixtures
