"""Tests for team-name canonicalization, Kalshi market matching, and the
/fixtures/upcoming endpoint (external APIs mocked)."""

from datetime import date, datetime, timezone

import pytest

from api import kalshi
from api.fixtures import Fixture
from data import canonical_team_name
from test_api import client  # noqa: F401  (shared TestClient fixture)

CANONICAL = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leeds", "Leicester", "Liverpool", "Luton", "Man City", "Man United",
    "Newcastle", "Nott'm Forest", "Sheffield United", "Southampton",
    "Sunderland", "Tottenham", "West Ham", "Wolves",
]


@pytest.mark.parametrize("source_name,expected", [
    # football-data.org spellings (live samples)
    ("Arsenal FC", "Arsenal"),
    ("AFC Bournemouth", "Bournemouth"),
    ("Sunderland AFC", "Sunderland"),
    ("Manchester United FC", "Man United"),
    ("Manchester City FC", "Man City"),
    ("Brighton & Hove Albion FC", "Brighton"),
    ("Nottingham Forest FC", "Nott'm Forest"),
    ("Wolverhampton Wanderers FC", "Wolves"),
    ("Tottenham Hotspur FC", "Tottenham"),
    ("Ipswich Town FC", "Ipswich"),
    # Kalshi spellings (live samples)
    ("Ipswich Town", "Ipswich"),
    ("Liverpool", "Liverpool"),
    ("Newcastle", "Newcastle"),
    ("Leeds United", "Leeds"),
    ("Manchester City", "Man City"),
    ("Crystal Palace", "Crystal Palace"),
    # Never in our data -> must be None, not a guess
    ("Coventry City FC", None),
    ("Coventry", None),
    ("Real Madrid", None),
])
def test_canonical_team_name(source_name, expected):
    assert canonical_team_name(source_name, CANONICAL) == expected


def test_parse_event_date():
    assert kalshi.parse_event_date("KXEPLGAME-26SEP04IPSLFC") == date(2026, 9, 4)
    assert kalshi.parse_event_date("KXEPLGAME-27JAN31ARSCHE") == date(2027, 1, 31)
    assert kalshi.parse_event_date("GARBAGE") is None


def _market(event, sub_title, bid, ask):
    return {"event_ticker": event, "yes_sub_title": sub_title,
            "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
            "last_price_dollars": f"{(bid + ask) / 2:.4f}"}


def test_market_index_and_fixture_matching():
    markets = [
        _market("KXEPLGAME-26SEP04IPSLFC", "Ipswich Town", 0.15, 0.16),
        _market("KXEPLGAME-26SEP04IPSLFC", "Liverpool", 0.65, 0.66),
        _market("KXEPLGAME-26SEP04IPSLFC", "Tie", 0.18, 0.19),
        # Unmappable team -> event skipped, not mis-assigned
        _market("KXEPLGAME-26SEP05MCICOV", "Manchester City", 0.80, 0.82),
        _market("KXEPLGAME-26SEP05MCICOV", "Coventry", 0.05, 0.07),
        _market("KXEPLGAME-26SEP05MCICOV", "Tie", 0.12, 0.14),
    ]
    fetched = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    index = kalshi.build_market_index(markets, CANONICAL)
    assert len(index) == 1  # Coventry event correctly excluded

    kickoff = datetime(2026, 9, 4, 19, 0, tzinfo=timezone.utc)
    m = kalshi.market_for_fixture(index, "Ipswich", "Liverpool", kickoff, fetched)
    assert m is not None
    assert m.home == pytest.approx(0.155)   # Ipswich is home in the fixture
    assert m.away == pytest.approx(0.655)
    assert m.draw == pytest.approx(0.185)
    assert m.mid_sum == pytest.approx(0.995)
    assert m.event_ticker == "KXEPLGAME-26SEP04IPSLFC"

    # Reversed fixture orientation still matches, with prices reassigned
    m2 = kalshi.market_for_fixture(index, "Liverpool", "Ipswich", kickoff, fetched)
    assert m2.home == pytest.approx(0.655)

    # No market listed -> None (expected case)
    assert kalshi.market_for_fixture(index, "Arsenal", "Chelsea", kickoff, fetched) is None


def test_upcoming_fixtures_endpoint(client, monkeypatch):  # noqa: F811
    """Endpoint against the synthetic-model TestClient (teams A-F, F is
    fallback), with both external APIs mocked."""
    import api.main as api_main

    kickoff = datetime(2026, 9, 4, 19, 0, tzinfo=timezone.utc)
    fixtures = [
        Fixture(kickoff_utc=kickoff, home_name="A FC", away_name="B FC", matchday=3),
        Fixture(kickoff_utc=kickoff, home_name="A FC", away_name="F FC", matchday=3),
        Fixture(kickoff_utc=kickoff, home_name="Narnia FC", away_name="B FC", matchday=3),
    ]
    markets = [
        _market("KXEPLGAME-26SEP04AB", "A", 0.50, 0.52),
        _market("KXEPLGAME-26SEP04AB", "B", 0.24, 0.26),
        _market("KXEPLGAME-26SEP04AB", "Tie", 0.23, 0.25),
    ]
    fetched = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(api_main, "fetch_upcoming_fixtures", lambda days=7: fixtures)
    monkeypatch.setattr(
        api_main, "get_market_index",
        lambda teams: (kalshi.build_market_index(markets, teams), fetched),
    )

    resp = client.get("/fixtures/upcoming")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_run_id"] == 1
    assert len(body["fixtures"]) == 3

    with_market, no_market, unknown = body["fixtures"]

    assert with_market["home_team"] == "A"
    assert with_market["model"]["fallback_rating"] is False
    assert with_market["kalshi"]["home"] == pytest.approx(0.51)
    assert with_market["kalshi"]["mid_sum"] == pytest.approx(1.0)
    assert with_market["kalshi"]["fetched_at"].startswith("2026-09-03T12:00")
    assert with_market["edge"]["home"] == pytest.approx(
        with_market["model"]["home_win"] - 0.51)

    assert no_market["model"]["fallback_rating"] is True  # F is fallback-rated
    assert no_market["kalshi"] is None
    assert "No Kalshi market" in no_market["kalshi_note"]
    assert no_market["edge"] is None

    assert unknown["model"] is None
    assert "Narnia FC" in unknown["model_note"]
    assert unknown["edge"] is None


def test_upcoming_fixtures_source_down(client, monkeypatch):  # noqa: F811
    import api.main as api_main
    from api.fixtures import FixturesUnavailable

    def boom(days=7):
        raise FixturesUnavailable("FOOTBALL_DATA_API_KEY is not set")

    monkeypatch.setattr(api_main, "fetch_upcoming_fixtures", boom)
    resp = client.get("/fixtures/upcoming")
    assert resp.status_code == 503
    assert "FOOTBALL_DATA_API_KEY" in resp.json()["detail"]
