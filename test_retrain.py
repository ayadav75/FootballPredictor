"""Tests for the retrain job's helpers (no network, no DB)."""

from datetime import date

import pandas as pd
import pytest

from retrain import (
    apply_low_data_fallbacks,
    ingest_team_name,
    results_dataframe,
    season_label,
)
from test_model import make_model

KNOWN = ["Arsenal", "Man United", "Coventry City"]


def test_season_label():
    assert season_label(date(2026, 8, 21)) == "26-27"
    assert season_label(date(2027, 5, 15)) == "26-27"
    assert season_label(date(2026, 7, 31)) == "25-26"


def test_ingest_team_name_maps_known_and_mints_new():
    assert ingest_team_name("Arsenal FC", KNOWN) == "Arsenal"
    assert ingest_team_name("Manchester United FC", KNOWN) == "Man United"
    # Known from a previous retrain -> maps, no duplicate entity
    assert ingest_team_name("Coventry City FC", KNOWN) == "Coventry City"
    # Genuinely new club -> minted stable name, not None
    assert ingest_team_name("Hull City AFC", KNOWN) == "Hull City"


def test_results_dataframe_shape():
    matches_json = [{
        "utcDate": "2026-08-21T19:00:00Z",
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Hull City AFC"},
        "score": {"fullTime": {"home": 3, "away": 0}},
    }, {
        # In-play/edge case: null score must be skipped, not crash
        "utcDate": "2026-08-22T14:00:00Z",
        "homeTeam": {"name": "Coventry City FC"},
        "awayTeam": {"name": "Manchester United FC"},
        "score": {"fullTime": {"home": None, "away": None}},
    }]
    df = results_dataframe(matches_json, KNOWN)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Hull City"
    assert row["season"] == "26-27"
    assert row["result"] == "H"
    assert pd.isna(row["ps_close_home_odds"])


def _train_df():
    """Synthetic history: A-D are established, E was 'relegated' (in season
    25-26 only), N is newly promoted with 2 matches in 26-27."""
    rows = []
    for i, (h, a) in enumerate([("A", "B"), ("C", "D"), ("A", "E"), ("E", "C")] * 5):
        rows.append(("25-26", f"2026-0{1 + i % 5}-10", h, a))
    rows += [("26-27", "2026-08-21", "A", "N"), ("26-27", "2026-08-28", "N", "B"),
             ("26-27", "2026-08-28", "C", "A"), ("26-27", "2026-08-29", "B", "D")]
    df = pd.DataFrame(rows, columns=["season", "date", "home_team", "away_team"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_goals"], df["away_goals"] = 1, 1
    return df


def test_low_data_fallback_replaces_degenerate_ratings():
    model = make_model(
        attack={t: 1.0 for t in "ABCDE"} | {"N": 3.0},     # N's fit is garbage
        defense={t: 1.3 for t in "ABCDE"} | {"N": 0.01},
        home_advantage=1.2,
    )
    model.teams = list("ABCDEN")
    flagged = apply_low_data_fallbacks(model, _train_df(), min_matches=5)
    assert flagged == {"N"}
    # N now carries the relegated side's (E's) mean ratings
    assert model.attack["N"] == pytest.approx(model.attack["E"])
    assert model.defense["N"] == pytest.approx(model.defense["E"])
    # Established teams untouched
    assert model.attack["A"] == 1.0


def test_low_data_fallback_noop_when_all_teams_have_history():
    model = make_model(
        attack={t: 1.0 for t in "ABCDE"},
        defense={t: 1.3 for t in "ABCDE"},
    )
    model.teams = list("ABCDE")
    train = _train_df()
    train = train[train["home_team"] != "N"]
    train = train[train["away_team"] != "N"]
    assert apply_low_data_fallbacks(model, train, min_matches=3) == set()
