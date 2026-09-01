"""Tests for the engineered features: correctness on a hand-built fixture
and a direct no-leakage check."""

import pandas as pd
import pytest

from features import compute_features


def make_matches(rows):
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def small_history():
    return make_matches([
        ("2023-08-01", "A", "B", 2, 0),  # A beats B
        ("2023-08-05", "B", "C", 1, 1),  # draw
        ("2023-08-09", "C", "A", 0, 3),  # A beats C
        ("2023-08-15", "A", "B", 1, 1),  # draw
        ("2023-08-20", "B", "A", 2, 1),  # B beats A
    ])


def test_form_uses_only_prior_matches(small_history):
    feats = compute_features(small_history)
    first = feats.iloc[0]
    assert first["home_form_matches"] == 0
    assert first["home_form_points"] == 0
    assert pd.isna(first["home_rest_days"])

    # Match 4 (A vs B on Aug 15): A has won twice before it (6 pts, 5-0 goals)
    fourth = feats.iloc[3]
    assert fourth["home_form_matches"] == 2
    assert fourth["home_form_points"] == 6
    assert fourth["home_form_goals_for"] == 5
    assert fourth["home_form_goals_against"] == 0
    # B has lost to A then drawn C: 1 point
    assert fourth["away_form_matches"] == 2
    assert fourth["away_form_points"] == 1


def test_rest_days(small_history):
    feats = compute_features(small_history)
    # Aug 15: A last played Aug 9, B last played Aug 5
    assert feats.iloc[3]["home_rest_days"] == 6
    assert feats.iloc[3]["away_rest_days"] == 10


def test_h2h_perspective(small_history):
    feats = compute_features(small_history)
    # Final match (B home vs A): prior meetings A 2-0 B and A 1-1 B
    last = feats.iloc[4]
    assert last["h2h_matches"] == 2
    assert last["h2h_home_team_wins"] == 0   # B (current home) has no wins
    assert last["h2h_draws"] == 1
    assert last["h2h_away_team_wins"] == 1   # A won one


def test_no_leakage_from_future_results(small_history):
    """Changing a later result must not change any earlier match's features."""
    feats_before = compute_features(small_history)
    altered = small_history.copy()
    altered.loc[4, ["home_goals", "away_goals"]] = [9, 0]  # rewrite the last match
    feats_after = compute_features(altered)
    pd.testing.assert_frame_equal(feats_before, feats_after)


def test_form_window_caps_at_five():
    rows = [(f"2023-08-{d:02d}", "A", "B", 1, 0) for d in range(1, 10)]
    feats = compute_features(make_matches(rows))
    last = feats.iloc[-1]
    assert last["home_form_matches"] == 5
    assert last["home_form_points"] == 15
    assert last["h2h_matches"] == 5
    assert last["h2h_home_team_wins"] == 5
