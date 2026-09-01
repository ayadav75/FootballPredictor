"""Unit tests for the Dixon-Coles model."""

import numpy as np
import pandas as pd
import pytest

from model import DixonColes, tau


def make_model(attack, defense, home_advantage=1.0, rho=0.0):
    """Build a DixonColes with hand-set parameters (no fitting)."""
    m = DixonColes()
    m.teams = sorted(attack)
    m.attack = attack
    m.defense = defense
    m.home_advantage = home_advantage
    m.rho = rho
    return m


@pytest.fixture
def equal_teams_model():
    return make_model(
        attack={"A": 1.0, "B": 1.0},
        defense={"A": 1.3, "B": 1.3},
        home_advantage=1.0,  # no home edge, so the matchup is fully symmetric
        rho=-0.05,
    )


def test_equal_ratings_give_symmetric_distribution(equal_teams_model):
    pred = equal_teams_model.predict_match("A", "B")
    matrix = pred.score_matrix.to_numpy()
    # P(home i, away j) should equal P(home j, away i)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert pred.home_win == pytest.approx(pred.away_win, abs=1e-12)


def test_probability_matrix_sums_to_one(equal_teams_model):
    pred = equal_teams_model.predict_match("A", "B", max_goals=15)
    assert pred.score_matrix.to_numpy().sum() == pytest.approx(1.0, abs=1e-6)
    # 1X2 probabilities partition the outcome space
    assert pred.home_win + pred.draw + pred.away_win == pytest.approx(1.0, abs=1e-6)


def test_default_matrix_covers_most_mass(equal_teams_model):
    pred = equal_teams_model.predict_match("A", "B")  # default 0-5 grid
    assert pred.score_matrix.to_numpy().sum() > 0.97


def test_home_advantage_shifts_probabilities():
    model = make_model(
        attack={"A": 1.0, "B": 1.0},
        defense={"A": 1.3, "B": 1.3},
        home_advantage=1.3,
        rho=-0.05,
    )
    pred = model.predict_match("A", "B")
    assert pred.home_win > pred.away_win


def test_tau_matches_paper_definition():
    lam, mu, rho = 1.4, 1.1, -0.08
    assert tau(0, 0, lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
    assert tau(0, 1, lam, mu, rho) == pytest.approx(1 + lam * rho)
    assert tau(1, 0, lam, mu, rho) == pytest.approx(1 + mu * rho)
    assert tau(1, 1, lam, mu, rho) == pytest.approx(1 - rho)
    assert tau(3, 2, lam, mu, rho) == pytest.approx(1.0)


def _synthetic_matches(seed=0):
    """Small synthetic fixture: round-robin schedule with known-ish strengths."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    true_attack = {"A": 1.5, "B": 1.2, "C": 1.0, "D": 1.0, "E": 0.8, "F": 0.7}
    true_defense = {"A": 1.0, "B": 1.1, "C": 1.3, "D": 1.3, "E": 1.5, "F": 1.6}
    rows = []
    date = pd.Timestamp("2023-08-01")
    for round_ in range(6):  # 6 double round-robins for enough data
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = true_attack[h] * true_defense[a] * 1.25
                mu = true_attack[a] * true_defense[h]
                rows.append({
                    "date": date,
                    "home_team": h,
                    "away_team": a,
                    "home_goals": rng.poisson(lam),
                    "away_goals": rng.poisson(mu),
                })
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_fit_is_deterministic():
    matches = _synthetic_matches()
    m1 = DixonColes(decay_rate=0.001).fit(matches)
    m2 = DixonColes(decay_rate=0.001).fit(matches)
    assert m1.attack == m2.attack
    assert m1.defense == m2.defense
    assert m1.home_advantage == m2.home_advantage
    assert m1.rho == m2.rho
    p1 = m1.predict_match("A", "F")
    p2 = m2.predict_match("A", "F")
    pd.testing.assert_frame_equal(p1.score_matrix, p2.score_matrix)


def test_fit_recovers_ordering_and_normalization():
    matches = _synthetic_matches()
    m = DixonColes(decay_rate=0.0).fit(matches)
    # Attack ratings average 1 by construction
    assert np.mean(list(m.attack.values())) == pytest.approx(1.0, abs=1e-8)
    # Best attacker and worst attacker recovered
    assert max(m.attack, key=m.attack.get) == "A"
    assert min(m.attack, key=m.attack.get) == "F"
    assert m.home_advantage > 1.0


def test_save_load_round_trip():
    """A model saved to the DB and loaded back gives identical predictions.
    Uses in-memory SQLite so the test needs no running Postgres."""
    from sqlalchemy import create_engine

    from models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    original = DixonColes(decay_rate=0.002).fit(_synthetic_matches())
    run_id = original.save(
        engine,
        train_start=pd.Timestamp("2023-08-01").date(),
        train_end=pd.Timestamp("2024-02-01").date(),
        n_training_matches=180,
        fallback_teams={"F"},
    )
    loaded = DixonColes.load(engine, run_id)

    assert loaded.attack == pytest.approx(original.attack)
    assert loaded.defense == pytest.approx(original.defense)
    assert loaded.home_advantage == pytest.approx(original.home_advantage)
    assert loaded.rho == pytest.approx(original.rho)
    assert loaded.decay_rate == original.decay_rate
    p1 = original.predict_match("A", "E")
    p2 = loaded.predict_match("A", "E")
    pd.testing.assert_frame_equal(p1.score_matrix, p2.score_matrix)

    # load() with no id returns the latest run
    latest = DixonColes.load(engine)
    assert latest.attack == pytest.approx(original.attack)


def test_unknown_team_raises():
    model = make_model(attack={"A": 1.0}, defense={"A": 1.3})
    with pytest.raises(KeyError):
        model.predict_match("A", "Narnia")
