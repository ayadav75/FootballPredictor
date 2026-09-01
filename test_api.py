"""API tests using FastAPI's TestClient against a temporary SQLite DB
seeded with a small fitted model (no Postgres needed to run the suite)."""

import os

import pytest
from fastapi.testclient import TestClient

from model import DixonColes
from models import Base, get_engine
from test_model import _synthetic_matches


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    engine = get_engine()
    Base.metadata.create_all(engine)
    matches = _synthetic_matches()
    model = DixonColes(decay_rate=0.001).fit(matches)
    # Mark F as fallback-rated to exercise the honesty flag end to end
    model.save(
        engine,
        train_start=matches["date"].min().date(),
        train_end=matches["date"].max().date(),
        n_training_matches=len(matches),
        fallback_teams={"F"},
    )
    engine.dispose()

    from api.main import app

    with TestClient(app) as test_client:  # context manager runs the lifespan
        yield test_client


def test_predict_valid(client):
    resp = client.get("/predict", params={"home": "A", "away": "B"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["home_team"] == "A"
    assert body["home_win"] + body["draw"] + body["away_win"] == pytest.approx(1.0, abs=1e-6)
    assert body["expected_home_goals"] > 0
    assert body["model_run_id"] == 1
    assert body["fallback_rating"] is False
    assert body["fallback_teams"] == []


def test_predict_fallback_flag_surfaces(client):
    body = client.get("/predict", params={"home": "A", "away": "F"}).json()
    assert body["fallback_rating"] is True
    assert body["fallback_teams"] == ["F"]


def test_unknown_team_returns_404_with_valid_teams(client):
    resp = client.get("/predict", params={"home": "A", "away": "Narnia"})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "Narnia" in detail["message"]
    assert "A" in detail["valid_teams"]


def test_same_team_rejected(client):
    assert client.get("/predict", params={"home": "A", "away": "A"}).status_code == 400


def test_scoreline_matrix(client):
    resp = client.get("/predict/A/B/scoreline")
    assert resp.status_code == 200
    body = resp.json()
    matrix = body["score_matrix"]
    assert len(matrix) == 6 and all(len(row) == 6 for row in matrix)
    total = sum(sum(row) for row in matrix)
    assert 0.9 < total <= 1.0  # 0-5 grid captures most but not all mass
    assert body["max_goals"] == 5


def test_teams_endpoint(client):
    body = client.get("/teams").json()
    assert body["teams"] == ["A", "B", "C", "D", "E", "F"]
    assert body["fallback_teams"] == ["F"]


def test_health_healthy(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["database"] is True
    assert body["model_loaded"] is True
    assert body["model_run_id"] == 1


def test_reload(client):
    resp = client.post("/reload")
    assert resp.status_code == 200
    assert resp.json() == {"model_run_id": 1, "teams": 6}
