"""FastAPI service serving Dixon-Coles predictions from the persisted model.

Run with:  uvicorn api.main:app --reload
Docs at:   http://localhost:8000/docs

The model is loaded from Postgres once at startup and cached in app.state;
POST /reload refreshes it to the latest model_run without a restart.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import text

from api.schemas import (
    HealthResponse,
    PredictionResponse,
    ReloadResponse,
    ScorelineResponse,
    TeamsResponse,
)
from model import DixonColes
from models import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("api")


def _load_model(app: FastAPI) -> DixonColes:
    model = DixonColes.load(app.state.engine)
    app.state.model = model
    logger.info(
        "model loaded: model_run_id=%s teams=%d fallback_teams=%s",
        model.model_run_id, len(model.teams), sorted(model.fallback_teams) or "none",
    )
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = get_engine()
    app.state.model = None
    try:
        _load_model(app)
    except Exception as exc:
        # Serve /health as unhealthy rather than crash-looping; /reload can
        # recover once the DB is up and a model_run exists.
        logger.error("startup model load failed: %s", exc)
    yield
    app.state.engine.dispose()


app = FastAPI(
    title="Football Predictor API",
    description="Premier League match outcome predictions from a Dixon-Coles "
    "model fitted on historical results (see /docs endpoints).",
    lifespan=lifespan,
)


def get_model(request: Request) -> DixonColes:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Check DB connectivity and that a "
            "model_run exists (run backtest.py or fit+save), then POST /reload.",
        )
    return model


def validate_teams(model: DixonColes, home: str, away: str) -> None:
    unknown = [t for t in (home, away) if t not in model.attack]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown team(s): {', '.join(unknown)}",
                "valid_teams": model.teams,
            },
        )
    if home == away:
        raise HTTPException(status_code=400, detail="home and away must differ")


def log_prediction(endpoint: str, model: DixonColes, home: str, away: str,
                   started: float, fallback: list[str]) -> None:
    logger.info(
        "%s home=%r away=%r model_run_id=%s latency_ms=%.1f fallback=%s",
        endpoint, home, away, model.model_run_id,
        (time.perf_counter() - started) * 1000, sorted(fallback) or "none",
    )


@app.get("/predict", response_model=PredictionResponse)
def predict(home: str, away: str, request: Request):
    """1X2 probabilities and expected goals for a fixture."""
    started = time.perf_counter()
    model = get_model(request)
    validate_teams(model, home, away)
    pred = model.predict_match(home, away)
    fallback = [t for t in (home, away) if t in model.fallback_teams]
    log_prediction("/predict", model, home, away, started, fallback)
    return PredictionResponse(
        home_team=home,
        away_team=away,
        expected_home_goals=pred.expected_home_goals,
        expected_away_goals=pred.expected_away_goals,
        home_win=pred.home_win,
        draw=pred.draw,
        away_win=pred.away_win,
        model_run_id=model.model_run_id,
        fallback_rating=bool(fallback),
        fallback_teams=fallback,
    )


@app.get("/predict/{home}/{away}/scoreline", response_model=ScorelineResponse)
def predict_scoreline(home: str, away: str, request: Request, max_goals: int = 5):
    """Full scoreline probability matrix (0..max_goals each way)."""
    started = time.perf_counter()
    model = get_model(request)
    validate_teams(model, home, away)
    if not 1 <= max_goals <= 10:
        raise HTTPException(status_code=400, detail="max_goals must be 1-10")
    pred = model.predict_match(home, away, max_goals=max_goals)
    fallback = [t for t in (home, away) if t in model.fallback_teams]
    log_prediction("/predict/scoreline", model, home, away, started, fallback)
    return ScorelineResponse(
        home_team=home,
        away_team=away,
        expected_home_goals=pred.expected_home_goals,
        expected_away_goals=pred.expected_away_goals,
        max_goals=max_goals,
        score_matrix=pred.score_matrix.to_numpy().tolist(),
        model_run_id=model.model_run_id,
        fallback_rating=bool(fallback),
        fallback_teams=fallback,
    )


@app.get("/teams", response_model=TeamsResponse)
def teams(request: Request):
    """Teams the served model knows — validate against this before /predict."""
    model = get_model(request)
    return TeamsResponse(
        model_run_id=model.model_run_id,
        teams=model.teams,
        fallback_teams=sorted(model.fallback_teams),
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    """DB connectivity plus a served model — 503 when either is missing."""
    db_ok = False
    try:
        with request.app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        logger.warning("health check DB failure: %s", exc)
    model = request.app.state.model
    body = HealthResponse(
        status="healthy" if (db_ok and model is not None) else "unhealthy",
        database=db_ok,
        model_loaded=model is not None,
        model_run_id=model.model_run_id if model else None,
    )
    if body.status != "healthy":
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


@app.post("/reload", response_model=ReloadResponse)
def reload(request: Request):
    """Re-load the latest model_run from the DB without restarting."""
    try:
        model = _load_model(request.app)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Reload failed: {exc}")
    return ReloadResponse(model_run_id=model.model_run_id, teams=len(model.teams))
