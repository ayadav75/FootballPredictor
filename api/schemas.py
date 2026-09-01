"""Pydantic response models for the prediction API.

Field descriptions feed the auto-generated OpenAPI docs at /docs.
"""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    expected_home_goals: float = Field(description="Model xG for the home side (lambda)")
    expected_away_goals: float = Field(description="Model xG for the away side (mu)")
    home_win: float = Field(description="P(home win)")
    draw: float = Field(description="P(draw)")
    away_win: float = Field(description="P(away win)")
    model_run_id: int = Field(description="Stored model_run these ratings came from")
    fallback_rating: bool = Field(
        description="True if either team's rating was assigned rather than "
        "fitted (e.g. a newly promoted side) — treat the prediction with "
        "lower confidence"
    )
    fallback_teams: list[str] = Field(
        description="Which of the two teams (if any) are on fallback ratings"
    )


class ScorelineResponse(BaseModel):
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    max_goals: int = Field(description="Matrix covers 0..max_goals for each side")
    score_matrix: list[list[float]] = Field(
        description="P(home_goals=i, away_goals=j): rows are home goals 0..max_goals, "
        "columns are away goals 0..max_goals. Sums to slightly under 1 "
        "(tail scorelines beyond max_goals are excluded)."
    )
    model_run_id: int
    fallback_rating: bool
    fallback_teams: list[str]


class TeamsResponse(BaseModel):
    model_run_id: int
    teams: list[str] = Field(description="All teams the model can predict for")
    fallback_teams: list[str] = Field(
        description="Subset of teams whose ratings are assigned, not fitted"
    )


class HealthResponse(BaseModel):
    status: str = Field(description='"healthy" when the DB is reachable and a model is loaded')
    database: bool = Field(description="DB connectivity check succeeded")
    model_loaded: bool = Field(description="A model_run is cached and ready to serve")
    model_run_id: int | None = Field(description="The cached run, when loaded")


class ReloadResponse(BaseModel):
    model_run_id: int = Field(description="Latest model_run now being served")
    teams: int = Field(description="Number of teams in the reloaded model")
