"""SQLAlchemy ORM models and engine helpers.

Connection details default to the docker-compose Postgres service
(user/password/db all "football" on localhost:5432); override with the
DATABASE_URL environment variable if needed.

Naming note: this module (models.py) holds the *database* models; the
Dixon-Coles statistical model lives in model.py.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Engine, ForeignKey, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DEFAULT_DATABASE_URL = "postgresql+psycopg://football:football@localhost:5432/football"


def get_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    # Managed providers (e.g. Render) hand out postgres:// or postgresql://
    # URLs; SQLAlchemy needs the explicit psycopg3 driver in the scheme.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url)


class Base(DeclarativeBase):
    pass


class Match(Base):
    """One played match, as cleaned by data.py (raw facts + Pinnacle odds)."""

    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("date", "home_team", "away_team"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(index=True)
    home_team: Mapped[str]
    away_team: Mapped[str]
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    result: Mapped[str]  # H / D / A
    ps_home_odds: Mapped[float | None]
    ps_draw_odds: Mapped[float | None]
    ps_away_odds: Mapped[float | None]
    ps_close_home_odds: Mapped[float | None]
    ps_close_draw_odds: Mapped[float | None]
    ps_close_away_odds: Mapped[float | None]

    features: Mapped["MatchFeature | None"] = relationship(back_populates="match")


class ModelRun(Base):
    """One Dixon-Coles fit: global parameters + training window."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    decay_rate: Mapped[float]
    home_advantage: Mapped[float]
    rho: Mapped[float]
    train_start: Mapped[date] = mapped_column(Date)
    train_end: Mapped[date] = mapped_column(Date)
    n_training_matches: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    ratings: Mapped[list["TeamRating"]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )


class TeamRating(Base):
    """Per-team attack/defense strengths for one model run."""

    __tablename__ = "team_ratings"
    __table_args__ = (UniqueConstraint("model_run_id", "team"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    team: Mapped[str]
    attack: Mapped[float]
    defense: Mapped[float]
    # True for teams not in the training data (promoted sides) that were
    # assigned fallback ratings rather than fitted ones.
    is_fallback: Mapped[bool] = mapped_column(default=False)

    model_run: Mapped[ModelRun] = relationship(back_populates="ratings")


class BacktestResult(Base):
    """Held-out prediction for one match from one model run, with the
    de-vigged Pinnacle closing probabilities and both scores."""

    __tablename__ = "backtest_results"
    __table_args__ = (UniqueConstraint("model_run_id", "match_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    actual_result: Mapped[str]

    model_p_home: Mapped[float]
    model_p_draw: Mapped[float]
    model_p_away: Mapped[float]
    model_brier: Mapped[float]
    model_log_loss: Mapped[float]

    # Null when Pinnacle closing odds are missing for the match.
    pinnacle_p_home: Mapped[float | None]
    pinnacle_p_draw: Mapped[float | None]
    pinnacle_p_away: Mapped[float | None]
    pinnacle_brier: Mapped[float | None]
    pinnacle_log_loss: Mapped[float | None]


class MatchFeature(Base):
    """Engineered pre-match features (see features.md). All values are
    computed only from matches strictly before this match's date."""

    __tablename__ = "match_features"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True)

    home_rest_days: Mapped[int | None]
    away_rest_days: Mapped[int | None]

    # Rolling form over each team's last `form_matches` games (up to 5).
    home_form_matches: Mapped[int]
    home_form_points: Mapped[int]
    home_form_goals_for: Mapped[int]
    home_form_goals_against: Mapped[int]
    away_form_matches: Mapped[int]
    away_form_points: Mapped[int]
    away_form_goals_for: Mapped[int]
    away_form_goals_against: Mapped[int]

    # Last up-to-5 meetings between the two clubs (either venue), counted
    # from the current home team's perspective.
    h2h_matches: Mapped[int]
    h2h_home_team_wins: Mapped[int]
    h2h_draws: Mapped[int]
    h2h_away_team_wins: Mapped[int]

    match: Mapped[Match] = relationship(back_populates="features")
