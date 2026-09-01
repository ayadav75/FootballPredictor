"""Dixon-Coles (1997) match outcome model.

Each team i has an attack strength alpha_i and a defense weakness beta_i
(higher beta = concedes more). One global home-advantage multiplier gamma.

For a match with home team h and away team a:
    expected home goals  lambda = alpha_h * beta_a * gamma
    expected away goals  mu     = alpha_a * beta_h

Goals are Poisson with the Dixon-Coles tau correction on the low scorelines
(0-0, 1-0, 0-1, 1-1), controlled by the dependence parameter rho. Fit by
weighted maximum likelihood with exponential time decay, so recent matches
count more than old ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Floor for probabilities/likelihood terms before taking logs, so a rho that
# wanders outside its valid range mid-optimization can't produce log(<=0).
_EPS = 1e-10


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score correction factor (vectorized).

    Multiplies the independent-Poisson probability. Equals 1 everywhere
    except the four scorelines 0-0, 1-0, 0-1, 1-1.
    """
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), home_goals.shape)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), home_goals.shape)

    out = np.ones(home_goals.shape, dtype=float)
    out = np.where((home_goals == 0) & (away_goals == 0), 1 - lam * mu * rho, out)
    out = np.where((home_goals == 0) & (away_goals == 1), 1 + lam * rho, out)
    out = np.where((home_goals == 1) & (away_goals == 0), 1 + mu * rho, out)
    out = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, out)
    return out


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    score_matrix: pd.DataFrame  # rows = home goals, cols = away goals
    home_win: float
    draw: float
    away_win: float

    def __str__(self) -> str:
        return (
            f"{self.home_team} vs {self.away_team}: "
            f"xG {self.expected_home_goals:.2f}-{self.expected_away_goals:.2f} | "
            f"H {self.home_win:.1%}  D {self.draw:.1%}  A {self.away_win:.1%}"
        )


@dataclass
class DixonColes:
    """Weighted maximum-likelihood Dixon-Coles model.

    decay_rate is the exponential time-decay constant per *day*: a match
    played t days before the most recent training match gets weight
    exp(-decay_rate * t). The Dixon-Coles paper's xi = 0.0065 per half-week
    corresponds to about 0.0019 per day.
    """

    decay_rate: float = 0.0019
    teams: list[str] = field(default_factory=list, init=False)
    attack: dict[str, float] = field(default_factory=dict, init=False)
    defense: dict[str, float] = field(default_factory=dict, init=False)
    home_advantage: float = field(default=np.nan, init=False)
    rho: float = field(default=np.nan, init=False)
    # Set by load(): which stored run this model came from, and which teams
    # carry assigned (not fitted) ratings. None/empty for a freshly fit model.
    model_run_id: int | None = field(default=None, init=False)
    fallback_teams: set[str] = field(default_factory=set, init=False)

    def fit(self, matches: pd.DataFrame) -> "DixonColes":
        """Fit on a DataFrame with columns date, home_team, away_team,
        home_goals, away_goals (the output of data.load_matches)."""
        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        idx = {team: i for i, team in enumerate(self.teams)}
        n = len(self.teams)

        home_idx = matches["home_team"].map(idx).to_numpy()
        away_idx = matches["away_team"].map(idx).to_numpy()
        home_goals = matches["home_goals"].to_numpy()
        away_goals = matches["away_goals"].to_numpy()

        days_before_latest = (matches["date"].max() - matches["date"]).dt.days.to_numpy()
        weights = np.exp(-self.decay_rate * days_before_latest)

        def unpack(params):
            # Attack strengths are renormalized to average exactly 1, which
            # removes the scale degeneracy (attack * c, defense / c) and is
            # the conventional identifiability constraint.
            attack = np.exp(params[:n])
            attack = attack / attack.mean()
            defense = np.exp(params[n : 2 * n])
            home_adv = np.exp(params[2 * n])
            rho = params[2 * n + 1]
            return attack, defense, home_adv, rho

        def negative_log_likelihood(params):
            attack, defense, home_adv, rho = unpack(params)
            lam = attack[home_idx] * defense[away_idx] * home_adv
            mu = attack[away_idx] * defense[home_idx]
            log_lik = (
                np.log(np.maximum(tau(home_goals, away_goals, lam, mu, rho), _EPS))
                + poisson.logpmf(home_goals, lam)
                + poisson.logpmf(away_goals, mu)
            )
            return -np.sum(weights * log_lik)

        x0 = np.concatenate([
            np.zeros(n),                    # log attack
            np.full(n, np.log(1.3)),        # log defense (~avg goals scale)
            [np.log(1.25)],                 # log home advantage
            [-0.05],                        # rho
        ])
        result = minimize(
            negative_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        if not result.success:
            raise RuntimeError(f"Dixon-Coles fit failed: {result.message}")

        attack, defense, home_adv, rho = unpack(result.x)
        self.attack = dict(zip(self.teams, attack))
        self.defense = dict(zip(self.teams, defense))
        self.home_advantage = float(home_adv)
        self.rho = float(rho)
        return self

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        for team in (home_team, away_team):
            if team not in self.attack:
                raise KeyError(f"Unknown team: {team!r}. Known: {self.teams}")
        lam = self.attack[home_team] * self.defense[away_team] * self.home_advantage
        mu = self.attack[away_team] * self.defense[home_team]
        return lam, mu

    def predict_match(
        self, home_team: str, away_team: str, max_goals: int = 5
    ) -> MatchPrediction:
        """Scoreline probability matrix (0..max_goals each way) and 1X2 probs.

        The 1X2 probabilities are computed on a larger internal grid so they
        aren't truncated by max_goals.
        """
        lam, mu = self.expected_goals(home_team, away_team)
        full = self._score_grid(lam, mu, grid_size=max(max_goals, 12))

        home_win = np.sum(np.tril(full, -1))
        draw = np.sum(np.diag(full))
        away_win = np.sum(np.triu(full, 1))

        shown = full[: max_goals + 1, : max_goals + 1]
        score_matrix = pd.DataFrame(
            shown,
            index=pd.Index(range(max_goals + 1), name="home_goals"),
            columns=pd.Index(range(max_goals + 1), name="away_goals"),
        )
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=lam,
            expected_away_goals=mu,
            score_matrix=score_matrix,
            home_win=float(home_win),
            draw=float(draw),
            away_win=float(away_win),
        )

    def _score_grid(self, lam: float, mu: float, grid_size: int) -> np.ndarray:
        goals = np.arange(grid_size + 1)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu)
        grid = np.outer(p_home, p_away)
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        grid *= np.maximum(tau(hg, ag, lam, mu, self.rho), 0.0)
        return grid

    def save(self, engine, train_start=None, train_end=None,
             n_training_matches: int = 0,
             fallback_teams: set[str] | None = None) -> int:
        """Persist this fitted model to model_runs/team_ratings.

        Returns the new model_run id. `fallback_teams` marks ratings that
        were assigned (promoted sides) rather than fitted, so a reloaded
        model is a faithful copy. Imported lazily so the statistical model
        stays usable without the DB stack.
        """
        from sqlalchemy.orm import Session

        from models import ModelRun, TeamRating

        if not self.attack:
            raise ValueError("Cannot save an unfitted model")
        fallback_teams = fallback_teams or set()
        run = ModelRun(
            decay_rate=self.decay_rate,
            home_advantage=self.home_advantage,
            rho=self.rho,
            train_start=train_start,
            train_end=train_end,
            n_training_matches=n_training_matches,
            ratings=[
                TeamRating(
                    team=team,
                    attack=self.attack[team],
                    defense=self.defense[team],
                    is_fallback=team in fallback_teams,
                )
                for team in self.teams
            ],
        )
        with Session(engine) as session:
            session.add(run)
            session.commit()
            return run.id

    @classmethod
    def load(cls, engine, model_run_id: int | None = None) -> "DixonColes":
        """Reconstruct a fitted model from a stored model_run.

        With no id, loads the most recent run. This is how the API will get
        a ready model without refitting on startup.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from models import ModelRun

        with Session(engine) as session:
            if model_run_id is None:
                run = session.scalars(
                    select(ModelRun).order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
                ).first()
                if run is None:
                    raise LookupError("No model_runs stored yet")
            else:
                run = session.get(ModelRun, model_run_id)
                if run is None:
                    raise LookupError(f"No model_run with id {model_run_id}")
            model = cls(decay_rate=run.decay_rate)
            model.home_advantage = run.home_advantage
            model.rho = run.rho
            model.attack = {r.team: r.attack for r in run.ratings}
            model.defense = {r.team: r.defense for r in run.ratings}
            model.teams = sorted(model.attack)
            model.model_run_id = run.id
            model.fallback_teams = {r.team for r in run.ratings if r.is_fallback}
            return model

    def ratings_table(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "attack": self.attack,
                "defense": self.defense,
            })
            .rename_axis("team")
            .sort_values("attack", ascending=False)
        )
