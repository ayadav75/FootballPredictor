"""Backtest the Dixon-Coles model on the held-out 25-26 season against
de-vigged Pinnacle closing odds.

- Fits on 22-23/23-24/24-25 only; 25-26 never enters the fit.
- Promoted teams unseen in training (Sunderland) get fallback ratings: the
  average attack/defense of the sides relegated at the end of the final
  training season — i.e. "a newly promoted team is roughly as strong as an
  average relegated team". Reported separately so they can't hide.
- Pinnacle closing odds are missing from mid-January 2026 onward in the
  source data (170 matches), so the head-to-head comparison covers the 210
  matches where the market probabilities exist; model-only metrics cover
  all 380.

Outputs backtest_25-26.csv and, when the docker-compose Postgres is up,
persists the run to model_runs/team_ratings/backtest_results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import load_matches
from model import DixonColes

TRAIN_SEASONS = ["22-23", "23-24", "24-25"]
TEST_SEASON = "25-26"
RESULT_COLS = ["H", "D", "A"]


def devig(odds_home, odds_draw, odds_away):
    """Convert decimal odds to fair probabilities by removing the overround.

    Implied probabilities 1/odds sum to slightly over 1 (the bookmaker's
    margin); normalizing them to sum to exactly 1 gives the market's fair
    opinion, so the comparison with the model is like-for-like.
    """
    inv = np.column_stack([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    return inv / inv.sum(axis=1, keepdims=True)


def brier(probs: np.ndarray, outcome_onehot: np.ndarray) -> np.ndarray:
    """Multiclass Brier score per match: sum of squared errors over H/D/A.

    0 = perfect, 2 = worst. A know-nothing (1/3, 1/3, 1/3) forecast scores 0.667.
    """
    return ((probs - outcome_onehot) ** 2).sum(axis=1)


def log_loss(probs: np.ndarray, outcome_onehot: np.ndarray) -> np.ndarray:
    """Per-match negative log-likelihood of the actual outcome."""
    p_actual = (probs * outcome_onehot).sum(axis=1)
    return -np.log(np.clip(p_actual, 1e-12, None))


def add_fallback_ratings(model: DixonColes, train: pd.DataFrame, test: pd.DataFrame) -> set[str]:
    """Give teams unseen in training the mean ratings of the sides that were
    relegated at the end of the last training season. Returns the set of
    teams that received fallback ratings."""
    test_teams = set(test["home_team"]) | set(test["away_team"])
    last_season = train[train["season"] == train["season"].max()]
    last_season_teams = set(last_season["home_team"]) | set(last_season["away_team"])
    relegated = last_season_teams - test_teams

    unseen = test_teams - set(model.teams)
    if unseen:
        fb_attack = float(np.mean([model.attack[t] for t in relegated]))
        fb_defense = float(np.mean([model.defense[t] for t in relegated]))
        for team in unseen:
            model.attack[team] = fb_attack
            model.defense[team] = fb_defense
        model.teams = sorted(model.attack)
    return unseen


def run_backtest(decay_rate: float = 0.0019):
    matches = load_matches(Path(__file__).parent)
    train = matches[matches["season"].isin(TRAIN_SEASONS)]
    test = matches[matches["season"] == TEST_SEASON].reset_index(drop=True)

    model = DixonColes(decay_rate=decay_rate).fit(train)
    fallback_teams = add_fallback_ratings(model, train, test)

    preds = np.array([
        [(p := model.predict_match(m.home_team, m.away_team)).home_win, p.draw, p.away_win]
        for m in test.itertuples()
    ])
    onehot = pd.get_dummies(test["result"])[RESULT_COLS].to_numpy(dtype=float)

    results = test[["date", "home_team", "away_team", "result",
                    "home_goals", "away_goals"]].copy()
    results[["model_p_home", "model_p_draw", "model_p_away"]] = preds
    results["model_brier"] = brier(preds, onehot)
    results["model_log_loss"] = log_loss(preds, onehot)

    has_odds = test[["ps_close_home_odds", "ps_close_draw_odds", "ps_close_away_odds"]].notna().all(axis=1)
    pin = np.full((len(test), 3), np.nan)
    pin[has_odds] = devig(
        test.loc[has_odds, "ps_close_home_odds"],
        test.loc[has_odds, "ps_close_draw_odds"],
        test.loc[has_odds, "ps_close_away_odds"],
    )
    results[["pinnacle_p_home", "pinnacle_p_draw", "pinnacle_p_away"]] = pin
    results["pinnacle_brier"] = np.where(has_odds, brier(pin, onehot), np.nan)
    results["pinnacle_log_loss"] = np.where(has_odds, log_loss(pin, onehot), np.nan)

    results["brier_diff"] = results["model_brier"] - results["pinnacle_brier"]
    results["log_loss_diff"] = results["model_log_loss"] - results["pinnacle_log_loss"]
    # Total variation distance between model and market: how much they disagree,
    # independent of the actual result.
    results["disagreement"] = 0.5 * np.abs(preds - pin).sum(axis=1)
    results["uses_fallback_rating"] = (
        test["home_team"].isin(fallback_teams) | test["away_team"].isin(fallback_teams)
    )
    return model, train, results, fallback_teams


def summarize(results: pd.DataFrame, fallback_teams: set[str]) -> str:
    both = results.dropna(subset=["pinnacle_brier"])
    clean = both[~both["uses_fallback_rating"]]
    lines = []

    def block(title, df, show_pinnacle=True):
        lines.append(f"\n{title} (n={len(df)}):")
        lines.append(f"  {'':14}{'Brier':>8}  {'Log loss':>8}")
        lines.append(f"  {'Dixon-Coles':14}{df['model_brier'].mean():8.4f}  {df['model_log_loss'].mean():8.4f}")
        if show_pinnacle:
            lines.append(f"  {'Pinnacle':14}{df['pinnacle_brier'].mean():8.4f}  {df['pinnacle_log_loss'].mean():8.4f}")

    block("Head-to-head vs Pinnacle closing (matches with odds)", both)
    if fallback_teams:
        block(f"Excluding fallback-rated teams ({', '.join(sorted(fallback_teams))})", clean)
    block("Model on ALL 25-26 matches (no market data after mid-Jan)", results,
          show_pinnacle=False)

    wins = (both["brier_diff"] < 0).sum()
    lines.append(f"\nModel beats market on Brier in {wins}/{len(both)} matches "
                 f"({wins / len(both):.0%}).")

    both = both.assign(date=both["date"].dt.strftime("%Y-%m-%d"))
    show = ["date", "home_team", "away_team", "result",
            "model_p_home", "model_p_draw", "model_p_away",
            "pinnacle_p_home", "pinnacle_p_draw", "pinnacle_p_away", "disagreement"]
    lines.append("\nBiggest model-vs-market disagreements (total variation distance):")
    top = both.nlargest(8, "disagreement")[show].round(3)
    lines.append(top.to_string(index=False))

    lines.append("\nMatches where the model most beat the market (Brier diff):")
    lines.append(both.nsmallest(5, "brier_diff")[show + ["brier_diff"]].round(3).to_string(index=False))
    lines.append("\nMatches where the market most beat the model:")
    lines.append(both.nlargest(5, "brier_diff")[show + ["brier_diff"]].round(3).to_string(index=False))
    return "\n".join(lines)


def persist(model, train, results, fallback_teams) -> str:
    """Store the run in Postgres. Returns a status message; never raises on
    an unreachable DB so the backtest is still usable standalone."""
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from models import BacktestResult, Match, get_engine

        engine = get_engine()
        run_id = model.save(
            engine,
            train_start=train["date"].min().date(),
            train_end=train["date"].max().date(),
            n_training_matches=len(train),
            fallback_teams=fallback_teams,
        )
        with Session(engine) as session:
            match_ids = {
                (d, h, a): mid
                for mid, d, h, a in session.execute(
                    select(Match.id, Match.date, Match.home_team, Match.away_team)
                )
            }
            for row in results.itertuples():
                session.add(BacktestResult(
                    model_run_id=run_id,
                    match_id=match_ids[(row.date.date(), row.home_team, row.away_team)],
                    actual_result=row.result,
                    model_p_home=row.model_p_home,
                    model_p_draw=row.model_p_draw,
                    model_p_away=row.model_p_away,
                    model_brier=row.model_brier,
                    model_log_loss=row.model_log_loss,
                    pinnacle_p_home=None if pd.isna(row.pinnacle_p_home) else row.pinnacle_p_home,
                    pinnacle_p_draw=None if pd.isna(row.pinnacle_p_draw) else row.pinnacle_p_draw,
                    pinnacle_p_away=None if pd.isna(row.pinnacle_p_away) else row.pinnacle_p_away,
                    pinnacle_brier=None if pd.isna(row.pinnacle_brier) else row.pinnacle_brier,
                    pinnacle_log_loss=None if pd.isna(row.pinnacle_log_loss) else row.pinnacle_log_loss,
                ))
            session.commit()
        return f"Persisted to DB as model_run {run_id} with {len(results)} backtest rows."
    except Exception as exc:  # DB down is an expected, non-fatal condition
        return f"DB persistence skipped ({type(exc).__name__}: {exc})"


def main():
    model, train, results, fallback_teams = run_backtest()
    print(f"Fitted on {len(train)} matches; home_advantage={model.home_advantage:.3f}, "
          f"rho={model.rho:.4f}")
    if fallback_teams:
        print(f"Fallback ratings assigned to: {sorted(fallback_teams)} "
              f"(mean of last season's relegated sides)")
    print(summarize(results, fallback_teams))

    out = Path(__file__).parent / "backtest_25-26.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved per-match results to {out}")
    print(persist(model, train, results, fallback_teams))


if __name__ == "__main__":
    main()
