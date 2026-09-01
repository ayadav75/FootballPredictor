"""Engineered pre-match features (see features.md for the reasoning).

Every feature for a match uses only matches strictly before that match's
date — the computation walks the fixture list in chronological order and
reads each team's history *before* appending the current match to it, so
leakage is impossible by construction.

Usage: python features.py   (writes match_features to Postgres and
match_features.csv alongside).
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

FORM_WINDOW = 5
H2H_WINDOW = 5

POINTS = {"W": 3, "D": 1, "L": 0}


def compute_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per match, aligned with the input rows.

    Input needs columns: date, home_team, away_team, home_goals, away_goals.
    Must cover the full history you want form to be built from — features for
    early rows will have small form_matches counts (documented, not dropped).
    """
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)

    team_history = defaultdict(list)  # team -> [(date, goals_for, goals_against)]
    h2h_history = defaultdict(list)   # frozenset({a, b}) -> [(winner_or_None, date)]
    rows = []

    for m in matches.itertuples():
        feats = {"date": m.date, "home_team": m.home_team, "away_team": m.away_team}

        for side, team in (("home", m.home_team), ("away", m.away_team)):
            history = team_history[team]
            recent = history[-FORM_WINDOW:]
            feats[f"{side}_rest_days"] = (
                (m.date - history[-1][0]).days if history else None
            )
            feats[f"{side}_form_matches"] = len(recent)
            feats[f"{side}_form_points"] = sum(
                POINTS["W" if gf > ga else "D" if gf == ga else "L"]
                for _, gf, ga in recent
            )
            feats[f"{side}_form_goals_for"] = sum(gf for _, gf, _ in recent)
            feats[f"{side}_form_goals_against"] = sum(ga for _, _, ga in recent)

        meetings = h2h_history[frozenset((m.home_team, m.away_team))][-H2H_WINDOW:]
        feats["h2h_matches"] = len(meetings)
        feats["h2h_home_team_wins"] = sum(
            1 for winner, _ in meetings if winner == m.home_team
        )
        feats["h2h_draws"] = sum(1 for winner, _ in meetings if winner is None)
        feats["h2h_away_team_wins"] = sum(
            1 for winner, _ in meetings if winner == m.away_team
        )
        rows.append(feats)

        # Only now does the current match enter history, so the next matches
        # (later dates) can see it but this one could not.
        team_history[m.home_team].append((m.date, m.home_goals, m.away_goals))
        team_history[m.away_team].append((m.date, m.away_goals, m.home_goals))
        winner = (
            m.home_team if m.home_goals > m.away_goals
            else m.away_team if m.away_goals > m.home_goals
            else None
        )
        h2h_history[frozenset((m.home_team, m.away_team))].append((winner, m.date))

    return pd.DataFrame(rows)


def persist(features: pd.DataFrame) -> str:
    """Write features to the match_features table (replacing existing rows)."""
    from sqlalchemy import delete, select
    from sqlalchemy.orm import Session

    from models import Match, MatchFeature, get_engine

    feature_cols = [c for c in features.columns
                    if c not in ("date", "home_team", "away_team")]
    engine = get_engine()
    with Session(engine) as session:
        match_ids = {
            (d, h, a): mid
            for mid, d, h, a in session.execute(
                select(Match.id, Match.date, Match.home_team, Match.away_team)
            )
        }
        session.execute(delete(MatchFeature))
        for row in features.itertuples():
            key = (row.date.date(), row.home_team, row.away_team)
            values = {c: getattr(row, c) for c in feature_cols}
            values = {k: (None if pd.isna(v) else v) for k, v in values.items()}
            session.add(MatchFeature(match_id=match_ids[key], **values))
        session.commit()
    return f"Wrote {len(features)} rows to match_features."


if __name__ == "__main__":
    from data import load_matches

    here = Path(__file__).parent
    features = compute_features(load_matches(here))
    features.to_csv(here / "match_features.csv", index=False)
    print(f"Computed features for {len(features)} matches "
          f"-> {here / 'match_features.csv'}")
    try:
        print(persist(features))
    except Exception as exc:
        print(f"DB persistence skipped ({type(exc).__name__}: {exc})")
