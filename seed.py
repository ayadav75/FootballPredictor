"""One-shot initializer: make an empty database ready to serve predictions.

Runs the schema migration + match load (migrate.py), then — only if no
model_run exists yet — fits a Dixon-Coles model on the training seasons,
adds fallback ratings for teams outside them, and saves it. Idempotent:
safe to run on every `docker compose up`; a seeded DB is left untouched.

Used by the `seed` service in docker-compose.yml and as the one-off seeding
command for a freshly provisioned managed Postgres (see DEPLOY.md).
"""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

import migrate
from backtest import TRAIN_SEASONS, add_fallback_ratings
from data import load_matches
from model import DixonColes
from models import ModelRun, get_engine


def seed_model_if_missing() -> str:
    engine = get_engine()
    with Session(engine) as session:
        existing = session.scalars(select(ModelRun.id)).first()
    if existing is not None:
        return f"model_run {existing} already exists — nothing to seed."

    matches = load_matches(Path(__file__).parent)
    train = matches[matches["season"].isin(TRAIN_SEASONS)]
    test = matches[~matches["season"].isin(TRAIN_SEASONS)]
    model = DixonColes().fit(train)
    fallback_teams = add_fallback_ratings(model, train, test)
    run_id = model.save(
        engine,
        train_start=train["date"].min().date(),
        train_end=train["date"].max().date(),
        n_training_matches=len(train),
        fallback_teams=fallback_teams,
    )
    return (f"Fitted on {len(train)} matches ({', '.join(TRAIN_SEASONS)}), "
            f"fallback ratings for {sorted(fallback_teams) or 'no teams'}, "
            f"saved as model_run {run_id}.")


if __name__ == "__main__":
    migrate.main()
    print(seed_model_if_missing())
