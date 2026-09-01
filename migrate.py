"""Create the database schema and load the cleaned match data.

Usage: python migrate.py   (Postgres from docker-compose must be running,
or set DATABASE_URL to point elsewhere.)

Idempotent: tables are created only if absent, and matches already present
(same date/home/away) are skipped on reload.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from data import load_matches
from models import Base, Match, get_engine


def upsert_matches(session: Session, df: pd.DataFrame) -> int:
    existing = {
        (d, h, a)
        for d, h, a in session.execute(
            select(Match.date, Match.home_team, Match.away_team)
        )
    }
    added = 0
    for row in df.itertuples(index=False):
        key = (row.date.date(), row.home_team, row.away_team)
        if key in existing:
            continue
        record = row._asdict()
        record["date"] = row.date.date()
        # NaN odds -> NULL
        session.add(Match(**{k: (None if pd.isna(v) else v) for k, v in record.items()}))
        added += 1
    return added


def main():
    engine = get_engine()
    Base.metadata.create_all(engine)
    df = load_matches(Path(__file__).parent)
    with Session(engine) as session:
        added = upsert_matches(session, df)
        session.commit()
        total = session.scalar(select(Match.id).order_by(Match.id.desc()).limit(1))
    print(f"Schema ready. Inserted {added} new matches ({len(df)} in source, max id {total}).")


if __name__ == "__main__":
    main()
