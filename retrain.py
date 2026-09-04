"""Weekly retrain job (run by .github/workflows/retrain.yml).

1. Fetch FINISHED Premier League matches for the current season from
   football-data.org and insert any not already in the matches table.
2. Refit Dixon-Coles on the full match history in the database (the
   exponential time decay downweights old seasons — no manual windowing).
3. Save the fit as a new model_run.
4. POST /reload on the live API and verify /health serves the new run.

Any failure raises and exits nonzero, so the GitHub Actions run fails
visibly rather than leaving the live API silently stale.

Env: DATABASE_URL (production Postgres), FOOTBALL_DATA_API_KEY,
RENDER_API_URL (live API base). Pass --skip-reload to retrain a database
without touching a live API (e.g. local testing).
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data import canonical_team_name
from migrate import upsert_matches
from model import DixonColes
from models import Base, Match, get_engine

FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4"


def season_start_year(d) -> int:
    return d.year if d.month >= 8 else d.year - 1


def season_label(d) -> str:
    start = season_start_year(d)
    return f"{start % 100:02d}-{(start + 1) % 100:02d}"


def ingest_team_name(source_name: str, known_teams: list[str]) -> str:
    """Canonical name for a result being ingested. Unlike prediction-time
    mapping, an unknown club here is a genuinely new team (promoted), so we
    mint a stable name from the source's by stripping the FC/AFC suffix
    (e.g. "Coventry City FC" -> "Coventry City") instead of returning None."""
    canonical = canonical_team_name(source_name, known_teams)
    if canonical:
        return canonical
    return source_name.removesuffix(" FC").removesuffix(" AFC").strip()


def fetch_finished_matches(api_key: str, season_year: int) -> pd.DataFrame:
    resp = requests.get(
        f"{FOOTBALL_DATA_API_URL}/competitions/PL/matches",
        params={"season": season_year, "status": "FINISHED"},
        headers={"X-Auth-Token": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def results_dataframe(matches_json: list[dict], known_teams: list[str]) -> pd.DataFrame:
    rows = []
    for m in matches_json:
        home_goals = m["score"]["fullTime"]["home"]
        away_goals = m["score"]["fullTime"]["away"]
        if home_goals is None or away_goals is None:
            continue
        kickoff = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        rows.append({
            "date": pd.Timestamp(kickoff.date()),
            "season": season_label(kickoff.date()),
            "home_team": ingest_team_name(m["homeTeam"]["name"], known_teams),
            "away_team": ingest_team_name(m["awayTeam"]["name"], known_teams),
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "result": "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D",
            "ps_home_odds": None, "ps_draw_odds": None, "ps_away_odds": None,
            "ps_close_home_odds": None, "ps_close_draw_odds": None,
            "ps_close_away_odds": None,
        })
    return pd.DataFrame(rows)


def load_training_matches(engine) -> pd.DataFrame:
    df = pd.read_sql(
        select(Match.date, Match.season, Match.home_team, Match.away_team,
               Match.home_goals, Match.away_goals),
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


MIN_MATCHES_FOR_FITTED_RATING = 10


def apply_low_data_fallbacks(model: DixonColes, train: pd.DataFrame,
                             min_matches: int = MIN_MATCHES_FOR_FITTED_RATING) -> set[str]:
    """Replace ratings of teams with very little data.

    Unregularized MLE on a handful of matches is degenerate — a promoted
    side that has played twice without conceding gets a near-zero defense
    parameter and produces absurd predictions (observed: 77/23/0 for a
    promoted team at home to an established one). Until a team has
    min_matches matches, give it the mean ratings of the sides relegated at
    the end of the previous season — the same "new team ~ average relegated
    team" policy the backtest uses — and return the set so save() flags
    them is_fallback.
    """
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    low_data = {t for t in model.teams if counts.get(t, 0) < min_matches}
    if not low_data:
        return set()

    seasons = sorted(train["season"].unique())
    if len(seasons) < 2:
        return set()
    current, previous = seasons[-1], seasons[-2]

    def teams_in(season: str) -> set[str]:
        rows = train[train["season"] == season]
        return set(rows["home_team"]) | set(rows["away_team"])

    relegated = teams_in(previous) - teams_in(current) - low_data
    if not relegated:
        return set()

    fb_attack = float(pd.Series([model.attack[t] for t in relegated]).mean())
    fb_defense = float(pd.Series([model.defense[t] for t in relegated]).mean())
    for team in low_data:
        model.attack[team] = fb_attack
        model.defense[team] = fb_defense
    model.fallback_teams = low_data
    return low_data


def call_api(api_url: str, run_id: int) -> None:
    reload_resp = requests.post(f"{api_url}/reload", timeout=60)
    reload_resp.raise_for_status()
    served = reload_resp.json()["model_run_id"]
    if served != run_id:
        raise RuntimeError(
            f"/reload picked up model_run {served}, expected {run_id} — "
            "is DATABASE_URL pointing at the same DB the API uses?"
        )
    health = requests.get(f"{api_url}/health", timeout=60)
    health.raise_for_status()
    body = health.json()
    if body.get("model_run_id") != run_id or body.get("status") != "healthy":
        raise RuntimeError(f"/health does not reflect the new run: {body}")


def summary_line(text: str) -> None:
    print(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(text + "\n\n")


def main() -> None:
    skip_reload = "--skip-reload" in sys.argv
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise SystemExit("FOOTBALL_DATA_API_KEY is not set")
    api_url = os.environ.get("RENDER_API_URL", "").rstrip("/")
    if not api_url and not skip_reload:
        raise SystemExit("RENDER_API_URL is not set (or pass --skip-reload)")

    engine = get_engine()
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        known_teams = sorted(
            set(session.scalars(select(Match.home_team).distinct()))
            | set(session.scalars(select(Match.away_team).distinct()))
        )
        latest = session.scalar(select(func.max(Match.date)))

    today = datetime.now(timezone.utc).date()
    season = season_start_year(today)
    if latest is None:
        raise SystemExit("matches table is empty — run migrate.py/seed.py first")
    if season_start_year(latest) < season - 1:
        raise SystemExit(
            f"Gap too large: latest stored match is {latest} but current season "
            f"is {season}. Backfill the missing season(s) before retraining."
        )

    fetched = fetch_finished_matches(api_key, season)
    new_df = results_dataframe(fetched, known_teams)
    with Session(engine) as session:
        added = upsert_matches(session, new_df) if len(new_df) else 0
        session.commit()
        total = session.scalar(select(func.count(Match.id)))
    summary_line(f"Fetched {len(new_df)} finished {season}/{season + 1} matches "
                 f"from football-data.org; {added} were new (DB now {total} matches).")

    train = load_training_matches(engine)
    model = DixonColes().fit(train)
    fallback_teams = apply_low_data_fallbacks(model, train)
    run_id = model.save(
        engine,
        train_start=train["date"].min().date(),
        train_end=train["date"].max().date(),
        n_training_matches=len(train),
        fallback_teams=fallback_teams,
    )
    summary_line(f"Refit on {len(train)} matches "
                 f"({train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}); "
                 f"saved as model_run {run_id} "
                 f"(home_advantage {model.home_advantage:.3f}, rho {model.rho:.4f}, "
                 f"{len(model.teams)} teams"
                 + (f"; fallback ratings for low-data teams: {sorted(fallback_teams)}"
                    if fallback_teams else "")
                 + ").")

    if skip_reload:
        summary_line("Skipped /reload (--skip-reload).")
    else:
        call_api(api_url, run_id)
        summary_line(f"Live API reloaded and /health confirms model_run {run_id}.")


if __name__ == "__main__":
    main()
