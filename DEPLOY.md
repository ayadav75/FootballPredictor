# Deploying to Render

The repo ships a Blueprint (`render.yaml`) defining two resources:

- **football-predictor-api** — Docker web service built from `api/Dockerfile`
  with the repo root as build context (the API imports `model.py`/`models.py`
  from there). Health-checked on `/health`. The container binds to Render's
  `$PORT` automatically.
- **football-predictor-db** — managed Postgres. Its connection string is
  injected into the API as `DATABASE_URL` via a Blueprint reference — no
  hardcoded credentials anywhere. Render hands out a `postgres://` URL;
  `models.get_engine()` rewrites the scheme to `postgresql+psycopg://` for
  SQLAlchemy, so no manual editing is needed.

## Steps

1. Push the repo to GitHub (Render deploys from the repo).
2. Render dashboard → **Blueprints → New Blueprint Instance** → select the
   repo. Render reads `render.yaml` and provisions both resources.
3. **Seed the database (one-time, required).** The API serves from a stored
   `model_run`; a fresh managed Postgres has none, so `/health` will report
   503 (unhealthy: `model_loaded: false`) until seeded. The docker-compose
   init-container doesn't apply to a managed DB, so run the seeder yourself.

### Seeding option A — from your machine (works on the free tier)

Grab the **External Database URL** from the Render dashboard
(football-predictor-db → Connect → External Database URL), then from the
repo root:

```bash
DATABASE_URL="<external-database-url>" .venv/bin/python seed.py
```

This creates the tables, loads the 1520 matches, fits the model on
22-23/23-24/24-25, and saves it as model_run 1. It is idempotent — re-running
against a seeded DB changes nothing. Afterwards, either wait for the next
health check or hit `POST /reload` on the service URL so the running API
picks up the model immediately:

```bash
curl -X POST https://football-predictor-api.onrender.com/reload
```

### Seeding option B — Render shell (paid instances only)

The free tier has no shell access; on a paid instance you can instead open
the service's **Shell** tab and run `python seed.py` — the image contains
the season CSVs precisely so this works.

## Environment variables

`FOOTBALL_DATA_API_KEY` (for `/fixtures/upcoming`) is declared with
`sync: false` in `render.yaml`, so Render prompts for the value in the
dashboard when applying the Blueprint — paste the token from your
football-data.org account (locally it lives in the gitignored `.env`).
Kalshi market data needs no key.

## Free-tier caveats worth knowing

- The web service **spins down after ~15 minutes idle**; the first request
  after that takes ~30-60s (cold start + model load from Postgres). Not a
  bug — retry or keep-alive-ping it.
- Render's **free Postgres expires after 30 days** unless upgraded. The DB
  is fully reproducible (`seed.py` rebuilds everything from the CSVs in the
  repo), so expiry loses nothing irreplaceable — but note backtest rows
  written by `backtest.py` would need re-running.
- Logs: the API's structured prediction logs go to stdout, visible in the
  service's **Logs** tab.
