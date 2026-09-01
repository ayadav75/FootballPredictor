# Build from the REPO ROOT (the api imports model.py/models.py/data.py from
# there):  docker build -f api/Dockerfile .
# docker-compose.yml and render.yaml are already set up this way.

# --- Stage 1: install runtime deps into a clean venv -------------------------
FROM python:3.14-slim AS builder

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Stage 2: runtime image (no pip cache, no build tools, no test deps) -----
FROM python:3.14-slim

RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# App code + the raw season data (needed by seed.py to fit a model on an
# empty database; also lets a one-off seed run on the deploy host).
COPY data.py model.py models.py migrate.py backtest.py seed.py ./
COPY 22-23.csv 23-24.csv 24-25.csv 25-26.csv ./
COPY api/ api/

USER appuser
EXPOSE 8000

# $PORT is set by Render; defaults to 8000 for local docker/compose.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
