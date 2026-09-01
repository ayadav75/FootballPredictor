"""Streamlit frontend for the Football Predictor API.

Pure frontend: every number shown comes from the API — no prediction logic
here. Run with:

    streamlit run ui/app.py

Points at http://localhost:8000 by default; set API_URL to target a deployed
instance (e.g. Render) without a code change.
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str, **params):
    """GET from the API. Returns (json, None) or (None, user-facing error)."""
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    except requests.RequestException:
        return None, (
            f"Can't reach the prediction service at {API_URL}. "
            "Is the API running? (docker compose up, or set API_URL)"
        )
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        if isinstance(detail, dict):
            detail = detail.get("message", str(detail))
        return None, f"API error {resp.status_code}: {detail}"
    return resp.json(), None


st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Match Predictor")
st.caption(
    "Dixon-Coles model — probabilities served live from the prediction API "
    f"({API_URL})"
)

teams_data, err = api_get("/teams")
if err:
    st.error(err)
    st.stop()
teams = teams_data["teams"]

col_home, col_away = st.columns(2)
home = col_home.selectbox("Home team", teams, index=None, placeholder="Pick the home side")
away = col_away.selectbox("Away team", teams, index=None, placeholder="Pick the away side")

same_team = home is not None and home == away
if same_team:
    st.info("A team can't play itself — pick two different teams.")

ready = home is not None and away is not None and not same_team
if st.button("Predict", type="primary", disabled=not ready) and ready:
    pred, err = api_get("/predict", home=home, away=away)
    scoreline, err2 = api_get(f"/predict/{home}/{away}/scoreline") if not err else (None, None)
    if err or err2:
        st.error(err or err2)
        st.stop()

    if pred["fallback_rating"]:
        names = " and ".join(pred["fallback_teams"])
        st.warning(
            f"### ⚠️ Lower-confidence prediction\n"
            f"**{names}** {'are' if len(pred['fallback_teams']) > 1 else 'is'} on a "
            f"**fallback rating**: the team wasn't in the model's training data "
            f"(e.g. newly promoted, so no Premier League history to fit on). It has "
            f"been assigned an average relegation-candidate rating instead — treat "
            f"this prediction with extra skepticism.",
        )

    st.subheader(
        f"{home} {pred['expected_home_goals']:.1f} — "
        f"{pred['expected_away_goals']:.1f} {away}"
    )
    st.caption("expected goals (model λ / μ)")

    outcomes = pd.DataFrame({
        "outcome": [f"{home} win", "Draw", f"{away} win"],
        "probability": [pred["home_win"], pred["draw"], pred["away_win"]],
        "order": [0, 1, 2],
    })
    bars = (
        alt.Chart(outcomes)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", axis=alt.Axis(format="%"), title="Probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("outcome:N", sort=alt.SortField("order"), title=None),
            color=alt.Color("outcome:N", legend=None),
            tooltip=[alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=140)
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("probability:Q", format=".1%"), color=alt.value("#888888")
    )
    st.altair_chart(bars + labels, width="stretch")

    matrix = scoreline["score_matrix"]
    n = len(matrix)
    long = pd.DataFrame(
        [
            {"home_goals": hg, "away_goals": ag, "probability": matrix[hg][ag]}
            for hg in range(n)
            for ag in range(n)
        ]
    )
    best = long.loc[long["probability"].idxmax()]
    st.markdown(
        f"**Most likely scoreline:** {home} {int(best['home_goals'])}–"
        f"{int(best['away_goals'])} {away} ({best['probability']:.1%})"
    )
    heatmap = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("away_goals:O", title=f"{away} goals (away)"),
            y=alt.Y("home_goals:O", title=f"{home} goals (home)"),
            color=alt.Color(
                "probability:Q",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Probability", format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("home_goals:O", title="Home goals"),
                alt.Tooltip("away_goals:O", title="Away goals"),
                alt.Tooltip("probability:Q", format=".1%"),
            ],
        )
        .properties(height=320)
    )
    cell_text = (
        alt.Chart(long)
        .mark_text(fontSize=11)
        .encode(
            x="away_goals:O",
            y="home_goals:O",
            text=alt.Text("probability:Q", format=".1%"),
            color=alt.condition(
                alt.datum.probability > long["probability"].max() * 0.6,
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )
    st.altair_chart(heatmap + cell_text, width="stretch")
    st.caption(
        f"Scoreline probabilities from model run #{pred['model_run_id']}. "
        "Cells cover 0–5 goals each way; rare higher scorelines account for "
        "the small remainder."
    )
