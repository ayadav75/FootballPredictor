# Engineered match features

Computed by `features.py`, stored in the `match_features` table (one row per
match, linked to `matches.id`). These are **not** inputs to Dixon-Coles — the
DC model derives everything from goals directly. They are groundwork for a
secondary blending model that can correct DC where it is known to be blind:
short-term form swings, fixture congestion, and opponent-specific history.

## Leakage guarantee

Every feature for a match is computed only from matches **strictly before
that match's date**. The implementation walks fixtures in chronological order
and reads a team's history before appending the current match to it, so a
match can never see its own result or any later one. `test_features.py`
verifies this directly: changing the result of a later match does not change
any earlier match's features.

Early-season rows have fewer than 5 prior matches; rather than dropping or
padding them, `*_form_matches` / `h2h_matches` record how many matches the
window actually contains so a downstream model can normalize (e.g. points per
game) or learn a "low information" signal.

## Features

### Rolling form (last 5 matches, either venue)

| Column | Meaning |
|---|---|
| `home_form_matches`, `away_form_matches` | How many matches the window holds (0–5) |
| `home_form_points`, `away_form_points` | League points earned (3/1/0) over the window |
| `home_form_goals_for`, `away_form_goals_for` | Goals scored over the window |
| `home_form_goals_against`, `away_form_goals_against` | Goals conceded over the window |

**Why:** Dixon-Coles' exponential decay adapts over weeks-to-months; a
5-match window reacts in days-to-weeks. It captures streaks a slow-moving
rating misses — a new manager bounce, an injury crisis, a collapse in
confidence. Splitting points from goals matters: a team winning 1-0 four
times running looks identical to one winning 4-0 on points but very
different on goals, and goals are the better predictor of future results
(which is exactly the premise of Dixon-Coles itself).

### Rest days

| Column | Meaning |
|---|---|
| `home_rest_days`, `away_rest_days` | Days since the team's previous match in the dataset; NULL for a team's first appearance |

**Why:** Fixture congestion (midweek European games, cup replays, the
Christmas schedule) measurably depresses performance, and the *difference*
in rest between the two sides is a classic edge the base model cannot see —
Dixon-Coles treats a team playing its third game in seven days identically
to one coming off a free week. Caveats a consumer should handle: values are
computed within this dataset only, so they miss cup/European matches (they
overstate rest for teams in many competitions), and the summer break shows
up as a ~90-day value — cap or bucket before feeding to a model.

### Head-to-head (last 5 meetings, either venue)

| Column | Meaning |
|---|---|
| `h2h_matches` | Meetings found in the dataset window (0–5) |
| `h2h_home_team_wins` | Of those, won by the *current* home team |
| `h2h_draws` | Drawn |
| `h2h_away_team_wins` | Won by the current away team |

**Why:** Some matchups have persistent stylistic patterns ("bogey team"
effects) that survive after controlling for team strength — a low-block side
that repeatedly frustrates a possession-heavy one, or derbies that draw more
often than ratings imply. The signal is weak and noisy (5 meetings spans
2–3 seasons), which is precisely why it belongs in a blending model that can
learn how much to trust it, rather than hand-weighted into the base model.

## Design choices worth defending in an interview

- **Counts, not rates, plus a window-size column** — keeps the stored data
  raw and lets the modeling layer choose the normalization instead of baking
  one in.
- **Venue-agnostic windows** — home/away form splits were considered but
  halve the effective window (2–3 matches), which is mostly noise; the DC
  model already carries a home-advantage term.
- **Chronological-walk implementation over pandas rolling windows** — a few
  lines longer but the no-leakage property is visible by inspection, and it
  handles the awkward cases (promoted teams, season boundaries, unequal
  match counts per team) without special-casing.
