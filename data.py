"""Load and clean Premier League match data from football-data.co.uk CSVs.

Each season lives in its own CSV named like "22-23.csv". The files carry many
bookmaker columns; we keep only the match facts plus Pinnacle opening and
closing 1X2 odds (needed later as a baseline for backtesting).
"""

from __future__ import annotations

import difflib
import itertools
from pathlib import Path

import pandas as pd

# Raw column -> cleaned column. Order here is the output column order.
COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "PSH": "ps_home_odds",
    "PSD": "ps_draw_odds",
    "PSA": "ps_away_odds",
    "PSCH": "ps_close_home_odds",
    "PSCD": "ps_close_draw_odds",
    "PSCA": "ps_close_away_odds",
}

SEASONS = ["22-23", "23-24", "24-25", "25-26"]


def load_season(csv_path: Path | str, season: str) -> pd.DataFrame:
    """Load one season's CSV, keeping only the columns we need.

    utf-8-sig strips the UTF-8 BOM present in some files (24-25, 25-26)
    and is a no-op for the ones without it.
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing expected columns {missing}")
    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df.insert(1, "season", season)
    return df


def load_matches(data_dir: Path | str, seasons: list[str] = SEASONS) -> pd.DataFrame:
    """Load, concatenate, and clean all seasons. Sorted by date."""
    data_dir = Path(data_dir)
    frames = [load_season(data_dir / f"{season}.csv", season) for season in seasons]
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def find_suspect_team_names(df: pd.DataFrame, threshold: float = 0.8) -> list[tuple[str, str]]:
    """Flag pairs of distinct team names that look like the same club.

    football-data.co.uk is consistent within its own files, but a typo or a
    renamed club ("Man United" vs "Manchester Utd") would silently split one
    team into two model entities, so we check anyway.
    """
    names = sorted(set(df["home_team"]) | set(df["away_team"]))

    def norm(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    suspects = []
    for a, b in itertools.combinations(names, 2):
        ratio = difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
        if ratio >= threshold or norm(a) in norm(b) or norm(b) in norm(a):
            suspects.append((a, b))
    return suspects


def summarize(df: pd.DataFrame) -> str:
    lines = [
        f"Total matches: {len(df)}",
        f"Date range:    {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
        f"Unique teams:  {len(set(df['home_team']) | set(df['away_team']))}",
        "Matches per season:",
    ]
    for season, count in df.groupby("season").size().items():
        lines.append(f"  {season}: {count}")
    suspects = find_suspect_team_names(df)
    if suspects:
        lines.append("SUSPICIOUS near-duplicate team names (check these!):")
        lines.extend(f"  {a!r} vs {b!r}" for a, b in suspects)
    else:
        lines.append("Team names: no suspicious near-duplicates found.")
    return "\n".join(lines)


if __name__ == "__main__":
    here = Path(__file__).parent
    matches = load_matches(here)
    print(summarize(matches))
    out = here / "matches_clean.csv"
    matches.to_csv(out, index=False)
    print(f"\nSaved cleaned dataset to {out}")
