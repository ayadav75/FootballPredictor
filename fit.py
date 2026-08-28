"""Fit the Dixon-Coles model on 22-23 through 24-25 and show example predictions.

25-26 is held out entirely for the upcoming backtest — it is never passed to
the model here.
"""

from pathlib import Path

from data import load_matches
from model import DixonColes

TRAIN_SEASONS = ["22-23", "23-24", "24-25"]


def main():
    here = Path(__file__).parent
    matches = load_matches(here)
    train = matches[matches["season"].isin(TRAIN_SEASONS)]
    print(f"Training on {len(train)} matches "
          f"({train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d})")

    model = DixonColes(decay_rate=0.0019).fit(train)

    print(f"\nHome advantage: {model.home_advantage:.3f}")
    print(f"Rho (low-score dependence): {model.rho:.4f}")
    print("\nRatings (attack normalized to average 1; higher defense = leakier):")
    print(model.ratings_table().round(3).to_string())

    for home, away in [
        ("Man City", "Liverpool"),
        ("Arsenal", "Everton"),
        ("Bournemouth", "Man City"),
    ]:
        pred = model.predict_match(home, away)
        print(f"\n{pred}")
        print("Scoreline probabilities (%):")
        print((pred.score_matrix * 100).round(1).to_string())


if __name__ == "__main__":
    main()
