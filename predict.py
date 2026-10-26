"""Step 4: plug in any player (or your own numbers) and get a predicted value.

    python predict.py "Bukayo Saka"
    python predict.py "Saka" --season 2023/24
    python predict.py --goals 15 --assists 8 --minutes 2800 --age 22 --position FWD
"""
import argparse
import unicodedata

import pandas as pd

from config import DATASET_FILE, FAIR_BAND, MIN_MINUTES, POSITIONS
from features import add_model_features
from model_utils import baseline, contributions, load_model, predict


def plain(text: str) -> str:
    """Lowercase and strip accents so 'odegaard' finds 'Ødegaard'."""
    text = text.replace("Ø", "O").replace("ø", "o")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def euros(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}€{abs(x) / 1e6:.1f}M"


def verdict(predicted: float, market: float) -> str:
    gap = predicted / market - 1
    if gap > FAIR_BAND:
        return f"model says UNDERPRICED (worth {gap:.0%} more)"
    if gap < -FAIR_BAND:
        return f"model says OVERPRICED (worth {-gap:.0%} less)"
    return f"model says FAIRLY PRICED ({gap:+.0%})"


def find_player(name: str, season: str | None) -> pd.Series:
    df = pd.read_csv(DATASET_FILE)
    season = season or df["season_label"].max()
    rows = df[(df["season_label"] == season) & df["name"].map(plain).str.contains(plain(name), regex=False)]
    if rows.empty:
        raise SystemExit(f"No player matching '{name}' with {MIN_MINUTES}+ Premier League minutes in {season}.")
    if len(rows) > 1:
        print(f"Several players match '{name}', showing the most valuable. Others: "
              + ", ".join(rows["name"].iloc[1:6]))
    return rows.iloc[0]


def show(row: pd.Series, model: dict) -> None:
    predicted = predict(row.to_frame().T, model)[0]
    print()
    if "name" in row:
        print(f"{row['name']} · {row['club']} · {row['role']} · {row['season_label']}")
    print(f"{int(row['goals'])} goals, {int(row['assists'])} assists, "
          f"{int(row['minutes']):,} minutes, age {int(row['age'])}, {row['position']}")
    print(f"\n  Predicted value: {euros(predicted)}")
    if "market_value" in row:
        print(f"  Market value:    {euros(row['market_value'])}  ->  {verdict(predicted, row['market_value'])}")

    print(f"\n  How the model got there:")
    print(f"    {'Average ' + model['test_season'] + ' player':<24} {euros(baseline(model)):>9}")
    for group, euros_added in contributions(row, model).items():
        print(f"    {group:<24} {('+' if euros_added >= 0 else '') + euros(euros_added):>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Premier League player's market value.")
    parser.add_argument("name", nargs="?", help="player name, e.g. 'Saka'")
    parser.add_argument("--season", help="season like 2024/25 (default: latest)")
    parser.add_argument("--goals", type=int)
    parser.add_argument("--assists", type=int)
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--age", type=int)
    parser.add_argument("--position", choices=POSITIONS)
    args = parser.parse_args()
    model = load_model()

    if args.name:
        row = find_player(args.name, args.season)
    else:
        custom = [args.goals, args.assists, args.minutes, args.age, args.position]
        if any(v is None for v in custom):
            parser.error("give a player name, or all of --goals --assists --minutes --age --position")
        season = int(model["test_season"][:4])
        row = add_model_features(pd.DataFrame([{
            "goals": args.goals, "assists": args.assists, "minutes": max(args.minutes, 1),
            "age": args.age, "position": args.position, "season": season,
        }])).iloc[0]
    show(row, model)


if __name__ == "__main__":
    main()
