"""Step 2: clean the raw data with pandas and build one row per player per season.

Each row has the season's Premier League stats (goals, assists, minutes),
the player's age and position, and his Transfermarkt market value right
after that season ended (the number the model learns to predict).
"""
import pandas as pd

from config import DATASET_FILE, FEATURES, FIRST_SEASON, MIN_MINUTES, PREMIER_LEAGUE, RAW_DIR
from features import add_model_features

POSITION_GROUPS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfield": "MID", "Attack": "FWD"}
ROLE_SHORT = {
    "Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "LB", "Right-Back": "RB",
    "Defensive Midfield": "DM", "Central Midfield": "CM", "Attacking Midfield": "AM",
    "Left Midfield": "LM", "Right Midfield": "RM", "Left Winger": "LW", "Right Winger": "RW",
    "Second Striker": "SS", "Centre-Forward": "CF",
}


def season_of(dates: pd.Series) -> pd.Series:
    """A season is named by the year it starts: Aug 2024 - May 2025 is 2024."""
    return dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)


def load_season_stats() -> pd.DataFrame:
    apps = pd.read_csv(
        RAW_DIR / "appearances.csv.gz",
        usecols=["player_id", "player_club_id", "date", "competition_id", "goals", "assists", "minutes_played"],
        parse_dates=["date"],
    )
    apps = apps[apps["competition_id"] == PREMIER_LEAGUE].copy()
    apps["season"] = season_of(apps["date"])

    stats = (
        apps.groupby(["player_id", "season"])
        .agg(goals=("goals", "sum"), assists=("assists", "sum"),
             minutes=("minutes_played", "sum"), apps=("date", "count"))
        .reset_index()
    )

    # The club he played the most minutes for that season (handles January transfers).
    club_minutes = apps.groupby(["player_id", "season", "player_club_id"])["minutes_played"].sum().reset_index()
    main_club = club_minutes.sort_values("minutes_played").drop_duplicates(["player_id", "season"], keep="last")
    clubs = pd.read_csv(RAW_DIR / "clubs.csv.gz", usecols=["club_id", "name"]).rename(columns={"name": "club"})
    clubs["club"] = clubs["club"].str.replace(r"\s+A?FC$|^AFC\s+", "", regex=True)
    main_club = main_club.merge(clubs, left_on="player_club_id", right_on="club_id")
    stats = stats.merge(main_club[["player_id", "season", "club"]], on=["player_id", "season"], how="left")
    # Drop the season that is still being played, if the data includes one.
    # Premier League seasons end in May.
    latest = stats["season"].max()
    season_over = apps["date"].max().month in (5, 6)
    last_finished = latest if season_over else latest - 1
    return stats[stats["season"].between(FIRST_SEASON, last_finished)]


def attach_market_values(stats: pd.DataFrame) -> pd.DataFrame:
    """Latest market value on or before 1 August after each season (Transfermarkt's summer update)."""
    vals = pd.read_csv(
        RAW_DIR / "player_valuations.csv.gz",
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    ).rename(columns={"date": "value_date", "market_value_in_eur": "market_value"})

    stats = stats.copy()
    stats["as_of"] = pd.to_datetime((stats["season"] + 1).astype(str) + "-08-01")
    merged = pd.merge_asof(
        stats.sort_values("as_of"),
        vals.sort_values("value_date"),
        left_on="as_of", right_on="value_date", by="player_id", direction="backward",
    )
    # A valuation from before the season started says nothing about this season.
    fresh = merged["value_date"] >= merged["as_of"] - pd.DateOffset(months=10)
    return merged[fresh].drop(columns=["as_of"])


def attach_player_info(df: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_csv(
        RAW_DIR / "players.csv.gz",
        usecols=["player_id", "name", "date_of_birth", "position", "sub_position"],
        parse_dates=["date_of_birth"],
    )
    df = df.merge(players, on="player_id", how="inner")
    df["position"] = df["position"].map(POSITION_GROUPS)
    df["role"] = df["sub_position"].map(ROLE_SHORT).fillna(df["position"])

    # Age on 1 January of the season, e.g. 1 Jan 2025 for 2024/25.
    mid_season = pd.to_datetime((df["season"] + 1).astype(str) + "-01-01")
    df["age"] = ((mid_season - df["date_of_birth"]).dt.days / 365.25).astype(int)
    return df.dropna(subset=["position", "age"])


def main() -> None:
    stats = load_season_stats()
    df = attach_player_info(attach_market_values(stats))
    df = add_model_features(df[df["minutes"] >= MIN_MINUTES])
    df["season_label"] = df["season"].astype(str) + "/" + (df["season"] + 1).astype(str).str[-2:]
    df = df.sort_values(["season", "market_value"], ascending=[True, False])

    info = ["player_id", "name", "season_label", "club", "position", "role", "age", "apps"]
    columns = info + [c for c in FEATURES if c not in info] + ["market_value"]
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(DATASET_FILE, index=False)

    print(f"Saved {len(df)} player-seasons to {DATASET_FILE.name}")
    print(df.groupby("season_label").size().to_string())


if __name__ == "__main__":
    main()
