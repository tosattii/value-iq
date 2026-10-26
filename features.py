"""Feature engineering shared by the dataset builder, the CLI and the dashboard."""
import pandas as pd

from config import PEAK_AGE, POSITIONS


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Needs goals, assists, minutes, age, position and season columns."""
    df = df.copy()
    df["goals_per90"] = df["goals"] / df["minutes"] * 90
    df["years_before_peak"] = (PEAK_AGE - df["age"]).clip(lower=0)
    df["years_after_peak"] = (df["age"] - PEAK_AGE).clip(lower=0)
    for pos in POSITIONS[1:]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    return df
