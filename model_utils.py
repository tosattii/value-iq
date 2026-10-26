"""Load the saved model and explain its predictions.

The model is a straight line, so it is stored as plain numbers in
output/model.json: prediction = intercept + sum(coefficient * feature).
"""
import json

import numpy as np
import pandas as pd

from config import FEATURE_GROUPS, FEATURES, MIN_PREDICTION, MODEL_FILE


def load_model() -> dict:
    if not MODEL_FILE.exists():
        raise SystemExit("No model yet. Run `python train_model.py` first.")
    return json.loads(MODEL_FILE.read_text())


def predict(df: pd.DataFrame, model: dict) -> np.ndarray:
    coefs = np.array([model["coefficients"][f] for f in FEATURES])
    raw = model["intercept"] + df[FEATURES].to_numpy() @ coefs
    return np.maximum(raw, MIN_PREDICTION)


def baseline(model: dict) -> float:
    """The prediction for a player with average everything."""
    c, m = model["coefficients"], model["feature_means"]
    return model["intercept"] + sum(c[f] * m[f] for f in FEATURES)


def contributions(row: pd.Series, model: dict) -> dict:
    """How many euros each feature group adds to (or takes from) the average player.

    baseline + all contributions = the prediction (before the floor).
    """
    return {
        group: sum(model["coefficients"][f] * (row[f] - model["feature_means"][f]) for f in cols)
        for group, cols in FEATURE_GROUPS.items()
    }
