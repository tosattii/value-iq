"""Step 3: fit a linear regression with scikit-learn and plot predicted vs actual.

The model trains on every season except the latest one, then predicts the
latest season it has never seen. That's an honest test: it's exactly how
you'd use it to call a player overpriced or underpriced.
"""
import json

import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from config import DATASET_FILE, FEATURES, MODEL_FILE, OUTPUT_DIR, PREDICTIONS_FILE, TARGET
from model_utils import predict


def euros(x: float, _=None) -> str:
    return f"€{x / 1e6:.0f}M"


def plot_predicted_vs_actual(test: pd.DataFrame, r2: float, mae: float, season: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(test[TARGET], test["predicted"], s=22, alpha=0.55, color="#2f5d9e", edgecolor="none")
    top = max(test[TARGET].max(), test["predicted"].max()) * 1.05
    ax.plot([0, top], [0, top], color="#c0392b", lw=1.5, label="perfect prediction")

    # Label the biggest misses in each direction.
    test = test.assign(gap=test["predicted"] - test[TARGET])
    for _, row in pd.concat([test.nlargest(4, "gap"), test.nsmallest(4, "gap")]).iterrows():
        ax.annotate(row["name"], (row[TARGET], row["predicted"]), fontsize=8,
                    xytext=(5, 3), textcoords="offset points", color="#333")

    ax.set(xlim=(0, top), ylim=(0, top),
           xlabel="Actual market value (Transfermarkt)", ylabel="Predicted market value",
           title=f"Premier League {season}: predicted vs actual ({len(test)} players)")
    ax.xaxis.set_major_formatter(FuncFormatter(euros))
    ax.yaxis.set_major_formatter(FuncFormatter(euros))
    ax.text(0.03, 0.95, f"R² = {r2:.2f}\nMean error = {euros(mae)}", transform=ax.transAxes,
            va="top", fontsize=11, bbox=dict(facecolor="white", edgecolor="#ccc"))
    ax.text(0.97, 0.03, "above the line: model says underpriced\nbelow the line: model says overpriced",
            transform=ax.transAxes, ha="right", fontsize=9, color="#666")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATASET_FILE)
    test_season = df["season"].max()
    train, test = df[df["season"] < test_season], df[df["season"] == test_season].copy()

    reg = LinearRegression().fit(train[FEATURES], train[TARGET])
    model = {
        "intercept": float(reg.intercept_),
        "coefficients": dict(zip(FEATURES, map(float, reg.coef_))),
        # Means of the season we explain, so contributions read as "vs an average player this season".
        "feature_means": {f: float(test[f].mean()) for f in FEATURES},
        "train_seasons": sorted(train["season_label"].unique().tolist()),
        "test_season": test["season_label"].iloc[0],
    }

    test["predicted"] = predict(test, model)
    r2 = r2_score(test[TARGET], test["predicted"])
    mae = mean_absolute_error(test[TARGET], test["predicted"])
    model["metrics"] = {"r2": round(r2, 3), "mae": round(mae), "n_train": len(train), "n_test": len(test)}

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps(model, indent=2))
    test.to_csv(PREDICTIONS_FILE, index=False)
    plot_predicted_vs_actual(test, r2, mae, model["test_season"])

    print(f"Trained on {len(train)} player-seasons ({model['train_seasons'][0]} to {model['train_seasons'][-1]})")
    print(f"Tested on {len(test)} players from {model['test_season']}: R² = {r2:.2f}, mean error = {euros(mae)}")
    print("\nWhat each feature is worth:")
    for f, c in model["coefficients"].items():
        print(f"  {f:<18} {c:>+14,.0f} €")
    print(f"\nSaved {MODEL_FILE.name}, {PREDICTIONS_FILE.name} and predicted_vs_actual.png to output/")


if __name__ == "__main__":
    main()
