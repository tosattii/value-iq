"""Step 5: build output/dashboard.html, an interactive page you open in any browser.

The page is self-contained: the predictions and the model's coefficients are
written straight into it, so it works offline and you can send it to anyone.
A copy goes to docs/index.html, which is what GitHub Pages publishes.
"""
import json

import pandas as pd

from config import (DOCS_DIR, FAIR_BAND, FEATURE_GROUPS, GOATCOUNTER_CODE, MIN_PREDICTION,
                    OUTPUT_DIR, PEAK_AGE, PREDICTIONS_FILE, ROOT)
from model_utils import baseline, contributions, load_model

TEMPLATE = ROOT / "dashboard_template.html"


def main() -> None:
    model = load_model()
    df = pd.read_csv(PREDICTIONS_FILE)

    players = []
    for _, row in df.iterrows():
        contrib = contributions(row, model)
        players.append({
            "id": int(row["player_id"]), "name": row["name"], "club": row["club"],
            "role": row["role"], "position": row["position"], "age": int(row["age"]),
            "apps": int(row["apps"]), "minutes": int(row["minutes"]),
            "goals": int(row["goals"]), "assists": int(row["assists"]),
            "market": float(row["market_value"]), "predicted": round(float(row["predicted"])),
            # Every player here is from the same season, so "Season" adds nothing.
            "contrib": {g: round(v) for g, v in contrib.items() if g != "Season"},
        })

    data = {
        "season": model["test_season"],
        "seasonYear": int(model["test_season"][:4]),
        "trainSeasons": model["train_seasons"],
        "metrics": model["metrics"],
        "intercept": model["intercept"],
        "coefficients": model["coefficients"],
        "baseline": round(baseline(model)),
        "peakAge": PEAK_AGE,
        "minPrediction": MIN_PREDICTION,
        "fairBand": FAIR_BAND,
        "groups": list(FEATURE_GROUPS),
        "analytics": bool(GOATCOUNTER_CODE),
        "players": players,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    )
    if GOATCOUNTER_CODE:
        html = html.replace(
            "<!--__ANALYTICS__-->",
            f'<script data-goatcounter="https://{GOATCOUNTER_CODE}.goatcounter.com/count" '
            'async src="//gc.zgo.at/count.js"></script>',
        )

    out = OUTPUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)} and docs/index.html with {len(players)} players. "
          f"Visitor stats: {'GoatCounter ' + GOATCOUNTER_CODE if GOATCOUNTER_CODE else 'off'}.")


if __name__ == "__main__":
    main()
