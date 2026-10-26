"""Shared settings for the Premier League player value predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"  # GitHub Pages serves docs/index.html
DATASET_FILE = PROCESSED_DIR / "player_seasons.csv"
MODEL_FILE = OUTPUT_DIR / "model.json"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

# Open Transfermarkt dataset (github.com/dcaribou/transfermarkt-datasets).
# Free, no API key needed. Updates paused since mid-2026: values end June 2026.
DATA_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
RAW_FILES = ["players.csv.gz", "appearances.csv.gz", "player_valuations.csv.gz", "clubs.csv.gz"]

PREMIER_LEAGUE = "GB1"   # Transfermarkt's competition id for the Premier League
FIRST_SEASON = 2019      # 2019/20 ... latest finished season
MIN_MINUTES = 900        # ignore players with fewer than 10 full games in a season
PEAK_AGE = 27            # age at which market value usually peaks
MIN_PREDICTION = 500_000 # a straight line can go below zero; market values can't
FAIR_BAND = 0.15         # within ±15% of the market value counts as fairly priced

# Visitor stats with GoatCounter (goatcounter.com): no cookies, no personal data.
# Put your site code here (e.g. "valueiq-yourname") and run make_dashboard.py again.
# Leave it empty to build the dashboard without analytics.
GOATCOUNTER_CODE = "dtosatti"

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# The five ideas from the video, turned into numbers the model can use.
FEATURE_GROUPS = {
    "Goals": ["goals", "goals_per90"],                       # season total & per 90
    "Assists": ["assists"],                                   # chances converted
    "Minutes": ["minutes"],                                   # how available he is
    "Age": ["years_before_peak", "years_after_peak"],         # years from peak
    "Position": ["pos_DEF", "pos_MID", "pos_FWD"],            # GK is the baseline
    "Season": ["season"],                                     # market inflation
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
TARGET = "market_value"
