from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

HM_ARTICLES_CSV = RAW_DIR / "articles.csv"
HM_TRANSACTIONS_CSV = RAW_DIR / "transactions_train.csv"

CATEGORY_KEYWORD_MAP = {
    "Jersey Basic": ["basic t-shirt", "plain t-shirt", "cotton t-shirt"],
    "Jersey Fancy": ["graphic t-shirt", "printed t-shirt", "band t-shirt"],
    "Dressed": ["dresses fashion", "womens dress", "midi dress"],
    "Trousers": ["trousers fashion", "womens trousers", "tailored trousers"],
    "Outdoor": ["winter jacket", "puffer jacket", "womens coat"],
}
CATEGORIES = list(CATEGORY_KEYWORD_MAP.keys())
# Ablation (5 -> 19 -> 11 -> 5 categories) showed the original 5 validation-matched
# categories generalize best to live Trends data, despite the 19-category set
# scoring better on in-domain H&M test data. Training-category relevance to the
# validation target matters more than raw training data volume. Reverted to 5.
TRAIN_CATEGORIES = CATEGORIES

HM_DATE_MIN = "2018-09-20"
HM_DATE_MAX = "2020-09-22"

FORECAST_HORIZON_WEEKS = 4
LAG_WEEKS = [1, 4, 8]
ROLLING_WINDOWS = [4, 8]
TREND_SLOPE_WINDOW = 4
TARGET_SMOOTH_RADIUS = 1

PROMO_WEEK_RANGES = {
    "is_near_black_friday": (46, 49),
    "is_near_midyear_sale": (25, 28),
    "is_near_holiday_season": (51, 2),  # wraps year-end
}

# Data-driven (33rd percentile of |target|), pooled across all categories.
# Recompute via check_threshold.py after any change to features, target
# smoothing, or category set. This is the threshold actually used by the
# final reported model (USE_CATEGORY_THRESHOLDS = False below).
STABLE_THRESHOLD = 0.1242

# Per-category version of the same 33rd-percentile logic, tested as a
# refinement: a single pooled threshold assumes every category is equally
# volatile, which isn't necessarily true. Tested and REJECTED — it nudged
# aggregate live-Trends accuracy up marginally (40.95% -> 42.38%) but came at
# the cost of Outdoor, which dropped further below chance (28.57% -> 21.43%
# ensemble), undoing part of the precipitation-feature gain. Likely
# overfitting: these thresholds are estimated on H&M (training-domain) data
# and applied unchanged to Trends (live-domain) data, assuming each
# category's relative volatility transfers across domains — an assumption
# that doesn't obviously hold. Kept here, and wired into get_category_
# thresholds() in models.py, for reproducibility, but left OFF by default.
USE_CATEGORY_THRESHOLDS = False

CATEGORY_STABLE_THRESHOLDS = {
    "Dressed": 0.1712,
    "Jersey Basic": 0.1158,
    "Jersey Fancy": 0.1248,
    "Outdoor": 0.1372,
    "Trousers": 0.0880,
}

TRAIN_FRAC = 70 / 104
VAL_FRAC = 17 / 104

CV_N_FOLDS = 3

# Diagnostic toggle: whether LightGBM trains with deterministic=True,
# force_row_wise=True. These flags remove multi-threaded run-to-run
# variance, but ALSO change how histogram sums are computed internally
# (row-wise vs column-wise), which can genuinely shift which splits get
# chosen — not just derandomize noise. Set to False temporarily to check
# whether results without the flags are (a) unstable run-to-run (confirms
# real non-determinism, and the flags are needed) or (b) stable at a
# different value than with the flags on (means the flags themselves are
# changing the trained model, not just removing noise). Run models.py and
# validate_on_trends.py twice with this False before drawing conclusions.
LIGHTGBM_DETERMINISTIC = True

TRENDS_TIMEFRAME = "today 12-m"
TRENDS_GEO = "AU"  # Australia — matches the user's own market

# Weather — Sydney, Australia (matches TRENDS_GEO; H&M training data itself
# follows a real Northern-Hemisphere seasonal pattern, so calendar features use
# hemisphere="northern" for H&M and hemisphere="southern" for the AU Trends table)
WEATHER_LATITUDE = -33.87
WEATHER_LONGITUDE = 151.21
WEATHER_CLIMATOLOGY_START = "2010-01-01"
WEATHER_CLIMATOLOGY_SMOOTH_DAYS = 7
WEATHER_MERGE_TOLERANCE_DAYS = 3