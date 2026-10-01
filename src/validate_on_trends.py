"""
Validate the H&M-trained models against LIVE, CURRENT Google Trends search
interest for Australia — a genuine distribution-shift test: different signal
type (search interest vs. sales volume), different market/hemisphere,
different time period (2018-2020 training vs. 2025-2026 live data).
"""
import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from features import build_feature_table
from models import (
    FEATURE_COLS,
    get_category_thresholds,
    train_lightgbm_cv,
    train_ridge_cv,
    train_direction_classifier_cv,
    predict_direction_classifier,
    naive_persistence_forecast,
    direction_from_pct_change,
    directional_accuracy,
    evaluate,
    evaluate_classifier,
    ensemble_direction,
    USE_CATEGORY_THRESHOLDS,
    STABLE_THRESHOLD,
)


def build_trends_feature_table() -> pd.DataFrame:
    """Same feature pipeline as H&M training, but on live Trends data and
    with hemisphere="southern" explicitly passed (Australia), since H&M
    training defaults to hemisphere="northern"."""
    trends_path = PROCESSED_DIR / "live_trends.csv"
    trends_df = pd.read_csv(trends_path, parse_dates=["date"])
    trends_df = trends_df.rename(columns={"date": "week_start", "interest": "unit_sales"})
    trends_df = trends_df[["week_start", "garment_group_name", "unit_sales"]]
    # pull_trends.py saves tz-aware (UTC) timestamps, but weekly_category_sales.csv
    # and weather_weekly.csv are both tz-naive — merge_asof requires matching
    # dtypes, so normalize to tz-naive here (values are already week-aligned,
    # so dropping the UTC offset doesn't shift any date).
    if trends_df["week_start"].dt.tz is not None:
        trends_df["week_start"] = trends_df["week_start"].dt.tz_localize(None)

    df_clean = build_feature_table(trends_df, save=False, hemisphere="southern")
    return df_clean


def main():
    # Train final models on the FULL H&M feature table (all data, since this
    # is now genuinely out-of-sample evaluation on a different dataset).
    hm_feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv", parse_dates=["week_start"])

    lgb_model, lgb_cv_acc = train_lightgbm_cv(hm_feature_table)
    ridge_model, ridge_cv_acc = train_ridge_cv(hm_feature_table)
    clf_model, label_map, clf_cv_acc = train_direction_classifier_cv(hm_feature_table)

    trends_df = build_trends_feature_table()
    trends_thresholds = get_category_thresholds(trends_df)
    print(f"Live Trends feature rows (after dropping NaN edges): {len(trends_df)}")
    print("Rows per category:")
    print(trends_df.groupby("garment_group_name").size())
    if USE_CATEGORY_THRESHOLDS:
        print("\n--- On live Google Trends data (per-category stable thresholds) ---")
    else:
        print(f"\n--- On live Google Trends data (pooled stable threshold: +-{STABLE_THRESHOLD * 100:.2f}%) ---")

    naive_preds = naive_persistence_forecast(trends_df)
    evaluate("Naive Persistence (Trends)", trends_df["target"].values, naive_preds, threshold=trends_thresholds)
    print()

    lgb_preds = lgb_model.predict(trends_df[FEATURE_COLS])
    evaluate("H&M-trained LightGBM (Trends)", trends_df["target"].values, lgb_preds, threshold=trends_thresholds)
    print()

    ridge_preds = ridge_model.predict(trends_df[FEATURE_COLS])
    evaluate("H&M-trained Ridge (Trends)", trends_df["target"].values, ridge_preds, threshold=trends_thresholds)
    print()

    clf_preds_dir = predict_direction_classifier(clf_model, label_map, trends_df[FEATURE_COLS])
    true_dir = direction_from_pct_change(trends_df["target"].values, threshold=trends_thresholds)
    evaluate_classifier("H&M-trained Direction Classifier (Trends)", true_dir, clf_preds_dir)
    print()

    lgb_dir = direction_from_pct_change(lgb_preds, threshold=trends_thresholds)
    ridge_dir = direction_from_pct_change(ridge_preds, threshold=trends_thresholds)

    ensemble_dir = ensemble_direction(lgb_dir, ridge_dir, clf_preds_dir)
    evaluate_classifier("Ensemble (Trends)", true_dir, ensemble_dir)

    trends_df = trends_df.copy()
    trends_df["clf_pred_dir"] = clf_preds_dir
    trends_df["ensemble_pred_dir"] = ensemble_dir
    trends_df["true_dir"] = true_dir

    print("\nPer-category directional accuracy on live Trends data (Direction Classifier):")
    for category, group in trends_df.groupby("garment_group_name"):
        acc = (group["true_dir"] == group["clf_pred_dir"]).mean()
        print(f"  {category}: {acc * 100:.2f}%  (n={len(group)})")

    print("\nPer-category directional accuracy on live Trends data (Ensemble):")
    for category, group in trends_df.groupby("garment_group_name"):
        acc = (group["true_dir"] == group["ensemble_pred_dir"]).mean()
        print(f"  {category}: {acc * 100:.2f}%  (n={len(group)})")

    out_path = PROCESSED_DIR / "trends_validation_results.csv"
    trends_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()