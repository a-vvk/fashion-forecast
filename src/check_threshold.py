"""
Computes STABLE_THRESHOLD (pooled, 33rd percentile of |target| across all
categories) and CATEGORY_STABLE_THRESHOLDS (the same logic computed
separately per category), so "stable" is data-driven rather than an
arbitrary fixed cutoff, and can reflect that different categories may have
genuinely different volatility.

Must be re-run — and the printed values pasted into config.py — after ANY
change to features, target smoothing, or the category set, since all of
those can shift the |target| distribution.
"""
import pandas as pd
import numpy as np

from config import PROCESSED_DIR

df = pd.read_csv(PROCESSED_DIR / "feature_table.csv")

pooled_threshold = np.percentile(df["target"].abs(), 33)
pooled_stable_frac = (df["target"].abs() < pooled_threshold).mean()
print(f"Pooled threshold (33rd percentile of |target|, all categories): {pooled_threshold:.4f}")
print(f"Pooled stable fraction: {pooled_stable_frac * 100:.2f}%")

print("\nPer-category thresholds — paste this dict into config.py as CATEGORY_STABLE_THRESHOLDS:")
print("CATEGORY_STABLE_THRESHOLDS = {")
category_thresholds = {}
for category, group in df.groupby("garment_group_name"):
    threshold = np.percentile(group["target"].abs(), 33)
    stable_frac = (group["target"].abs() < threshold).mean()
    category_thresholds[category] = threshold
    print(f'    "{category}": {threshold:.4f},  # stable fraction: {stable_frac * 100:.2f}%  (n={len(group)})')
print("}")