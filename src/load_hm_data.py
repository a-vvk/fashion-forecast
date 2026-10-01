"""
Loads H&M articles + transactions, joins them, and aggregates to
weekly unit sales per garment_group_name category.

Expects (downloaded manually from Kaggle — see README):
  data/raw/articles.csv
  data/raw/transactions_train.csv

Kaggle competition page:
  https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
"""
import os

import pandas as pd
from tqdm import tqdm

from config import HM_ARTICLES_CSV, HM_TRANSACTIONS_CSV, TRAIN_CATEGORIES, PROCESSED_DIR


def load_articles() -> pd.DataFrame:
    """Load articles.csv, keep only columns we need."""
    cols = ["article_id", "garment_group_name"]
    df = pd.read_csv(HM_ARTICLES_CSV, usecols=cols)
    return df


def _estimate_total_chunks(csv_path, chunksize: int) -> int:
    """
    Estimates total chunk count from file size vs. a small sample's
    average row size, so tqdm can show a real ETA instead of just a
    row counter. Approximate by design — good enough for progress
    feedback, not used for anything else.
    """
    sample = pd.read_csv(csv_path, nrows=10_000)
    sample_csv_bytes = sample.memory_usage(deep=True).sum()  # rough proxy, fine for an estimate
    file_size = os.path.getsize(csv_path)
    # Better proxy: bytes actually on disk for those rows, measured via seek
    with open(csv_path, "rb") as f:
        f.readline()  # skip header
        start = f.tell()
        for _ in range(10_000):
            f.readline()
        sample_bytes = f.tell() - start
    avg_row_bytes = sample_bytes / 10_000
    est_total_rows = file_size / avg_row_bytes
    return max(1, int(est_total_rows // chunksize) + 1)


def load_transactions_in_chunks(chunksize: int = 2_000_000) -> pd.DataFrame:
    """
    transactions_train.csv is ~31M rows — read in chunks and keep only
    the columns needed for weekly aggregation, to avoid memory blowups.
    Shows a progress bar with an estimated ETA since this file is large
    enough (~3.5GB) that silent processing is hard to distinguish from
    a hang.
    """
    cols = ["t_dat", "article_id", "customer_id"]
    est_chunks = _estimate_total_chunks(HM_TRANSACTIONS_CSV, chunksize)

    chunks = []
    reader = pd.read_csv(HM_TRANSACTIONS_CSV, usecols=cols, chunksize=chunksize)
    for chunk in tqdm(reader, total=est_chunks, desc="Reading transactions", unit="chunk"):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def build_weekly_category_sales(save: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: [week_start, garment_group_name, unit_sales]
    restricted to the categories in config.TRAIN_CATEGORIES (the expanded
    19-category training set — not just the 5 Trends-mapped categories,
    since the model needs volume and variety to learn generalizable
    dynamics from, not category-specific memorization).

    unit_sales = count of transaction rows (each row = one unit sold)
    for that category in that week.
    """
    print("Loading articles.csv...")
    articles = load_articles()
    articles = articles[articles["garment_group_name"].isin(TRAIN_CATEGORIES)]
    print(f"  {len(articles)} articles match the {len(TRAIN_CATEGORIES)} training categories.")

    transactions = load_transactions_in_chunks()
    print(f"Loaded {len(transactions):,} transaction rows.")

    print("Joining transactions to selected-category articles...")
    merged = transactions.merge(articles, on="article_id", how="inner")
    print(f"  {len(merged):,} transaction rows matched (rest were other categories).")

    # Bucket into weeks (Monday-start weeks, matching typical retail reporting)
    print("Aggregating to weekly sales per category...")
    merged["week_start"] = merged["t_dat"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    weekly = (
        merged.groupby(["week_start", "garment_group_name"])
        .size()
        .reset_index(name="unit_sales")
        .sort_values(["garment_group_name", "week_start"])
        .reset_index(drop=True)
    )

    # Drop the first and last week PER CATEGORY: the raw data starts/ends
    # mid-week (2018-09-20 to 2020-09-22), so those buckets only contain a
    # partial week's transactions and read as artificially low — visible
    # as a sharp dip at both ends of every category's time series plot.
    before_trim = len(weekly)
    group_sizes = weekly.groupby("garment_group_name")["week_start"].transform("size")
    position_in_group = weekly.groupby("garment_group_name").cumcount()
    keep_mask = (position_in_group != 0) & (position_in_group != group_sizes - 1)
    weekly = weekly[keep_mask].reset_index(drop=True)
    print(f"Trimmed {before_trim - len(weekly)} partial-week rows (first/last week per category).")

    if save:
        out_path = PROCESSED_DIR / "weekly_category_sales.csv"
        weekly.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(weekly)} rows)")

    return weekly


if __name__ == "__main__":
    df = build_weekly_category_sales()
    print(df.head(10))
    print(f"\nCategories found: {df['garment_group_name'].unique()}")
    print(f"Date range: {df['week_start'].min()} to {df['week_start'].max()}")
    print(f"Weeks per category:\n{df.groupby('garment_group_name').size()}")