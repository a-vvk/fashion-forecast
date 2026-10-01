"""
Pulls live Google Trends interest-over-time data for each category's
mapped keywords, for use as the deployment-phase validation signal.

Uses download_google_trends_comparison (2-5 keywords per browser load)
instead of one request per keyword — cuts this pull from 15 requests
(3 keywords x 5 categories, one at a time) down to just 5 (one per
category, comparing its 3 keywords together in a single call).
Google's rate limit is roughly 8-10 fresh sessions/hour, so this
version should comfortably complete in one run without needing to
wait and resume. Bonus: comparison mode returns all keywords in a
category on one shared 0-100 scale (relative to whichever is
strongest), which preserves their real relative popularity — unlike
pulling them separately, where each gets independently inflated to
its own peak = 100.

Also passes cookies="disk", a trendspyg feature specifically aimed at
this rate limit: it reuses Google's session cookie across calls so
each request looks like a returning visitor rather than a fresh
session (Google's hard 429 page is specifically triggered by bursts
of NEW sessions).

Still resumable (per CATEGORY now, not per keyword) in case a burst
still gets rate-limited — re-running skips categories already pulled.

Requires: pip install trendspyg
Requires: Chrome installed locally.
"""
import pandas as pd

from trendspyg import download_google_trends_comparison
from trendspyg.exceptions import RateLimitError

from config import CATEGORY_KEYWORD_MAP, TRENDS_TIMEFRAME, TRENDS_GEO, PROCESSED_DIR

RAW_PROGRESS_PATH = PROCESSED_DIR / "live_trends_raw_by_keyword.csv"


def load_existing_progress() -> pd.DataFrame:
    if RAW_PROGRESS_PATH.exists():
        df = pd.read_csv(RAW_PROGRESS_PATH, parse_dates=["date"])
        done_categories = set(df["garment_group_name"].unique())
        print(f"Resuming: found {len(done_categories)} already-pulled categories.")
        return df
    return pd.DataFrame(columns=["garment_group_name", "keyword", "date", "interest", "is_partial"])


def average_and_save(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = (
        raw_df.groupby(["garment_group_name", "date"])
        .agg(interest=("interest", "mean"), is_partial=("is_partial", "any"))
        .reset_index()
    )
    out_path = PROCESSED_DIR / "live_trends.csv"
    df.to_csv(out_path, index=False)
    raw_df.to_csv(RAW_PROGRESS_PATH, index=False)
    print(f"Saved: {out_path} ({len(df)} rows, averaged across keywords)")
    print(f"Saved: {RAW_PROGRESS_PATH} ({len(raw_df)} rows, per-keyword detail)")
    return df


def pull_trends_for_all_categories() -> pd.DataFrame:
    raw_df = load_existing_progress()
    done_categories = set(raw_df["garment_group_name"].unique())
    new_rows = []

    remaining = [c for c in CATEGORY_KEYWORD_MAP if c not in done_categories]
    print(f"{len(done_categories)} categories already done, {len(remaining)} remaining this run.\n")

    for category in remaining:
        keywords = CATEGORY_KEYWORD_MAP[category]
        print(f"Pulling comparison for '{category}' -> keywords {keywords} ...")
        try:
            result_df = download_google_trends_comparison(
                keywords=keywords,
                geo=TRENDS_GEO,
                timeframe=TRENDS_TIMEFRAME,
                output_format="dataframe",
                include_geo=False,  # skip the region breakdown, we don't use it
                cookies="disk",     # reuse session cookie to avoid the new-session rate limit
            )
        except RateLimitError:
            print(
                "\nRate-limited by Google Trends despite batching + cookie reuse. "
                "Progress so far has been saved — wait ~30+ minutes, then just "
                "re-run this script to continue (already-pulled categories will be skipped)."
            )
            break

        keyword_cols = [c for c in result_df.columns if c not in ("date", "is_partial")]
        for _, row in result_df.iterrows():
            for kw in keyword_cols:
                new_rows.append(
                    {
                        "garment_group_name": category,
                        "keyword": kw,
                        "date": row["date"],
                        "interest": row[kw],
                        "is_partial": row.get("is_partial", False),
                    }
                )

        # Save progress after every successful category, not just at the end.
        updated_raw_df = pd.concat([raw_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_raw_df["date"] = pd.to_datetime(updated_raw_df["date"])
        updated_raw_df.to_csv(RAW_PROGRESS_PATH, index=False)

    final_raw_df = pd.concat([raw_df, pd.DataFrame(new_rows)], ignore_index=True)
    final_raw_df["date"] = pd.to_datetime(final_raw_df["date"])
    return average_and_save(final_raw_df)


if __name__ == "__main__":
    df = pull_trends_for_all_categories()

    all_categories = set(CATEGORY_KEYWORD_MAP.keys())
    progress_df = pd.read_csv(RAW_PROGRESS_PATH)
    done_categories = set(progress_df["garment_group_name"].unique())

    if done_categories >= all_categories:
        print("\nAll categories pulled successfully.")
        print(df.head(10))
        print(f"\nCategories pulled: {df['garment_group_name'].unique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        zero_series = df.groupby("garment_group_name")["interest"].sum()
        flat = zero_series[zero_series == 0]
        if len(flat) > 0:
            print(f"\nWARNING: these categories returned all zeros: {list(flat.index)}")
            print("Consider adjusting the keyword mapping in config.py for these categories.")
    else:
        missing = all_categories - done_categories
        print(f"\nStill missing {len(missing)} categor(y/ies): {missing}")
        print("Re-run this script after waiting out the rate limit to continue.")