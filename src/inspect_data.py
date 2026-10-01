"""
Quick visual inspection of weekly category sales — run this after
load_hm_data.py to sanity-check seasonality/noise per category before
committing to model design choices.
"""
import matplotlib.pyplot as plt
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR, CATEGORIES

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "weekly_category_sales.csv", parse_dates=["week_start"])

    fig, axes = plt.subplots(len(CATEGORIES), 1, figsize=(10, 2.5 * len(CATEGORIES)), sharex=True)
    for ax, cat in zip(axes, CATEGORIES):
        cat_df = df[df["garment_group_name"] == cat]
        ax.plot(cat_df["week_start"], cat_df["unit_sales"])
        ax.set_title(cat)
        ax.set_ylabel("units/week")
    plt.tight_layout()

    out_path = OUTPUTS_DIR / "category_time_series.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot: {out_path}")

    # Quick noise/variance summary — helps decide which categories will
    # forecast well vs. poorly (useful for the Discussion section)
    summary = df.groupby("garment_group_name")["unit_sales"].agg(["mean", "std"])
    summary["coefficient_of_variation"] = summary["std"] / summary["mean"]
    summary = summary.sort_values("coefficient_of_variation")
    print("\nVariability per category (lower CoV = more predictable):")
    print(summary)
