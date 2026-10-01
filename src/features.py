import numpy as np
import pandas as pd

from config import (
    LAG_WEEKS,
    ROLLING_WINDOWS,
    FORECAST_HORIZON_WEEKS,
    TARGET_SMOOTH_RADIUS,
    TREND_SLOPE_WINDOW,
    PROMO_WEEK_RANGES,
    WEATHER_MERGE_TOLERANCE_DAYS,
    PROCESSED_DIR,
)

EPS = 1e-3


def symmetric_pct_change(current, reference):
    """Bounded relative-change measure: (-2, 2) instead of the unbounded
    current/reference ratio, so it doesn't blow up near zero (needed once we
    mix H&M sales-volume scale with Trends' 0-100 interest scale)."""
    return (current - reference) / ((np.abs(current) + np.abs(reference)) / 2 + EPS)


def _week_in_range(week_of_year: pd.Series, week_range: tuple) -> pd.Series:
    start, end = week_range
    if start <= end:
        return (week_of_year >= start) & (week_of_year <= end)
    # wraps year-end, e.g. (51, 2) == weeks 51,52,53,1,2
    return (week_of_year >= start) | (week_of_year <= end)


def add_calendar_features(df: pd.DataFrame, hemisphere: str = "northern") -> pd.DataFrame:
    """month + is_summer/is_winter (hemisphere-aware) + promo-period flags.

    H&M training data visually confirms a real Northern-Hemisphere seasonal
    pattern (its retail calendar/stock cycle), so H&M feature building uses
    the default hemisphere="northern". The live Google Trends validation
    table is Australian (Southern Hemisphere) search interest, so it must be
    built with hemisphere="southern" explicitly — getting this wrong would
    silently invert is_summer/is_winter for one of the two tables.
    """
    if hemisphere not in ("northern", "southern"):
        raise ValueError(f"hemisphere must be 'northern' or 'southern', got {hemisphere!r}")

    df = df.copy()
    df["month"] = df["week_start"].dt.month
    week_of_year = df["week_start"].dt.isocalendar().week.astype(int)

    if hemisphere == "northern":
        summer_months = {6, 7, 8}
        winter_months = {12, 1, 2}
    else:
        summer_months = {12, 1, 2}
        winter_months = {6, 7, 8}

    df["is_summer"] = df["month"].isin(summer_months).astype(int)
    df["is_winter"] = df["month"].isin(winter_months).astype(int)

    # Promo/calendar-event flags — hemisphere-independent (these are global
    # calendar dates, e.g. Black Friday, not seasonal), low-cardinality by
    # design (raw week_of_year itself was ablation-verified to overfit due to
    # its high cardinality and was dropped as a direct feature).
    for flag_name, week_range in PROMO_WEEK_RANGES.items():
        df[flag_name] = _week_in_range(week_of_year, week_range).astype(int)

    return df


def add_relative_lag_and_rolling_features(df: pd.DataFrame, value_col: str = "unit_sales") -> pd.DataFrame:
    df = df.sort_values(["garment_group_name", "week_start"]).copy()
    for lag in LAG_WEEKS:
        lag_col = f"ratio_lag_{lag}"
        df[lag_col] = df.groupby("garment_group_name")[value_col].transform(
            lambda s, lag=lag: symmetric_pct_change(s, s.shift(lag))
        )
    for window in ROLLING_WINDOWS:
        roll_col = f"ratio_rolling_{window}"
        rolling_mean = df.groupby("garment_group_name")[value_col].transform(
            lambda s, window=window: s.shift(1).rolling(window).mean()
        )
        df[roll_col] = symmetric_pct_change(df[value_col], rolling_mean)
    return df


def add_trend_slope_feature(df: pd.DataFrame, value_col: str = "unit_sales") -> pd.DataFrame:
    """Momentum feature: normalized linear-fit slope over the last
    TREND_SLOPE_WINDOW weeks. Looped per category (not groupby().apply()) —
    pandas 3.0 can silently drop the grouping column under .apply()."""
    df = df.sort_values(["garment_group_name", "week_start"]).copy()
    slopes = pd.Series(index=df.index, dtype=float)
    x = np.arange(TREND_SLOPE_WINDOW)
    for category, group in df.groupby("garment_group_name"):
        values = group[value_col].values
        cat_slopes = [np.nan] * len(values)
        for i in range(TREND_SLOPE_WINDOW - 1, len(values)):
            window_vals = values[i - TREND_SLOPE_WINDOW + 1: i + 1]
            slope = np.polyfit(x, window_vals, 1)[0]
            denom = np.abs(window_vals).mean() + EPS
            cat_slopes[i] = slope / denom
        slopes.loc[group.index] = cat_slopes
    df["trend_slope"] = slopes
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge weekly weather onto the feature table by nearest week_start
    (tolerant merge_asof, within WEATHER_MERGE_TOLERANCE_DAYS).

    Adds: avg_temp, temp_anomaly (deviation from smoothed day-of-year
    climatology — a genuinely causal, non-collinear-with-calendar signal),
    plus precipitation, precip_anomaly (same idea for rainfall — added to
    help the Outdoor category, where footfall/rainwear demand plausibly
    responds directly to rain).
    """
    weather_path = PROCESSED_DIR / "weather_weekly.csv"
    if not weather_path.exists():
        raise FileNotFoundError(
            f"{weather_path} not found. Run `python3 pull_weather.py` first to build it."
        )
    weather = pd.read_csv(weather_path, parse_dates=["week_start"])

    df = df.sort_values("week_start").copy()
    weather = weather.sort_values("week_start")

    merged = pd.merge_asof(
        df,
        weather,
        on="week_start",
        direction="nearest",
        tolerance=pd.Timedelta(days=WEATHER_MERGE_TOLERANCE_DAYS),
    )
    return merged


def add_target(df: pd.DataFrame, value_col: str = "unit_sales") -> pd.DataFrame:
    """Smoothed target: symmetric % change from current value to the average
    of the FORECAST_HORIZON_WEEKS +/- TARGET_SMOOTH_RADIUS weeks-ahead
    window, to reduce single-week noise in what we're trying to predict."""
    df = df.sort_values(["garment_group_name", "week_start"]).copy()

    def _future_smoothed(s):
        window_vals = [s.shift(-(FORECAST_HORIZON_WEEKS + off)) for off in range(-TARGET_SMOOTH_RADIUS, TARGET_SMOOTH_RADIUS + 1)]
        return pd.concat(window_vals, axis=1).mean(axis=1)

    future_smoothed = df.groupby("garment_group_name")[value_col].transform(_future_smoothed)
    df["target"] = symmetric_pct_change(future_smoothed, df[value_col])
    return df


def build_feature_table(weekly_df: pd.DataFrame, save: bool = True, hemisphere: str = "northern") -> pd.DataFrame:
    df = add_calendar_features(weekly_df, hemisphere=hemisphere)
    df = add_relative_lag_and_rolling_features(df)
    df = add_trend_slope_feature(df)
    df = add_weather_features(df)
    df = add_target(df)

    feature_cols = (
        [c for c in df.columns if c.startswith("ratio_")]
        + ["trend_slope", "avg_temp", "temp_anomaly", "precipitation", "precip_anomaly"]
    )
    before = len(df)
    df_clean = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    dropped = before - len(df_clean)

    if save:
        out_path = PROCESSED_DIR / "feature_table.csv"
        df_clean.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({df_clean.shape[0]} rows, {df_clean.shape[1]} cols)")
        print(df_clean.head())
        print(f"\nRows dropped to NaN (start/end of series): {dropped}")
        print("Rows per category after cleaning:")
        print(df_clean.groupby("garment_group_name").size())

    return df_clean


if __name__ == "__main__":
    weekly_path = PROCESSED_DIR / "weekly_category_sales.csv"
    weekly_df = pd.read_csv(weekly_path, parse_dates=["week_start"])
    build_feature_table(weekly_df, save=True)