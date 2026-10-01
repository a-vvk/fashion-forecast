"""
Pull historical daily weather for Sydney, AU from the free, keyless Open-Meteo
archive API, and build weekly features:
  - avg_temp:      mean daily temperature for the week (deg C)
  - temp_anomaly:  avg_temp minus a smoothed day-of-year climatology
                    (captures "unusually warm/cold for the time of year",
                    which is a genuinely causal driver of clothing demand,
                    independent of the month/season calendar features we
                    already have)
  - precipitation: total precipitation for the week (mm)
  - precip_anomaly: precipitation minus a smoothed day-of-year climatology
                    (captures "wetter/drier than usual" — rain plausibly
                    affects general shopping footfall and specific categories
                    such as Outdoor/rainwear demand)

Covers WEATHER_CLIMATOLOGY_START through 2 days before today (Open-Meteo's
archive endpoint needs a short reanalysis lag before a date is finalized).

Usage: python3 pull_weather.py
Output: data/processed/weather_weekly.csv
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from config import (
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE,
    WEATHER_CLIMATOLOGY_START,
    WEATHER_CLIMATOLOGY_SMOOTH_DAYS,
    PROCESSED_DIR,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily mean temperature and total precipitation for the
    configured Sydney lat/lon from Open-Meteo's free archive API. No API key
    needed."""
    params = {
        "latitude": WEATHER_LATITUDE,
        "longitude": WEATHER_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    print(f"Fetching Open-Meteo archive: {start_date} to {end_date} ...")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    daily = payload["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temp": daily["temperature_2m_mean"],
            "precip": daily["precipitation_sum"],
        }
    )
    print(f"Fetched {len(df)} days of data ({df['date'].min()} to {df['date'].max()}).")
    return df


def _smoothed_climatology(df: pd.DataFrame, value_col: str, smooth_days: int) -> pd.Series:
    """Mean value by day-of-year, circularly smoothed so Dec 31 / Jan 1 don't
    have an artificial discontinuity."""
    doy_mean = df.groupby(df["date"].dt.dayofyear)[value_col].mean()
    # pad circularly, smooth, then trim back to 1..366
    padded = pd.concat([doy_mean.tail(smooth_days), doy_mean, doy_mean.head(smooth_days)])
    smoothed = padded.rolling(window=smooth_days * 2 + 1, center=True, min_periods=1).mean()
    smoothed = smoothed.iloc[smooth_days: smooth_days + len(doy_mean)]
    smoothed.index = doy_mean.index
    return smoothed


def add_anomaly(df: pd.DataFrame, value_col: str, anomaly_col: str, smooth_days: int) -> pd.DataFrame:
    climatology = _smoothed_climatology(df, value_col, smooth_days)
    df = df.copy()
    df[anomaly_col] = df[value_col] - df["date"].dt.dayofyear.map(climatology)
    return df


def build_weekly_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather to weekly avg_temp / temp_anomaly /
    precipitation / precip_anomaly, keyed by week_start (W-SUN convention to
    match the rest of the pipeline's weekly aggregation)."""
    df = df.copy()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    weekly = (
        df.groupby("week_start")
        .agg(
            avg_temp=("temp", "mean"),
            temp_anomaly=("temp_anomaly", "mean"),
            precipitation=("precip", "sum"),
            precip_anomaly=("precip_anomaly", "mean"),
        )
        .reset_index()
    )
    return weekly


def main():
    end_date = (date.today() - timedelta(days=2)).isoformat()
    daily = fetch_daily_weather(WEATHER_CLIMATOLOGY_START, end_date)

    daily = add_anomaly(daily, "temp", "temp_anomaly", WEATHER_CLIMATOLOGY_SMOOTH_DAYS)
    daily = add_anomaly(daily, "precip", "precip_anomaly", WEATHER_CLIMATOLOGY_SMOOTH_DAYS)

    weekly = build_weekly_weather(daily)

    out_path = PROCESSED_DIR / "weather_weekly.csv"
    weekly.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(weekly)} weeks, {weekly['week_start'].min()} to {weekly['week_start'].max()})")
    print(weekly.head(5))
    print(weekly.tail(5))
    print("\nTemp anomaly stats:")
    print(weekly["temp_anomaly"].describe())
    print("\nPrecipitation stats:")
    print(weekly["precipitation"].describe())
    print("\nPrecip anomaly stats:")
    print(weekly["precip_anomaly"].describe())


if __name__ == "__main__":
    main()