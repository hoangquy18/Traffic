"""Feature engineering utilities for traffic-weather dataset."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def create_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create speed-related derived features."""
    df = df.copy()

    # Speed ratio (current speed / max velocity) - key indicator of congestion
    df["speed_ratio"] = df["current_speed"] / (df["max_velocity"] + 1e-6)
    df["speed_ratio"] = df["speed_ratio"].clip(0, 1.5)  # Cap at 1.5x

    # Speed ratio using free flow speed
    df["speed_ratio_freeflow"] = df["current_speed"] / (df["free_flow_speed"] + 1e-6)
    df["speed_ratio_freeflow"] = df["speed_ratio_freeflow"].clip(0, 1.5)

    # Congestion indicator (1 if speed < 50% of max, 0 otherwise)
    df["is_congested"] = (df["speed_ratio"] < 0.5).astype(float)

    # Speed deficit (how much slower than max)
    df["speed_deficit"] = df["max_velocity"] - df["current_speed"]
    df["speed_deficit"] = df["speed_deficit"].clip(0, None)

    # Relative speed (current vs free flow)
    df["relative_speed"] = df["current_speed"] - df["free_flow_speed"]

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal/cyclical features from datetime."""
    df = df.copy()

    if "datetime_traffic" not in df.columns:
        return df

    df["datetime_traffic"] = pd.to_datetime(df["datetime_traffic"])

    # Extract time components
    df["hour"] = df["datetime_traffic"].dt.hour
    df["day_of_month"] = df["datetime_traffic"].dt.day
    df["month"] = df["datetime_traffic"].dt.month
    df["day_of_week"] = df["datetime_traffic"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_year"] = df["datetime_traffic"].dt.dayofyear

    # Cyclical encoding for hour (sin/cos)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclical encoding for day of week
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Day of month (useful for single-month data)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    # Note: Month and day-of-year features removed for single-month data
    # Uncomment if you have multi-month data:
    # df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    # df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    # df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Binary indicators
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["is_weekday"] = (df["day_of_week"] < 5).astype(float)

    # Time of day categories (morning, afternoon, evening, night)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(float)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(float)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(float)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(float)

    # Rush hour indicators
    df["is_rush_morning"] = ((df["hour"] >= 7) & (df["hour"] < 9)).astype(float)
    df["is_rush_evening"] = ((df["hour"] >= 17) & (df["hour"] < 19)).astype(float)
    # Use logical OR on boolean, then convert to float
    df["is_rush_hour"] = ((df["hour"] >= 6) & (df["hour"] <= 9)) | (
        (df["hour"] >= 16) & (df["hour"] <= 20)
    )
    df["is_rush_hour"] = df["is_rush_hour"].astype(float)

    # Day of month (useful for single-month patterns, e.g., beginning/end of month)
    df["is_first_week"] = (df["day_of_month"] <= 7).astype(float)
    df["is_last_week"] = (df["day_of_month"] > 24).astype(float)
    df["is_mid_month"] = ((df["day_of_month"] > 7) & (df["day_of_month"] <= 24)).astype(
        float
    )

    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived weather features."""
    df = df.copy()

    # Temperature-Humidity Index (THI) - comfort indicator
    # THI = 0.8 * T + (RH/100) * (T - 14.4) + 46.4
    df["temperature_humidity_index"] = (
        0.8 * df["temperature_2m"]
        + (df["relative_humidity_2m"] / 100) * (df["temperature_2m"] - 14.4)
        + 46.4
    )

    # Heat index approximation (when temp > 27°C)
    df["heat_index"] = np.where(
        df["temperature_2m"] > 27,
        -8.78469475556
        + 1.61139411 * df["temperature_2m"]
        + 2.33854883889 * df["relative_humidity_2m"]
        - 0.14611605 * df["temperature_2m"] * df["relative_humidity_2m"]
        - 0.012308094 * df["temperature_2m"] ** 2
        - 0.0164248277778 * df["relative_humidity_2m"] ** 2
        + 0.002211732 * df["temperature_2m"] ** 2 * df["relative_humidity_2m"]
        + 0.00072546 * df["temperature_2m"] * df["relative_humidity_2m"] ** 2
        - 0.000003582 * df["temperature_2m"] ** 2 * df["relative_humidity_2m"] ** 2,
        df["temperature_2m"],
    )

    # Wind chill (when temp < 10°C)
    df["wind_chill"] = np.where(
        df["temperature_2m"] < 10,
        13.12
        + 0.6215 * df["temperature_2m"]
        - 11.37 * (df["wind_speed_10m"] ** 0.16)
        + 0.3965 * df["temperature_2m"] * (df["wind_speed_10m"] ** 0.16),
        df["temperature_2m"],
    )

    # Dew point depression (temperature - dew point)
    df["dew_point_depression"] = df["temperature_2m"] - df["dew_point_2m"]

    # Pressure gradient (difference between MSL and surface)
    df["pressure_gradient"] = df["pressure_msl"] - df["surface_pressure"]

    # Weather severity indicators
    df["is_rainy"] = (df["rain"] > 0.1).astype(float)  # > 0.1mm
    df["is_heavy_rain"] = (df["rain"] > 5.0).astype(float)  # > 5mm
    df["is_windy"] = (df["wind_speed_10m"] > 10).astype(float)  # > 10 m/s
    df["is_very_windy"] = (df["wind_speed_10m"] > 15).astype(float)  # > 15 m/s

    # Cloud cover categories
    df["is_cloudy"] = (df["cloud_cover"] > 50).astype(float)
    df["is_overcast"] = (df["cloud_cover"] > 80).astype(float)

    # Wind direction components (for easier learning)
    df["wind_u"] = df["wind_speed_10m"] * np.sin(np.radians(df["wind_direction_10m"]))
    df["wind_v"] = df["wind_speed_10m"] * np.cos(np.radians(df["wind_direction_10m"]))

    # Cyclical encoding for wind direction
    df["wind_direction_sin"] = np.sin(np.radians(df["wind_direction_10m"]))
    df["wind_direction_cos"] = np.cos(np.radians(df["wind_direction_10m"]))

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between different feature groups."""
    df = df.copy()

    # Weather-traffic interactions
    df["rain_speed_interaction"] = df["rain"] * df["current_speed"]
    df["wind_speed_interaction"] = df["wind_speed_10m"] * df["current_speed"]
    df["temp_speed_interaction"] = df["temperature_2m"] * df["current_speed"]

    # Weather-street interactions
    df["rain_length_interaction"] = df["rain"] * df["length"]
    df["wind_length_interaction"] = df["wind_speed_10m"] * df["length"]

    # Temperature-humidity interaction
    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]

    # Pressure-temperature interaction
    df["pressure_temp_interaction"] = df["pressure_msl"] * df["temperature_2m"]

    return df


def create_rolling_features(
    df: pd.DataFrame, group_col: str = "segment_id"
) -> pd.DataFrame:
    """Create rolling window statistics per segment."""
    df = df.copy()

    # Sort by segment and time
    df = df.sort_values([group_col, "datetime_traffic"]).reset_index(drop=True)

    # Features to compute rolling stats for
    rolling_cols = [
        "current_speed",
        "speed_ratio",
        "temperature_2m",
        "rain",
        "wind_speed_10m",
    ]

    # Only compute for columns that exist
    rolling_cols = [col for col in rolling_cols if col in df.columns]

    for col in rolling_cols:
        # Rolling mean (3-hour window)
        df[f"{col}_rolling_mean_3h"] = (
            df.groupby(group_col)[col]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Rolling std (3-hour window)
        df[f"{col}_rolling_std_3h"] = (
            df.groupby(group_col)[col]
            .rolling(window=3, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(0, drop=True)
        )

        # Lag features (previous hour)
        df[f"{col}_lag_1h"] = (
            df.groupby(group_col)[col]
            .shift(1)
            .bfill()  # Use bfill() instead of deprecated fillna(method="bfill")
            .reset_index(0, drop=True)
        )

        # Change from previous hour
        df[f"{col}_diff_1h"] = df[col] - df[f"{col}_lag_1h"]

    return df


def engineer_features(
    input_path: Path,
    output_path: Optional[Path] = None,
    add_rolling: bool = True,
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Args:
        input_path: Path to input CSV
        output_path: Path to save output CSV (if None, returns DataFrame only)
        add_rolling: Whether to add rolling features (slower but more informative)

    Returns:
        DataFrame with engineered features
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Creating speed features...")
    df = create_speed_features(df)

    print("Creating temporal features...")
    df = create_temporal_features(df)

    print("Creating weather features...")
    df = create_weather_features(df)

    print("Creating interaction features...")
    df = create_interaction_features(df)

    if add_rolling:
        print("Creating rolling features (this may take a while)...")
        df = create_rolling_features(df)

    print(f"Final shape: {len(df)} rows, {len(df.columns)} columns")

    if output_path:
        print(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print("Done!")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer features for traffic-weather dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save output CSV with engineered features",
    )
    parser.add_argument(
        "--no-rolling",
        action="store_true",
        help="Skip rolling features (faster but less informative)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engineer_features(
        input_path=args.input_path,
        output_path=args.output_path,
        add_rolling=not args.no_rolling,
    )


if __name__ == "__main__":
    main()
