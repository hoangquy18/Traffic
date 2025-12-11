# Feature Engineering Guide

This document describes the feature engineering process for the traffic-weather dataset.

## Overview

The feature engineering script (`engineer_features.py`) creates additional features from the base dataset to improve model performance. These features fall into several categories:

**Note: This configuration is optimized for single-month datasets.** Features that require multi-month data (month, day_of_year) have been removed or replaced with day-of-month features.

## Feature Categories

### 1. Speed-Derived Features

These features capture traffic flow characteristics:

- **speed_ratio**: `current_speed / max_velocity` - Normalized speed (0-1.5)
- **speed_ratio_freeflow**: `current_speed / free_flow_speed` - Speed relative to free flow
- **is_congested**: Binary indicator (1 if speed < 50% of max)
- **speed_deficit**: `max_velocity - current_speed` - How much slower than max
- **relative_speed**: `current_speed - free_flow_speed` - Difference from free flow

### 2. Temporal Features (Optimized for Single-Month Data)

#### Cyclical Encodings (sin/cos)
These encode cyclical patterns without creating artificial boundaries:

- **hour_sin/cos**: Hour of day (0-23) encoded cyclically - **Very Important**
- **day_of_week_sin/cos**: Day of week (0-6) encoded cyclically - **Very Important**
- **day_of_month_sin/cos**: Day of month (1-31) encoded cyclically - **Useful for single-month data**

#### Binary Indicators
- **is_weekend**: 1 if Saturday or Sunday
- **is_weekday**: 1 if Monday-Friday
- **is_morning**: 1 if 6:00-11:59
- **is_afternoon**: 1 if 12:00-17:59
- **is_evening**: 1 if 18:00-21:59
- **is_night**: 1 if 22:00-05:59
- **is_rush_morning**: 1 if 7:00-8:59
- **is_rush_evening**: 1 if 17:00-18:59
- **is_rush_hour**: 1 if rush hour (morning or evening)
- **is_first_week**: 1 if day 1-7 of month
- **is_last_week**: 1 if day 25-31 of month
- **is_mid_month**: 1 if day 8-24 of month

### 3. Weather-Derived Features

#### Comfort Indices
- **temperature_humidity_index**: THI = 0.8*T + (RH/100)*(T-14.4) + 46.4
- **heat_index**: Apparent temperature when temp > 27°C
- **wind_chill**: Apparent temperature when temp < 10°C

#### Weather Indicators
- **dew_point_depression**: `temperature_2m - dew_point_2m`
- **pressure_gradient**: `pressure_msl - surface_pressure`
- **is_rainy**: 1 if rain > 0.1mm
- **is_heavy_rain**: 1 if rain > 5.0mm
- **is_windy**: 1 if wind_speed > 10 m/s
- **is_very_windy**: 1 if wind_speed > 15 m/s
- **is_cloudy**: 1 if cloud_cover > 50%
- **is_overcast**: 1 if cloud_cover > 80%

#### Wind Components
- **wind_u**: East-west wind component (sin component)
- **wind_v**: North-south wind component (cos component)
- **wind_direction_sin/cos**: Cyclical encoding of wind direction

### 4. Interaction Features

These capture relationships between different feature groups:

- **rain_speed_interaction**: `rain * current_speed`
- **wind_speed_interaction**: `wind_speed_10m * current_speed`
- **temp_speed_interaction**: `temperature_2m * current_speed`
- **rain_length_interaction**: `rain * length`
- **wind_length_interaction**: `wind_speed_10m * length`
- **temp_humidity_interaction**: `temperature_2m * relative_humidity_2m`
- **pressure_temp_interaction**: `pressure_msl * temperature_2m`

### 5. Rolling Features (HIGHLY RECOMMENDED for Single-Month Data)

**These are especially important when you have limited temporal data!** They provide temporal context and help the model learn patterns.

For each of: `current_speed`, `speed_ratio`, `temperature_2m`, `rain`, `wind_speed_10m`:

- **{feature}_rolling_mean_3h**: 3-hour rolling mean
- **{feature}_rolling_std_3h**: 3-hour rolling standard deviation
- **{feature}_lag_1h**: Value from previous hour
- **{feature}_diff_1h**: Change from previous hour

## Usage

### Step 1: Run Feature Engineering

```bash
# RECOMMENDED: With rolling features (slower but much better for single-month data)
python3 -m traffic_trainer.utils.feature_engineering \
  --input-path traffic_weather_2025_converted.csv \
  --output-path traffic_weather_2025_engineered.csv

# Faster option (without rolling features - less recommended for single-month data)
python3 -m traffic_trainer.utils.feature_engineering \
  --input-path traffic_weather_2025_converted.csv \
  --output-path traffic_weather_2025_engineered.csv \
  --no-rolling
```

### Step 2: Update Config

Use the `config_engineered.yaml` file which includes all the new features optimized for single-month data. The config already includes rolling features (uncommented) since they're important for limited temporal data.

### Step 3: Train Model

```bash
python -m traffic_trainer.trainers.rnn_trainer \
  --config traffic_trainer/configs/config_engineered.yaml
```

## Feature Selection Tips for Single-Month Data

1. **Keep all temporal features** - Hour, day-of-week, and day-of-month patterns are crucial
2. **Use rolling features** - They're essential when you have limited time range
3. **Speed ratios are key** - These are direct congestion indicators
4. **Focus on hourly patterns** - With only 1 month, daily/hourly patterns are most important
5. **Weather interactions** - Can help capture how weather affects traffic in your specific month

## Notes for Single-Month Data

- **Month and day-of-year features removed** - Not useful with only 1 month
- **Day-of-month features added** - Can capture patterns like beginning/end of month
- **Rolling features highly recommended** - Provide temporal context when data is limited
- **Hour and day-of-week are most important** - These will show the strongest patterns
- **Consider weekday vs weekend patterns** - These should be very clear in your data

## If You Get Multi-Month Data Later

If you later get data spanning multiple months, you can:
1. Uncomment the month and day_of_year features in the code
2. Add them back to the config
3. Re-run feature engineering
