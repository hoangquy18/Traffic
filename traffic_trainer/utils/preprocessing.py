import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


VELOCITY_THRESHOLDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "40_50": {
        "A": (35, np.inf),
        "B": (30, 35),
        "C": (25, 30),
        "D": (20, 25),
        "E": (15, 20),
        "F": (0, 15),
    },
    "60": {
        "A": (40, np.inf),
        "B": (35, 40),
        "C": (30, 35),
        "D": (25, 30),
        "E": (20, 25),
        "F": (0, 20),
    },
    "70": {
        "A": (45, np.inf),
        "B": (40, 45),
        "C": (35, 40),
        "D": (30, 35),
        "E": (25, 30),
        "F": (0, 25),
    },
}


def _select_threshold_group(max_velocity: float) -> Dict[str, Tuple[float, float]]:
    if max_velocity <= 55:
        return VELOCITY_THRESHOLDS["40_50"]
    if max_velocity <= 65:
        return VELOCITY_THRESHOLDS["60"]
    return VELOCITY_THRESHOLDS["70"]


def _classify_velocity(speed: float, thresholds: Dict[str, Tuple[float, float]]) -> str:
    for level, (lower, upper) in thresholds.items():
        if lower <= speed < upper or (level == "A" and speed >= lower):
            return level
    return "F"


def infer_los(row: pd.Series) -> str:
    cs = row.get("currentSpeed")
    max_velocity = row.get("maxVelocity")
    free_flow = row.get("freeFlowSpeed")

    try:
        cs_val = float(cs)
    except (TypeError, ValueError):
        cs_val = np.nan

    candidates: Iterable[float] = [
        row.get("freeFlowSpeed"),
        row.get("maxVelocity"),
    ]
    try:
        design_speed = max(float(x) for x in candidates if pd.notna(x) and x > 0)
    except ValueError:
        design_speed = np.nan

    if pd.isna(cs_val) or cs_val <= 0 or pd.isna(design_speed):
        return None

    thresholds = _select_threshold_group(design_speed)
    los = _classify_velocity(cs_val, thresholds)

    # fallback using congestion if available
    congestion = row.get("congestion")
    try:
        congestion_val = float(congestion)
    except (TypeError, ValueError):
        congestion_val = None

    if congestion_val is not None:
        if congestion_val >= 0.9:
            los = "F"
        elif congestion_val > 0.7 and los != "F":
            los = chr(min(ord(los) + 1, ord("F")))

    return los


def make_segment_id(row: pd.Series) -> int:
    key_parts = [
        str(row.get("streetName", "")),
        str(row.get("roadType", "")),
        f"{round(float(row.get('lat', 0.0)), 6)}",
        f"{round(float(row.get('lon', 0.0)), 6)}",
        f"{round(float(row.get('length', 0.0)), 2)}",
    ]
    key = "|".join(key_parts)
    digest = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:12], 16)


NUMERIC_COLUMNS = [
    "length",
    "maxVelocity",
    "currentSpeed",
    "freeFlowSpeed",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "rain",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "sunshine_duration",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "confidence",
    "congestion",
]


OUTPUT_COLUMNS: List[str] = [
    "_id",
    "segment_id",
    "street_id",
    "street_name",
    "street_type",
    "street_level",
    "length",
    "max_velocity",
    "current_speed",
    "free_flow_speed",
    "weekday",
    "period",
    "LOS",
    "datetime_traffic",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "rain",
    "apparent_temperature",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "sunshine_duration",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
]


SEGMENT_COORD_COLUMNS: List[str] = [
    "segment_id",
    "coordinates",
]


def process_file(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df["segment_id"] = df.apply(make_segment_id, axis=1)
    df["street_id"] = df["segment_id"]
    df["street_name"] = df["streetName"].fillna("unknown")
    df["street_type"] = df["roadType"].fillna("unknown")
    df["street_level"] = (
        df.get("frc", "FRC9").astype(str).str.extract(r"(\d+)").fillna(9).astype(int)
    )

    df["weekday"] = df["timestamp"].dt.weekday
    df["period"] = df["timestamp"].dt.strftime("period_%H_%M")
    df["datetime_traffic"] = df["timestamp"]

    df["max_velocity"] = df["maxVelocity"].fillna(df["freeFlowSpeed"])
    df["current_speed"] = df["currentSpeed"]
    df["free_flow_speed"] = df["freeFlowSpeed"]

    df["sunshine_duration"] = df.get("sunshine_duration", 0.0).fillna(0.0)
    df["rain"] = df.get("rain", 0.0).fillna(0.0)

    df["LOS"] = df.apply(infer_los, axis=1)
    df = df[df["LOS"].notna()].copy()

    # build per-segment coordinate mapping before restricting columns
    segment_cols = [c for c in SEGMENT_COORD_COLUMNS if c in df.columns]
    if segment_cols:
        segments = (
            df[segment_cols]
            .dropna(subset=["segment_id"])
            .drop_duplicates(subset=["segment_id"])
        )
    else:
        segments = pd.DataFrame(columns=SEGMENT_COORD_COLUMNS)

    df = df.sort_values("timestamp")
    df["_id"] = np.arange(1, len(df) + 1)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[OUTPUT_COLUMNS]
    return df, segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert traffic_weather_data CSV files into unified training format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw traffic weather CSV files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination path for the combined CSV.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Glob pattern to match files in input directory.",
    )
    parser.add_argument(
        "--segments-output-path",
        type=Path,
        required=False,
        help="Optional path for CSV mapping `segment_id` to coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError("No CSV files found with provided pattern.")

    feature_frames: List[pd.DataFrame] = []
    segment_frames: List[pd.DataFrame] = []

    for path in files:
        features, segments = process_file(path)
        feature_frames.append(features)
        segment_frames.append(segments)

    combined = pd.concat(feature_frames, ignore_index=True)
    combined.to_csv(args.output_path, index=False)
    print(f"Wrote {len(combined)} rows to {args.output_path}")

    if args.segments_output_path is not None:
        all_segments = (
            pd.concat(segment_frames, ignore_index=True)
            .drop_duplicates(subset=["segment_id"])
            .sort_values("segment_id")
        )
        all_segments.to_csv(args.segments_output_path, index=False)
        print(
            f"Wrote {len(all_segments)} unique segment coordinates to "
            f"{args.segments_output_path}"
        )


if __name__ == "__main__":
    main()
