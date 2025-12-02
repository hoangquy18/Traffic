"""Sequential data loading for RNN-based traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from traffic_trainer.data.constants import LOS_LEVELS


@dataclass
class SequenceSplit:
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp
    test_cutoff: pd.Timestamp


def _infer_split_cutoffs(
    timestamps: Sequence[pd.Timestamp],
    train_ratio: float,
    val_ratio: float,
) -> SequenceSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    sorted_times = np.sort(np.unique(np.array(timestamps, dtype="datetime64[ns]")))
    if len(sorted_times) < 3:
        raise ValueError("Not enough timestamps to perform chronological split")

    total = len(sorted_times)
    train_idx = max(1, int(total * train_ratio))
    val_idx = max(train_idx + 1, int(total * (train_ratio + val_ratio)))

    return SequenceSplit(
        train_cutoff=pd.Timestamp(sorted_times[train_idx - 1]),
        val_cutoff=pd.Timestamp(sorted_times[val_idx - 1]),
        test_cutoff=pd.Timestamp(sorted_times[-1]),
    )


class TrafficWeatherDataset(Dataset):
    """Builds rolling sequences of traffic/weather features to predict LOS."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        time_ids: Optional[np.ndarray] = None,
        segment_ids: Optional[np.ndarray] = None,
        sample_times: Optional[np.ndarray] = None,
    ) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = torch.from_numpy(targets).long()
        self.time_ids = (
            torch.from_numpy(time_ids).long() if time_ids is not None else None
        )
        self.segment_ids = (
            torch.from_numpy(segment_ids).long() if segment_ids is not None else None
        )
        self.sample_times = (
            torch.from_numpy(sample_times).long() if sample_times is not None else None
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sample: Dict[str, torch.Tensor] = {"features": self.sequences[idx]}
        if self.time_ids is not None:
            sample["time_ids"] = self.time_ids[idx]
        if self.segment_ids is not None:
            sample["segment_ids"] = self.segment_ids[idx]
        if self.sample_times is not None:
            sample["target_time"] = self.sample_times[idx]
        return sample, self.targets[idx]


def load_dataset(
    csv_path: Path,
    sequence_length: int,
    feature_columns: Optional[Dict[str, Sequence[str]]] = None,
    prediction_horizons: Sequence[int] = (1,),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize: bool = True,
    resample_rule: Optional[str] = "1H",
    minimum_segment_length: Optional[int] = None,
    use_time_embedding: bool = False,
    use_segment_embedding: bool = False,
) -> Tuple[
    TrafficWeatherDataset,
    TrafficWeatherDataset,
    TrafficWeatherDataset,
    List[str],
    StandardScaler,
    Dict[str, Any],
]:
    """Load CSV and construct train/val/test datasets."""

    df = pd.read_csv(csv_path)
    if "LOS" not in df.columns:
        raise ValueError("CSV must contain LOS column for supervised training.")

    df["datetime_traffic"] = pd.to_datetime(
        df["datetime_traffic"], utc=True
    ).dt.tz_convert(None)
    df["LOS"] = df["LOS"].str.strip().str.upper()
    df = df[df["LOS"].isin(LOS_LEVELS.keys())].copy()
    df["los_idx"] = df["LOS"].map(LOS_LEVELS)

    if feature_columns is None:
        raise ValueError(
            "feature_columns must be provided explicitly via configuration."
        )

    if not prediction_horizons:
        raise ValueError("prediction_horizons must contain at least one horizon step.")
    if any(h <= 0 for h in prediction_horizons):
        raise ValueError("prediction_horizons must contain positive integers.")
    prediction_horizons = tuple(sorted({int(h) for h in prediction_horizons}))
    max_horizon = prediction_horizons[-1]

    numerical_columns = feature_columns.get("numerical", [])
    categorical_columns = feature_columns.get("categorical", [])

    missing_columns = [
        col for col in numerical_columns + categorical_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Columns not found in CSV: {missing_columns}")

    df_categorical = pd.get_dummies(
        df[categorical_columns], columns=categorical_columns, dummy_na=False
    )
    feature_df = pd.concat(
        [
            df[numerical_columns].reset_index(drop=True),
            df_categorical.reset_index(drop=True),
        ],
        axis=1,
    )

    feature_df = feature_df.ffill().bfill().fillna(0)

    scaler = StandardScaler()
    feature_array = feature_df.to_numpy(dtype=np.float64, copy=True)

    if normalize:
        scaler.fit(feature_array)
        scale = scaler.scale_.copy()
        zero_scale_mask = scale == 0
        if zero_scale_mask.any():
            scale[zero_scale_mask] = 1.0
            scaler.scale_[zero_scale_mask] = 1.0
            scaler.var_[zero_scale_mask] = 1.0
        features_scaled = (feature_array - scaler.mean_) / scale
    else:
        scaler.fit(feature_array)
        features_scaled = feature_array

    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    metadata_cols = {"segment_id", "datetime_traffic", "los_idx"}
    feature_df = feature_df.loc[:, ~feature_df.columns.isin(metadata_cols)]

    feature_df = pd.DataFrame(
        features_scaled,
        columns=feature_df.columns,
        index=df.index,
    )
    feature_names = feature_df.columns.tolist()

    processed_df = pd.concat(
        [
            df[["segment_id", "datetime_traffic", "los_idx"]].reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )

    split_cutoffs = _infer_split_cutoffs(
        timestamps=processed_df["datetime_traffic"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    sequences: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    targets: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    time_sequences: Optional[Dict[str, List[np.ndarray]]] = (
        {"train": [], "val": [], "test": []} if use_time_embedding else None
    )
    segment_sequences: Optional[Dict[str, List[np.ndarray]]] = (
        {"train": [], "val": [], "test": []} if use_segment_embedding else None
    )
    target_times: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    segment_encoder: Dict[int, int] = {}
    if use_segment_embedding:
        unique_segments = processed_df["segment_id"].unique()
        segment_encoder = {int(seg): idx for idx, seg in enumerate(unique_segments)}

    grouped = processed_df.sort_values(["segment_id", "datetime_traffic"]).groupby(
        "segment_id"
    )

    for _, group in grouped:
        group = group.sort_values("datetime_traffic")

        if resample_rule is not None:
            rule = resample_rule.lower()
            features_resampled = (
                group.set_index("datetime_traffic")[feature_names]
                .resample(rule)
                .interpolate(method="time")
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            los_resampled = (
                group.set_index("datetime_traffic")["los_idx"]
                .resample(rule)
                .ffill()
                .bfill()
            )
            group = (
                features_resampled.assign(
                    los_idx=los_resampled,
                    segment_id=group["segment_id"].iloc[0],
                )
                .dropna(subset=["los_idx"])
                .reset_index()
            )
            group["datetime_traffic"] = pd.to_datetime(group["datetime_traffic"])
            group["los_idx"] = group["los_idx"].round().astype(int)
        else:
            group = group.copy()

        if minimum_segment_length and len(group) < minimum_segment_length:
            continue

        if len(group) < sequence_length + max_horizon:
            continue

        feature_matrix = group[feature_names].to_numpy()
        los_array = group["los_idx"].to_numpy(dtype=np.int64)
        time_index = group["datetime_traffic"]
        hour_array = (
            group["datetime_traffic"].dt.hour.to_numpy(dtype=np.int64)
            if use_time_embedding
            else None
        )
        segment_idx = (
            segment_encoder[int(group["segment_id"].iloc[0])]
            if use_segment_embedding
            else None
        )

        for start_idx in range(0, len(group) - sequence_length - max_horizon + 1):
            end_idx = start_idx + sequence_length
            sequence = feature_matrix[start_idx:end_idx]
            targets_vector = [
                los_array[end_idx - 1 + horizon] for horizon in prediction_horizons
            ]
            seq_end_time = time_index.iloc[end_idx - 1]
            final_time = time_index.iloc[end_idx - 1 + max_horizon]

            cutoff_time = final_time
            if cutoff_time <= split_cutoffs.train_cutoff:
                split = "train"
            elif cutoff_time <= split_cutoffs.val_cutoff:
                split = "val"
            else:
                split = "test"

            sequences[split].append(sequence)
            targets[split].append(targets_vector)
            target_times[split].append(int(cutoff_time.value))
            if time_sequences is not None and hour_array is not None:
                time_sequences[split].append(hour_array[start_idx:end_idx])
            if segment_sequences is not None and segment_idx is not None:
                segment_sequences[split].append(
                    np.full(sequence_length, segment_idx, dtype=np.int64)
                )

    def to_dataset(split: str) -> TrafficWeatherDataset:
        if not sequences[split]:
            raise ValueError(f"No sequences generated for {split} split.")
        seq_array = np.stack(sequences[split])
        tgt_array = np.array(targets[split], dtype=np.int64)
        time_array = (
            np.stack(time_sequences[split])
            if time_sequences is not None and time_sequences[split]
            else None
        )
        segment_array = (
            np.stack(segment_sequences[split])
            if segment_sequences is not None and segment_sequences[split]
            else None
        )
        target_time_array = (
            np.array(target_times[split], dtype=np.int64)
            if target_times[split]
            else None
        )
        return TrafficWeatherDataset(
            seq_array,
            tgt_array,
            time_ids=time_array,
            segment_ids=segment_array,
            sample_times=target_time_array,
        )

    metadata: Dict[str, Any] = {}
    if use_segment_embedding:
        metadata["segment_encoder"] = segment_encoder
        metadata["segment_vocab_size"] = len(segment_encoder)

    return (
        to_dataset("train"),
        to_dataset("val"),
        to_dataset("test"),
        feature_names,
        scaler,
        metadata,
    )


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic weather dataset utility.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Absolute path to the labelled CSV file",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=6,
        help="Number of time steps (rows) per input sequence",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of timestamps used for training split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of timestamps used for validation split",
    )
    parser.add_argument(
        "--resample-rule",
        type=str,
        default="1H",
        help="Pandas resample rule, set to None to keep original frequency",
    )
    return parser


def cli() -> None:
    args = get_arg_parser().parse_args()
    load_dataset(
        csv_path=args.csv_path,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        resample_rule=args.resample_rule,
    )
    print("Dataset loaded successfully.")


if __name__ == "__main__":
    cli()


