"""Data service for loading and querying traffic/weather data."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataService:
    """Service for loading and querying historical traffic/weather data."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.data_path: Optional[Path] = None

        # Indexed data for fast lookup
        self._segment_data: Dict[int, pd.DataFrame] = {}
        self._all_segments: List[int] = []
        self._timestamps: List[datetime] = []

        # Feature columns
        self.feature_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []

        # Expected input dim (set by model_service)
        self.expected_input_dim: Optional[int] = None

    def load_data(self, data_path: Path) -> None:
        """Load traffic/weather data from CSV."""
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        # Parse datetime
        if "datetime_traffic" in self.df.columns:
            self.df["datetime_traffic"] = pd.to_datetime(self.df["datetime_traffic"])
        elif "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Sort by datetime
        self.df = self.df.sort_values(["segment_id", "datetime_traffic"])

        # Build index for fast lookup
        self._build_index()

        # Identify feature columns
        self._identify_features()

        print(f"Data loaded: {len(self.df)} rows, {len(self._all_segments)} segments")

    def _build_index(self) -> None:
        """Build index for fast segment lookup."""
        self._all_segments = self.df["segment_id"].unique().tolist()
        self._timestamps = sorted(self.df["datetime_traffic"].unique())

        # Group by segment for fast lookup
        for seg_id in self._all_segments:
            seg_df = self.df[self.df["segment_id"] == seg_id].copy()
            seg_df = seg_df.set_index("datetime_traffic").sort_index()
            self._segment_data[seg_id] = seg_df

    def _identify_features(self) -> None:
        """Identify numerical and categorical feature columns."""
        # Columns to exclude (non-features)
        exclude_cols = {
            "segment_id",
            "street_id",
            "street_name",  # string column
            "datetime_traffic",
            "timestamp",
            "los",
            "los_index",
        }

        # Only include numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [c for c in numeric_cols if c not in exclude_cols]

        # Numerical features
        numerical_candidates = [
            "length",
            "max_velocity",
            "temperature",
            "humidity",
            "rain",
            "wind_speed",
            "pressure",
            "visibility",
            "clouds",
        ]
        self.numerical_columns = [
            c for c in numerical_candidates if c in self.df.columns
        ]

        # Categorical features (usually one-hot encoded in CSV)
        self.categorical_columns = [
            c
            for c in self.feature_columns
            if c not in self.numerical_columns
            and c.startswith(("weekday_", "period_", "street_type_", "street_level_"))
        ]

        print(f"Features: {len(self.feature_columns)} columns")

    def get_sequence(
        self,
        segment_id: int,
        target_time: datetime,
        sequence_length: int = 8,
    ) -> Optional[np.ndarray]:
        """
        Get historical sequence for a segment ending at target_time.

        Args:
            segment_id: Segment ID
            target_time: Target datetime (will find closest available time)
            sequence_length: Number of timesteps to retrieve

        Returns:
            Feature array of shape [sequence_length, num_features] or None
        """
        if segment_id not in self._segment_data:
            return None

        seg_df = self._segment_data[segment_id]

        if len(seg_df) == 0:
            return None

        # Find closest timestamp to target_time
        closest_time = self._find_closest_time(seg_df.index, target_time)
        if closest_time is None:
            return None

        # Get data up to and including closest_time
        valid_times = seg_df.index[seg_df.index <= closest_time]

        if len(valid_times) == 0:
            # All data is after target_time, use earliest available
            valid_times = seg_df.index[:sequence_length]

        if len(valid_times) < sequence_length:
            # Not enough history, pad with earliest data
            available = seg_df.loc[valid_times].tail(len(valid_times))
            # Pad by repeating first row
            padding_needed = sequence_length - len(available)
            first_row = available.iloc[[0]]
            padding = pd.concat([first_row] * padding_needed, ignore_index=True)
            padding.index = [
                available.index[0] - timedelta(hours=i + 1)
                for i in range(padding_needed)
            ]
            sequence_df = pd.concat([padding, available])
        else:
            sequence_df = seg_df.loc[valid_times].tail(sequence_length)

        # Extract feature columns
        feature_cols = [c for c in self.feature_columns if c in sequence_df.columns]
        if not feature_cols:
            return None

        features = sequence_df[feature_cols].values.astype(np.float32)

        # Pad to expected_input_dim if needed
        if self.expected_input_dim and features.shape[1] < self.expected_input_dim:
            padding = np.zeros(
                (features.shape[0], self.expected_input_dim - features.shape[1]),
                dtype=np.float32,
            )
            features = np.concatenate([features, padding], axis=1)

        return features

    def _find_closest_time(
        self, timestamps: pd.DatetimeIndex, target_time: datetime
    ) -> Optional[datetime]:
        """
        Find the closest timestamp to target_time.

        Prefers timestamps <= target_time, but will use future timestamp
        if target_time is before all available data.
        """
        if len(timestamps) == 0:
            return None

        # Convert target_time to pandas Timestamp for comparison
        target_ts = pd.Timestamp(target_time)

        # Find timestamps before or equal to target
        before_mask = timestamps <= target_ts
        if before_mask.any():
            # Return the closest one before target
            return timestamps[before_mask].max()

        # All timestamps are after target, return the earliest
        return timestamps.min()

    def get_batch_sequences(
        self,
        segment_ids: List[int],
        target_time: datetime,
        sequence_length: int = 8,
        num_nodes: int = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get sequences for multiple segments.

        Args:
            segment_ids: List of segment IDs to retrieve
            target_time: Target datetime
            sequence_length: Sequence length
            num_nodes: Total number of nodes (for full batch)

        Returns:
            Tuple of (features array [num_nodes, seq_len, features], valid_segment_ids)
        """
        if num_nodes is None:
            num_nodes = len(self._all_segments)

        # Determine feature dimension from first available segment
        sample_seq = None
        for seg_id in segment_ids:
            sample_seq = self.get_sequence(seg_id, target_time, sequence_length)
            if sample_seq is not None:
                break

        if sample_seq is None:
            # Fallback: use feature columns count
            num_features = len(self.feature_columns) if self.feature_columns else 50
            return (
                np.zeros((num_nodes, sequence_length, num_features), dtype=np.float32),
                [],
            )

        num_features = sample_seq.shape[1]

        # Build full batch
        features = np.zeros(
            (num_nodes, sequence_length, num_features), dtype=np.float32
        )
        valid_ids = []

        # Create segment_id to index mapping
        seg_to_idx = {seg_id: idx for idx, seg_id in enumerate(self._all_segments)}

        for seg_id in segment_ids:
            if seg_id not in seg_to_idx:
                continue

            seq = self.get_sequence(seg_id, target_time, sequence_length)
            if seq is not None:
                idx = seg_to_idx[seg_id]
                if idx < num_nodes:
                    features[idx] = seq
                    valid_ids.append(seg_id)

        return features, valid_ids

    def get_all_segments_sequence(
        self,
        target_time: datetime,
        sequence_length: int = 8,
        segment_to_idx: Dict[int, int] = None,
    ) -> np.ndarray:
        """
        Get sequences for all segments at target_time.

        Args:
            target_time: Target datetime
            sequence_length: Sequence length
            segment_to_idx: Mapping from segment_id to node index

        Returns:
            Features array [num_nodes, seq_len, features]
        """
        if segment_to_idx is None:
            segment_to_idx = {
                seg_id: idx for idx, seg_id in enumerate(self._all_segments)
            }

        num_nodes = len(segment_to_idx)

        # Get sample for feature dimension
        sample_seq = None
        for seg_id in list(segment_to_idx.keys())[:10]:
            sample_seq = self.get_sequence(seg_id, target_time, sequence_length)
            if sample_seq is not None:
                break

        num_features = (
            sample_seq.shape[1] if sample_seq is not None else len(self.feature_columns)
        )

        # Build features for all nodes
        features = np.zeros(
            (num_nodes, sequence_length, num_features), dtype=np.float32
        )

        for seg_id, idx in segment_to_idx.items():
            seq = self.get_sequence(seg_id, target_time, sequence_length)
            if seq is not None and idx < num_nodes:
                features[idx] = seq

        return features

    def get_time_features(self, target_time: datetime) -> Dict[str, Any]:
        """Extract time-based features from datetime."""
        return {
            "weekday": target_time.weekday(),
            "hour": target_time.hour,
            "period": self._get_period(target_time.hour),
            "is_weekend": target_time.weekday() >= 5,
        }

    def _get_period(self, hour: int) -> str:
        """Get period string from hour."""
        if 0 <= hour < 6:
            return "period_00_06"
        elif 6 <= hour < 12:
            return "period_06_12"
        elif 12 <= hour < 18:
            return "period_12_18"
        else:
            return "period_18_00"

    def get_available_time_range(self) -> Tuple[datetime, datetime]:
        """Get available time range in data."""
        if not self._timestamps:
            return None, None
        return self._timestamps[0], self._timestamps[-1]

    @property
    def num_segments(self) -> int:
        return len(self._all_segments)

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)


# Global data service instance
data_service = DataService()
