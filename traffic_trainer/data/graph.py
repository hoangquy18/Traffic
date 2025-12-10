"""
Graph/Transformer-based data loading for traffic prediction.

Supports two modes:
1. Graph mode (requires s_node_id, e_node_id): Builds explicit road network topology
2. Transformer mode (no topology needed): All segments in same timestep, model learns relationships

Both modes create "snapshot" datasets where each sample contains all segments at a given time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from traffic_trainer.data.constants import LOS_LEVELS


@dataclass
class RoadGraph:
    """Represents the road network topology."""

    num_nodes: int  # Number of road segments
    edge_index: torch.Tensor  # [2, num_edges] connectivity
    edge_attr: Optional[torch.Tensor]  # Optional edge features
    segment_to_idx: Dict[int, int]  # segment_id -> node index mapping
    idx_to_segment: Dict[int, int]  # node index -> segment_id mapping


def build_road_graph(
    df: pd.DataFrame,
    add_reverse_edges: bool = True,
    graph_mode: Literal["topology", "fully_connected", "none"] = "topology",
) -> RoadGraph:
    """
    Build a graph where each road segment is a node.

    Args:
        df: DataFrame with segment_id (and optionally s_node_id, e_node_id)
        add_reverse_edges: If True, add edges in both directions
        graph_mode:
            - "topology": Use s_node_id/e_node_id to build real road network
            - "fully_connected": All segments connected (for attention-based models)
            - "none": No edges, just segment mapping (for pure Transformer)

    Returns:
        RoadGraph containing topology information
    """
    # Get unique segments
    segment_ids = df["segment_id"].unique()
    segment_to_idx = {int(seg): idx for idx, seg in enumerate(segment_ids)}
    idx_to_segment = {idx: int(seg) for seg, idx in segment_to_idx.items()}
    num_nodes = len(segment_to_idx)

    edges_src: List[int] = []
    edges_dst: List[int] = []

    if graph_mode == "topology":
        # Check if topology columns exist
        if "s_node_id" not in df.columns or "e_node_id" not in df.columns:
            print("Warning: s_node_id/e_node_id not found, falling back to fully_connected mode")
            graph_mode = "fully_connected"

    if graph_mode == "topology":
        # Build graph from actual road network topology
        segment_info = (
            df[["segment_id", "s_node_id", "e_node_id"]]
            .drop_duplicates(subset=["segment_id"])
            .reset_index(drop=True)
        )

        start_node_to_segments: Dict[int, List[int]] = {}
        end_node_to_segments: Dict[int, List[int]] = {}

        for _, row in segment_info.iterrows():
            seg_id = int(row["segment_id"])
            s_node = int(row["s_node_id"])
            e_node = int(row["e_node_id"])

            if s_node not in start_node_to_segments:
                start_node_to_segments[s_node] = []
            start_node_to_segments[s_node].append(seg_id)

            if e_node not in end_node_to_segments:
                end_node_to_segments[e_node] = []
            end_node_to_segments[e_node].append(seg_id)

        # Create edges: segment A -> segment B if A's end_node == B's start_node
        for node_id, ending_segments in end_node_to_segments.items():
            if node_id in start_node_to_segments:
                starting_segments = start_node_to_segments[node_id]
                for seg_a in ending_segments:
                    for seg_b in starting_segments:
                        if seg_a != seg_b:
                            edges_src.append(segment_to_idx[seg_a])
                            edges_dst.append(segment_to_idx[seg_b])

        if add_reverse_edges:
            reverse_src = edges_dst.copy()
            reverse_dst = edges_src.copy()
            edges_src.extend(reverse_src)
            edges_dst.extend(reverse_dst)

    elif graph_mode == "fully_connected":
        # All segments connected to all others (for attention-based learning)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)

    # Add self-loops for all modes (except "none")
    if graph_mode != "none":
        for i in range(num_nodes):
            edges_src.append(i)
            edges_dst.append(i)

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        # Empty graph for Transformer mode
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return RoadGraph(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_attr=None,
        segment_to_idx=segment_to_idx,
        idx_to_segment=idx_to_segment,
    )


class GraphTrafficDataset(Dataset):
    """
    Dataset for graph/transformer-based traffic prediction.

    Each sample contains:
    - Node features: [num_nodes, sequence_length, num_features] - temporal features for each segment
    - Targets: [num_nodes, num_horizons] - LOS predictions for each segment
    - Mask: [num_nodes] - which nodes have valid data at this timestep
    - Segment IDs: [num_nodes] - for learnable segment embeddings
    - Time IDs: [num_nodes, seq_len] - hour-of-day for each timestep
    """

    def __init__(
        self,
        node_features: np.ndarray,  # [num_samples, num_nodes, seq_len, num_features]
        targets: np.ndarray,  # [num_samples, num_nodes, num_horizons]
        masks: np.ndarray,  # [num_samples, num_nodes] - valid nodes per sample
        timestamps: np.ndarray,  # [num_samples] - timestamp for each sample
        edge_index: torch.Tensor,  # [2, num_edges]
        segment_indices: Optional[np.ndarray] = None,  # [num_nodes] - segment index for embedding
        time_ids: Optional[np.ndarray] = None,  # [num_samples, seq_len] - hour-of-day for each timestep
        use_time_embedding: bool = False,
        use_segment_embedding: bool = False,
    ) -> None:
        self.node_features = torch.from_numpy(node_features).float()
        self.targets = torch.from_numpy(targets).long()
        self.masks = torch.from_numpy(masks).bool()
        self.timestamps = timestamps
        self.edge_index = edge_index
        self.use_time_embedding = use_time_embedding
        self.use_segment_embedding = use_segment_embedding
        
        self.segment_indices = (
            torch.from_numpy(segment_indices).long()
            if segment_indices is not None
            else torch.arange(node_features.shape[1])
        )
        
        # Time IDs: [num_samples, seq_len] -> need to broadcast to [num_nodes, seq_len]
        if time_ids is not None:
            self.time_ids = torch.from_numpy(time_ids).long()
        else:
            self.time_ids = None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {
            "node_features": self.node_features[idx],  # [num_nodes, seq_len, features]
            "targets": self.targets[idx],  # [num_nodes, num_horizons]
            "mask": self.masks[idx],  # [num_nodes]
            "edge_index": self.edge_index,  # [2, num_edges]
        }
        
        # Add segment embeddings if enabled
        if self.use_segment_embedding:
            result["segment_ids"] = self.segment_indices  # [num_nodes]
        else:
            result["segment_ids"] = self.segment_indices  # Always include for transformer
        
        # Add time embeddings if enabled
        if self.use_time_embedding and self.time_ids is not None:
            # Broadcast time_ids from [seq_len] to [num_nodes, seq_len]
            num_nodes = self.node_features.shape[1]
            result["time_ids"] = self.time_ids[idx].unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, seq_len]
        
        return result


def load_graph_dataset(
    csv_path: Path,
    sequence_length: int,
    feature_columns: Optional[Dict[str, Sequence[str]]] = None,
    prediction_horizons: Sequence[int] = (1,),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize: bool = True,
    resample_rule: Optional[str] = "1H",
    add_reverse_edges: bool = True,
    graph_mode: Literal["topology", "fully_connected", "none"] = "topology",
    use_time_embedding: bool = False,
    use_segment_embedding: bool = False,
) -> Tuple[
    GraphTrafficDataset,
    GraphTrafficDataset,
    GraphTrafficDataset,
    RoadGraph,
    List[str],
    StandardScaler,
    Dict[str, Any],
]:
    """
    Load CSV and construct graph/transformer-based train/val/test datasets.

    Unlike the sequential approach where each segment is processed independently,
    this creates snapshots of the entire road network at each timestep.

    Args:
        graph_mode:
            - "topology": Build graph from s_node_id/e_node_id (requires these columns)
            - "fully_connected": All segments connected (for GAT with learned attention)
            - "none": No graph edges, pure Transformer with self-attention

    Returns:
        train_dataset, val_dataset, test_dataset, road_graph, feature_names, scaler, metadata
    """
    df = pd.read_csv(csv_path)
    if "LOS" not in df.columns:
        raise ValueError("CSV must contain LOS column for supervised training.")

    # Parse datetime
    df["datetime_traffic"] = pd.to_datetime(
        df["datetime_traffic"], utc=True
    ).dt.tz_convert(None)
    df["LOS"] = df["LOS"].str.strip().str.upper()
    df = df[df["LOS"].isin(LOS_LEVELS.keys())].copy()
    df["los_idx"] = df["LOS"].map(LOS_LEVELS)

    if feature_columns is None:
        raise ValueError("feature_columns must be provided explicitly.")

    if not prediction_horizons:
        raise ValueError("prediction_horizons must contain at least one horizon.")
    prediction_horizons = tuple(sorted({int(h) for h in prediction_horizons}))
    max_horizon = prediction_horizons[-1]

    # Build road graph (or segment mapping for Transformer)
    road_graph = build_road_graph(df, add_reverse_edges=add_reverse_edges, graph_mode=graph_mode)
    print(f"Built road graph: {road_graph.num_nodes} segments, {road_graph.edge_index.shape[1]} edges (mode: {graph_mode})")

    numerical_columns = feature_columns.get("numerical", [])
    categorical_columns = feature_columns.get("categorical", [])

    missing_columns = [
        col for col in numerical_columns + categorical_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Columns not found in CSV: {missing_columns}")

    # One-hot encode categorical features
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
    feature_names = feature_df.columns.tolist()

    # Normalize features
    scaler = StandardScaler()
    feature_array = feature_df.to_numpy(dtype=np.float64)

    if normalize:
        scaler.fit(feature_array)
        scale = scaler.scale_.copy()
        zero_mask = scale == 0
        if zero_mask.any():
            scale[zero_mask] = 1.0
            scaler.scale_[zero_mask] = 1.0
            scaler.var_[zero_mask] = 1.0
        features_scaled = (feature_array - scaler.mean_) / scale
    else:
        scaler.fit(feature_array)
        features_scaled = feature_array

    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Add scaled features back to dataframe
    df = df.reset_index(drop=True)
    
    # Create feature columns DataFrame at once to avoid fragmentation
    feat_cols = [f"feat_{col}" for col in feature_names]
    features_df = pd.DataFrame(
        features_scaled,
        columns=feat_cols,
        index=df.index
    )
    
    # Concatenate instead of inserting columns one by one
    df = pd.concat([df, features_df], axis=1)

    # Resample to regular time intervals per segment
    if resample_rule:
        resampled_dfs = []
        for seg_id, group in df.groupby("segment_id"):
            group = group.set_index("datetime_traffic").sort_index()
            resampled = group[feat_cols + ["los_idx"]].resample(resample_rule).mean()
            resampled["los_idx"] = resampled["los_idx"].ffill().bfill()
            # Fill NaN in feature columns as well
            resampled[feat_cols] = resampled[feat_cols].ffill().bfill().fillna(0)
            resampled = resampled.dropna(subset=["los_idx"])
            resampled["los_idx"] = resampled["los_idx"].round().astype(int)
            
            # Add segment_id after reset_index to avoid fragmentation
            resampled_reset = resampled.reset_index()
            resampled_reset["segment_id"] = seg_id
            resampled_dfs.append(resampled_reset)
        df = pd.concat(resampled_dfs, ignore_index=True)
        df = df.rename(columns={"index": "datetime_traffic"})

    # Get all unique timestamps across all segments
    all_timestamps = sorted(df["datetime_traffic"].unique())
    print(f"Total timestamps: {len(all_timestamps)}")

    # Create time-aligned snapshots
    # For each timestamp t, we need features from [t-seq_len+1, t] and targets at [t+h for h in horizons]
    num_nodes = road_graph.num_nodes
    num_features = len(feature_names)

    # Build a lookup: (segment_id, timestamp) -> (features, los)
    segment_time_data: Dict[Tuple[int, pd.Timestamp], Tuple[np.ndarray, int]] = {}
    for _, row in df.iterrows():
        seg_id = int(row["segment_id"])
        ts = row["datetime_traffic"]
        feats = row[feat_cols].values.astype(np.float32)
        los = int(row["los_idx"])
        segment_time_data[(seg_id, ts)] = (feats, los)

    # Create samples
    samples_features: List[np.ndarray] = []
    samples_targets: List[np.ndarray] = []
    samples_masks: List[np.ndarray] = []
    samples_timestamps: List[pd.Timestamp] = []
    samples_time_ids: List[np.ndarray] = []  # For time embeddings

    # We need sequence_length consecutive timestamps + max_horizon future timestamps
    for t_idx in range(sequence_length - 1, len(all_timestamps) - max_horizon):
        current_ts = all_timestamps[t_idx]

        # Get sequence timestamps
        seq_timestamps = all_timestamps[t_idx - sequence_length + 1 : t_idx + 1]
        if len(seq_timestamps) != sequence_length:
            continue

        # Get target timestamps
        target_timestamps = [all_timestamps[t_idx + h] for h in prediction_horizons]

        # Build node features and targets for this snapshot
        node_features = np.zeros((num_nodes, sequence_length, num_features), dtype=np.float32)
        node_targets = np.full((num_nodes, len(prediction_horizons)), -1, dtype=np.int64)
        node_mask = np.zeros(num_nodes, dtype=bool)

        for seg_id, node_idx in road_graph.segment_to_idx.items():
            # Check if segment has data for all sequence timestamps
            has_sequence = all(
                (seg_id, ts) in segment_time_data for ts in seq_timestamps
            )
            # Check if segment has target data
            has_targets = all(
                (seg_id, ts) in segment_time_data for ts in target_timestamps
            )

            if has_sequence and has_targets:
                # Fill sequence features
                for s_idx, ts in enumerate(seq_timestamps):
                    feats, _ = segment_time_data[(seg_id, ts)]
                    node_features[node_idx, s_idx, :] = feats

                # Fill targets
                for h_idx, ts in enumerate(target_timestamps):
                    _, los = segment_time_data[(seg_id, ts)]
                    node_targets[node_idx, h_idx] = los

                node_mask[node_idx] = True

        # Only add sample if we have at least some valid nodes
        if node_mask.sum() > 0:
            samples_features.append(node_features)
            samples_targets.append(node_targets)
            samples_masks.append(node_mask)
            samples_timestamps.append(current_ts)
            
            # Extract hour-of-day for each timestamp in the sequence
            if use_time_embedding:
                time_ids = np.array([ts.hour for ts in seq_timestamps], dtype=np.int64)
                samples_time_ids.append(time_ids)

    if not samples_features:
        raise ValueError("No valid samples generated. Check data coverage.")

    # Convert to arrays
    all_features = np.stack(samples_features)
    all_targets = np.stack(samples_targets)
    all_masks = np.stack(samples_masks)
    all_timestamps_arr = np.array(samples_timestamps)
    all_time_ids = np.stack(samples_time_ids) if samples_time_ids else None

    print(f"Generated {len(all_features)} graph snapshots")
    print(f"Average valid nodes per snapshot: {all_masks.mean() * num_nodes:.1f}/{num_nodes}")

    # Chronological split
    n_samples = len(all_features)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Create segment indices for embeddings
    segment_indices = np.arange(num_nodes)

    train_dataset = GraphTrafficDataset(
        all_features[:train_end],
        all_targets[:train_end],
        all_masks[:train_end],
        all_timestamps_arr[:train_end],
        road_graph.edge_index,
        segment_indices,
        time_ids=all_time_ids[:train_end] if all_time_ids is not None else None,
        use_time_embedding=use_time_embedding,
        use_segment_embedding=use_segment_embedding,
    )
    val_dataset = GraphTrafficDataset(
        all_features[train_end:val_end],
        all_targets[train_end:val_end],
        all_masks[train_end:val_end],
        all_timestamps_arr[train_end:val_end],
        road_graph.edge_index,
        segment_indices,
        time_ids=all_time_ids[train_end:val_end] if all_time_ids is not None else None,
        use_time_embedding=use_time_embedding,
        use_segment_embedding=use_segment_embedding,
    )
    test_dataset = GraphTrafficDataset(
        all_features[val_end:],
        all_targets[val_end:],
        all_masks[val_end:],
        all_timestamps_arr[val_end:],
        road_graph.edge_index,
        segment_indices,
        time_ids=all_time_ids[val_end:] if all_time_ids is not None else None,
        use_time_embedding=use_time_embedding,
        use_segment_embedding=use_segment_embedding,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    metadata = {
        "num_nodes": road_graph.num_nodes,
        "num_edges": road_graph.edge_index.shape[1],
        "segment_to_idx": road_graph.segment_to_idx,
        "idx_to_segment": road_graph.idx_to_segment,
        "segment_vocab_size": num_nodes,  # For segment embeddings
    }

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        road_graph,
        feature_names,
        scaler,
        metadata,
    )


def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for graph batches."""
    batch_size = len(batch)
    segment_ids = batch[0]["segment_ids"]  # [num_nodes]
    
    result = {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "edge_index": batch[0]["edge_index"],  # Same graph structure for all
        "segment_ids": segment_ids.unsqueeze(0).expand(batch_size, -1),  # [batch, num_nodes]
    }
    
    # Add time_ids if present
    if "time_ids" in batch[0]:
        result["time_ids"] = torch.stack([b["time_ids"] for b in batch])
    
    return result


