import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from traffic_trainer.data import LOS_LEVELS, load_dataset
from traffic_trainer.models import create_model


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a mapping.")
    return cfg


def aggregate_importance(
    importance: np.ndarray,
    prediction_horizons: Sequence[int],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for horizon_idx, horizon in enumerate(prediction_horizons):
        for feat, value in zip(feature_names, importance[horizon_idx]):
            rows.append(
                {
                    "feature": feat,
                    "horizon": horizon,
                    "importance_mean_abs": float(value),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute gradient-based feature importance for a trained traffic model."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint (best_model.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to analyse.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to store the importance table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    numerical_features = data_cfg.get("numerical_features", [])
    categorical_features = data_cfg.get("categorical_features", [])
    prediction_horizons = tuple(
        sorted({int(h) for h in data_cfg.get("prediction_horizons", [1])})
    )
    sequence_length = int(data_cfg.get("sequence_length", 8))
    resample_rule = data_cfg.get("resample_rule", "1H")

    time_embedding_dim = model_cfg.get("time_embedding_dim")
    segment_embedding_dim = model_cfg.get("segment_embedding_dim")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        feature_names,
        _,
        metadata,
    ) = load_dataset(
        csv_path=Path(config["paths"]["csv_path"]),
        sequence_length=sequence_length,
        feature_columns={
            "numerical": numerical_features,
            "categorical": categorical_features,
        },
        prediction_horizons=prediction_horizons,
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        resample_rule=resample_rule,
        use_time_embedding=bool(time_embedding_dim),
        use_segment_embedding=bool(segment_embedding_dim),
    )

    dataset = (
        train_dataset
        if args.split == "train"
        else val_dataset if args.split == "val" else test_dataset
    )

    segment_encoder = metadata.get("segment_encoder")
    segment_vocab_size = metadata.get("segment_vocab_size")

    device = torch.device(args.device)
    torch.backends.cudnn.enabled = False
    model = create_model(
        input_dim=len(feature_names),
        hidden_dim=model_cfg.get("hidden_dim", 192),
        num_layers=model_cfg.get("num_layers", 2),
        num_classes=len(LOS_LEVELS),
        rnn_type=model_cfg.get("rnn_type", "gru"),
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", True),
        num_horizons=len(prediction_horizons),
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = (
        checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    num_features = len(feature_names)
    importance_accumulator = np.zeros(
        (len(prediction_horizons), num_features), dtype=np.float64
    )
    total_sequences = 0

    for batch_inputs, _ in dataloader:
        features = batch_inputs["features"].to(device).float()
        features = features.clone().detach().requires_grad_(True)

        time_ids = (
            batch_inputs.get("time_ids").to(device).long()
            if batch_inputs.get("time_ids") is not None
            else None
        )
        segment_ids = (
            batch_inputs.get("segment_ids").to(device).long()
            if batch_inputs.get("segment_ids") is not None
            else None
        )

        logits = model(features, time_ids=time_ids, segment_ids=segment_ids)
        probs = torch.softmax(logits, dim=-1)
        top_classes = probs.argmax(dim=-1)

        for horizon_idx in range(len(prediction_horizons)):
            class_idx = top_classes[:, horizon_idx]
            selected = logits[:, horizon_idx, :].gather(
                1, class_idx.unsqueeze(1)
            ).sum()

            model.zero_grad(set_to_none=True)
            if features.grad is not None:
                features.grad.zero_()
            selected.backward(retain_graph=(horizon_idx + 1) < len(prediction_horizons))
            grad = features.grad.detach()

            contribution = grad.abs().sum(dim=1)  # sum over sequence length
            importance_accumulator[horizon_idx] += contribution.sum(dim=0).cpu().numpy()

        total_sequences += features.size(0)

    importance_accumulator /= max(total_sequences, 1)

    importance_df = aggregate_importance(
        importance=importance_accumulator,
        prediction_horizons=prediction_horizons,
        feature_names=feature_names,
    )
    importance_df.sort_values(
        by=["horizon", "importance_mean_abs"], ascending=[True, False], inplace=True
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(args.output_csv, index=False)

    print(f"Wrote feature importances to {args.output_csv}")
    for horizon in prediction_horizons:
        subset = importance_df[importance_df["horizon"] == horizon].head(10)
        print(f"\nTop features (horizon +{horizon}):")
        if subset.empty:
            print("  <none>")
        else:
            print(subset[["feature", "importance_mean_abs"]].to_string(index=False))


if __name__ == "__main__":
    main()