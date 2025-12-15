"""Scan deep learning models over sequence_length and prediction horizon.

Models được scan:
- RNN
- GNN
- Transformer
- TCN
- Informer
- TimesNet
- GMAN (SOTA)

TimesNet++ KHÔNG được scan theo yêu cầu.

Với mỗi model:
- sequence_length từ 1 đến 6
- prediction_horizon từ 1 đến 5 (prediction_horizons = [h])

Kết quả được ghi incremental vào file CSV (mặc định: experiments/dl_scan.csv).
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from traffic_trainer.trainers import (
    GraphTrainer,
    GraphTrainingConfig,
    InformerTrainer,
    InformerTrainingConfig,
    RNNTrainer,
    SOTATrainer,
    SOTATrainingConfig,
    TCNTrainer,
    TCNTrainingConfig,
    TimesNetTrainer,
    TimesNetTrainingConfig,
    TransformerTrainer,
    TransformerTrainingConfig,
)
from traffic_trainer.trainers.rnn_trainer import load_config as load_rnn_config
from traffic_trainer.trainers.gnn_trainer import load_config as load_gnn_config
from traffic_trainer.trainers.transformer_trainer import (
    load_config as load_transformer_config,
)
from traffic_trainer.trainers.tcn_trainer import load_config as load_tcn_config
from traffic_trainer.trainers.informer_trainer import (
    load_config as load_informer_config,
)
from traffic_trainer.trainers.timesnet_trainer import (
    load_config as load_timesnet_config,
)
from traffic_trainer.trainers.gman_trainer import load_config as load_sota_config


def make_run_output_dir(
    base_dir: Path, model_name: str, seq_len: int, horizon: int
) -> Path:
    """Create a readable, unique output directory for a given combination."""
    sig = f"seq={seq_len}__h={horizon}"
    out_dir = base_dir / model_name / sig
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_row_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Append a single row to CSV, tạo file + header nếu chưa tồn tại."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)


def extract_last_val_metrics(history: Dict[str, Any]) -> Dict[str, float]:
    """Extract last epoch validation metrics from BaseTrainer history."""
    val_hist = history.get("val", [])
    if not val_hist:
        return {}
    last = val_hist[-1] or {}
    return {
        "val_loss": float(last.get("loss", 0.0)),
        "val_accuracy": float(last.get("accuracy", 0.0)),
        "val_f1_macro": float(last.get("f1_macro", 0.0)),
        "val_f1_weighted": float(last.get("f1_weighted", 0.0)),
    }


def scan_rnn(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # RNN không phải attention-based: ẩn tầng 64/128/256, layers 1/2/3
    hidden_dims = [128, 256]
    num_layers_list = [2, 3]
    emb_dims = [8, 16, 32]  # dùng cho cả time & segment embedding nếu có

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== RNN | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config = load_rnn_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = [horizon]
                        config.hidden_dim = hidden_dim
                        config.num_layers = num_layers
                        # embedding (nếu có)
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "rnn", seq_len, horizon
                        )

                        trainer = RNNTrainer(config)
                        results = trainer.train()

                        metrics = extract_last_val_metrics(results["history"])
                        row: Dict[str, Any] = {
                            "model": "RNN",
                            "primary_horizon": horizon,
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "embedding_dim": emb_dim,
                            "best_val_f1": float(results.get("best_val_f1", 0.0)),
                            "output_dir": str(config.output_dir),
                            **metrics,
                        }
                        append_row_csv(csv_path, fieldnames, row)


def scan_gnn(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # GNN (RNN+GNN) không hoàn toàn attention-based: cho phép 64/128/256
    hidden_dims = [128, 256]
    num_layers_list = [2, 3]
    emb_dims = [8, 16, 32]

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== GNN | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: GraphTrainingConfig = load_gnn_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = [horizon]
                        config.hidden_dim = hidden_dim
                        # dùng cùng một số layers cho RNN & GNN
                        config.num_rnn_layers = num_layers
                        config.num_gnn_layers = num_layers
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "gnn", seq_len, horizon
                        )

                        trainer = GraphTrainer(config)
                        results = trainer.train()

                        metrics = extract_last_val_metrics(results["history"])
                        row: Dict[str, Any] = {
                            "model": "GNN",
                            "primary_horizon": horizon,
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "embedding_dim": emb_dim,
                            "best_val_f1": float(results.get("best_val_f1", 0.0)),
                            "output_dir": str(config.output_dir),
                            **metrics,
                        }
                        append_row_csv(csv_path, fieldnames, row)


def scan_transformer(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # Attention-based: chỉ 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]  # số layer Transformer
    emb_dims = [8, 16, 32]

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== Transformer | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: TransformerTrainingConfig = load_transformer_config(
                            config_path
                        )
                        config.sequence_length = seq_len
                        config.prediction_horizons = [horizon]
                        config.hidden_dim = hidden_dim
                        config.num_transformer_layers = num_layers
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "transformer", seq_len, horizon
                        )

                        trainer = TransformerTrainer(config)
                        results = trainer.train()

                        metrics = extract_last_val_metrics(results["history"])
                        row: Dict[str, Any] = {
                            "model": "Transformer",
                            "primary_horizon": horizon,
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "embedding_dim": emb_dim,
                            "best_val_f1": float(results.get("best_val_f1", 0.0)),
                            "output_dir": str(config.output_dir),
                            **metrics,
                        }
                        append_row_csv(csv_path, fieldnames, row)


def scan_tcn(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # TCN có thể dùng attention, coi như attention-based: 64,128
    hidden_dims = [64, 128]
    # Không có tham số layer rõ ràng, dùng num_layers=0 như placeholder để log
    num_layers_list = [1, 2, 3]
    emb_dims = [8, 16, 32]

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== TCN | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: TCNTrainingConfig = load_tcn_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = [horizon]
                        config.hidden_dim = hidden_dim
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "tcn", seq_len, horizon
                        )

                        trainer = TCNTrainer(config)
                        results = trainer.train()

                        metrics = extract_last_val_metrics(results["history"])
                        row: Dict[str, Any] = {
                            "model": "TCN",
                            "primary_horizon": horizon,
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "embedding_dim": emb_dim,
                            "best_val_f1": float(results.get("best_val_f1", 0.0)),
                            "output_dir": str(config.output_dir),
                            **metrics,
                        }
                        append_row_csv(csv_path, fieldnames, row)


def scan_informer(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # Attention-based: 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]  # dùng cho e_layers/d_layers
    emb_dims = [8, 16, 32]

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== Informer | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: InformerTrainingConfig = load_informer_config(
                            config_path
                        )
                        # sequence_length cho dataset và seq_len/out_len cho model
                        config.sequence_length = seq_len
                        config.seq_len = seq_len
                        config.out_len = max(horizon, 1)
                        config.prediction_horizons = [horizon]
                        config.hidden_dim = hidden_dim
                        config.e_layers = num_layers
                        config.d_layers = max(1, num_layers - 1)
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "informer", seq_len, horizon
                        )

                        trainer = InformerTrainer(config)
                        results = trainer.train()

                        metrics = extract_last_val_metrics(results["history"])
                        row: Dict[str, Any] = {
                            "model": "Informer",
                            "primary_horizon": horizon,
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "embedding_dim": emb_dim,
                            "best_val_f1": float(results.get("best_val_f1", 0.0)),
                            "output_dir": str(config.output_dir),
                            **metrics,
                        }
                        append_row_csv(csv_path, fieldnames, row)


def scan_timesnet(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # Attention-based: 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]  # e_layers

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    print(
                        f"\n=== TimesNet | seq_len={seq_len}, horizon={horizon}, "
                        f"hidden_dim={hidden_dim}, layers={num_layers} ==="
                    )
                    config: TimesNetTrainingConfig = load_timesnet_config(config_path)
                    config.sequence_length = seq_len
                    config.prediction_horizons = [horizon]
                    # pred_len cho TimesNet nên >= horizon
                    config.pred_len = max(horizon, 1)
                    config.hidden_dim = hidden_dim
                    config.e_layers = num_layers
                    config.output_dir = make_run_output_dir(
                        output_root, "timesnet", seq_len, horizon
                    )

                    trainer = TimesNetTrainer(config)
                    results = trainer.train()

                    metrics = extract_last_val_metrics(results["history"])
                    row: Dict[str, Any] = {
                        "model": "TimesNet",
                        "primary_horizon": horizon,
                        "sequence_length": seq_len,
                        "prediction_horizon": horizon,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "embedding_dim": "",
                        "best_val_f1": float(results.get("best_val_f1", 0.0)),
                        "output_dir": str(config.output_dir),
                        **metrics,
                    }
                    append_row_csv(csv_path, fieldnames, row)


def scan_sota(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
) -> None:
    # GMAN là attention-based: 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]
    emb_dims = [8, 16, 32]

    for seq_len in seq_range:
        for horizon in horizon_range:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== GMAN++ | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: SOTATrainingConfig = load_sota_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = [horizon]
                        config.hidden_dim = hidden_dim
                        config.num_layers = num_layers
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "gman", seq_len, horizon
                        )

                        trainer = SOTATrainer(config)
                        results = trainer.train()

                        metrics = extract_last_val_metrics(results["history"])
                        row: Dict[str, Any] = {
                            "model": "GMAN++",
                            "primary_horizon": horizon,
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "embedding_dim": emb_dim,
                            "best_val_f1": float(results.get("best_val_f1", 0.0)),
                            "output_dir": str(config.output_dir),
                            **metrics,
                        }
                        append_row_csv(csv_path, fieldnames, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan deep learning models over sequence length and prediction horizon"
    )
    parser.add_argument(
        "--rnn-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "config.yaml",
        help="YAML config path cho RNN",
    )
    parser.add_argument(
        "--gnn-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "graph_config.yaml",
        help="YAML config path cho GNN",
    )
    parser.add_argument(
        "--transformer-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "transformer_config.yaml",
        help="YAML config path cho Transformer",
    )
    parser.add_argument(
        "--tcn-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "tcn_config.yaml",
        help="YAML config path cho TCN",
    )
    parser.add_argument(
        "--informer-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "informer_config.yaml",
        help="YAML config path cho Informer",
    )
    parser.add_argument(
        "--timesnet-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "timesnet_config.yaml",
        help="YAML config path cho TimesNet",
    )
    parser.add_argument(
        "--sota-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "sota_config.yaml",
        help="YAML config path cho GMAN/SOTA",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments") / "dl_scan",
        help="Thư mục gốc để lưu toàn bộ experiments của deep models",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "dl_scan.csv",
        help="File CSV lưu kết quả scan",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1,
        help="sequence_length nhỏ nhất (mặc định 1)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=6,
        help="sequence_length lớn nhất (mặc định 6, inclusive)",
    )
    parser.add_argument(
        "--min-horizon",
        type=int,
        default=1,
        help="prediction_horizon nhỏ nhất (mặc định 1)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=5,
        help="prediction_horizon lớn nhất (mặc định 5, inclusive)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seq_range = range(args.min_seq_len, args.max_seq_len + 1)
    horizon_range = range(args.min_horizon, args.max_horizon + 1)

    # Cột cố định cho CSV
    fieldnames: List[str] = [
        "model",
        "primary_horizon",
        "sequence_length",
        "prediction_horizon",
        "hidden_dim",
        "num_layers",
        "embedding_dim",
        "best_val_f1",
        "val_loss",
        "val_accuracy",
        "val_f1_macro",
        "val_f1_weighted",
        "output_dir",
    ]

    # RNN
    try:
        scan_rnn(
            args.rnn_config,
            args.output_root,
            seq_range,
            horizon_range,
            args.results_csv,
            fieldnames,
        )
    except Exception as e:
        print(f"[WARN] RNN scan failed: {e}")

    # GNN
    try:
        scan_gnn(
            args.gnn_config,
            args.output_root,
            seq_range,
            horizon_range,
            args.results_csv,
            fieldnames,
        )
    except Exception as e:
        print(f"[WARN] GNN scan failed: {e}")

    # Transformer
    try:
        scan_transformer(
            args.transformer_config,
            args.output_root,
            seq_range,
            horizon_range,
            args.results_csv,
            fieldnames,
        )
    except Exception as e:
        print(f"[WARN] Transformer scan failed: {e}")

    # TCN
    try:
        scan_tcn(
            args.tcn_config,
            args.output_root,
            seq_range,
            horizon_range,
            args.results_csv,
            fieldnames,
        )
    except Exception as e:
        print(f"[WARN] TCN scan failed: {e}")

    # # Informer
    # try:
    #     scan_informer(
    #         args.informer_config,
    #         args.output_root,
    #         seq_range,
    #         horizon_range,
    #         args.results_csv,
    #         fieldnames,
    #     )
    # except Exception as e:
    #     print(f"[WARN] Informer scan failed: {e}")

    # TimesNet (TimesNet++ không được scan)
    try:
        scan_timesnet(
            args.timesnet_config,
            args.output_root,
            seq_range,
            horizon_range,
            args.results_csv,
            fieldnames,
        )
    except Exception as e:
        print(f"[WARN] TimesNet scan failed: {e}")

    # GMAN / SOTA
    try:
        scan_sota(
            args.sota_config,
            args.output_root,
            seq_range,
            horizon_range,
            args.results_csv,
            fieldnames,
        )
    except Exception as e:
        print(f"[WARN] SOTA scan failed: {e}")

    print(f"\n✅ Deep model scan complete. Results saved to {args.results_csv}")


if __name__ == "__main__":
    main()
