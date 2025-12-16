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
from traffic_trainer.data import load_dataset, load_graph_dataset

# Global dataset cache: key = (seq_len, tuple(prediction_horizons), model_type, csv_path)
_DATASET_CACHE: Dict[tuple, Any] = {}

# Weather features to filter out when --no-weather flag is used
WEATHER_FEATURES = {
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
}


def remove_weather_features(config: Any) -> None:
    """Remove weather features from config's numerical_features if present."""
    if hasattr(config, "numerical_features") and config.numerical_features:
        # Filter out weather features
        config.numerical_features = [
            f for f in config.numerical_features if f not in WEATHER_FEATURES
        ]
        print(
            f"  [NO-WEATHER] Removed weather features. Remaining: {config.numerical_features}"
        )


def get_or_load_dataset(
    csv_path: Path,
    seq_len: int,
    prediction_horizons: List[int],
    model_type: str,  # "sequential" or "graph"
    config: Any,  # Config object để lấy các tham số khác
) -> tuple:
    """Load dataset từ cache hoặc load mới nếu chưa có.

    Returns:
        (train_ds, val_ds, test_ds, feature_names, scaler, metadata) cho sequential
        hoặc (train_ds, val_ds, test_ds, road_graph, feature_names, scaler, metadata) cho graph
    """
    cache_key = (
        seq_len,
        tuple(prediction_horizons),
        model_type,
        str(csv_path),
        config.train_ratio,
        config.val_ratio,
        config.resample_rule,
        tuple(sorted(config.numerical_features or [])),
        tuple(sorted(config.categorical_features or [])),
    )

    if cache_key in _DATASET_CACHE:
        print(
            f"  [CACHE HIT] Reusing dataset for seq_len={seq_len}, horizons={prediction_horizons}"
        )
        return _DATASET_CACHE[cache_key]

    print(
        f"  [CACHE MISS] Loading dataset for seq_len={seq_len}, horizons={prediction_horizons}..."
    )

    use_time_embedding = (
        hasattr(config, "time_embedding_dim")
        and config.time_embedding_dim is not None
        and config.time_embedding_dim > 0
    )
    use_segment_embedding = (
        hasattr(config, "segment_embedding_dim")
        and config.segment_embedding_dim is not None
        and config.segment_embedding_dim > 0
    )

    if model_type == "graph":
        # Load graph dataset
        datasets = load_graph_dataset(
            csv_path=csv_path,
            sequence_length=seq_len,
            feature_columns={
                "numerical": config.numerical_features or [],
                "categorical": config.categorical_features or [],
            },
            prediction_horizons=tuple(prediction_horizons),
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            resample_rule=config.resample_rule,
            add_reverse_edges=getattr(config, "add_reverse_edges", True),
            graph_mode=getattr(config, "graph_mode", "topology"),
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )
    else:
        # Load sequential dataset
        datasets = load_dataset(
            csv_path=csv_path,
            sequence_length=seq_len,
            feature_columns={
                "numerical": config.numerical_features or [],
                "categorical": config.categorical_features or [],
            },
            prediction_horizons=tuple(prediction_horizons),
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            resample_rule=config.resample_rule,
            normalize=True,
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )

    _DATASET_CACHE[cache_key] = datasets
    return datasets


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


def create_trainer_with_cached_data(
    trainer_class, config, cached_datasets, model_type: str
):
    """Create trainer instance với cached datasets.

    Args:
        trainer_class: Trainer class (RNNTrainer, GraphTrainer, etc.)
        config: Config object
        cached_datasets: Pre-loaded datasets tuple
        model_type: "sequential" or "graph"

    Returns:
        Trainer instance với datasets đã được inject
    """
    import torch
    from traffic_trainer.trainers.base import BaseTrainer
    from traffic_trainer.data import LOS_LEVELS
    from torch.optim import AdamW

    # Create trainer instance nhưng skip __init__
    trainer = trainer_class.__new__(trainer_class)

    # Setup basic attributes (copy from BaseTrainer.__init__)
    trainer.config = config
    trainer.device = torch.device(
        config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    )

    # Inject cached datasets thay vì gọi _load_data()
    if model_type == "sequential":
        (
            trainer.train_dataset,
            trainer.val_dataset,
            trainer.test_dataset,
            trainer.feature_names,
            trainer.scaler,
            metadata,
        ) = cached_datasets
        trainer.segment_encoder = metadata.get("segment_encoder")
        trainer.segment_vocab_size = metadata.get("segment_vocab_size")
        trainer.metadata = metadata
    else:  # graph
        (
            trainer.train_dataset,
            trainer.val_dataset,
            trainer.test_dataset,
            trainer.road_graph,
            trainer.feature_names,
            trainer.scaler,
            trainer.metadata,
        ) = cached_datasets
        trainer.edge_index = trainer.road_graph.edge_index.to(trainer.device)
        trainer.num_nodes = trainer.road_graph.num_nodes
        trainer.segment_vocab_size = trainer.metadata.get("segment_vocab_size")
        if hasattr(trainer, "num_segments"):
            trainer.num_segments = trainer.road_graph.num_nodes

    # Derived attributes
    trainer.prediction_horizons = sorted({int(h) for h in config.prediction_horizons})
    trainer.num_horizons = len(trainer.prediction_horizons)
    trainer.num_classes = len(LOS_LEVELS)

    # Create data loaders
    trainer._create_dataloaders()

    # Create model
    trainer.model = trainer._create_model()
    trainer.model = trainer.model.to(trainer.device)

    # Loss function
    trainer.criterion = trainer._create_criterion()

    # Optimizer
    trainer.optimizer = AdamW(
        trainer.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    trainer.scheduler = trainer._create_scheduler()

    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    trainer._init_wandb()

    # Training state
    trainer.best_val_f1 = 0.0
    trainer.patience_counter = 0
    trainer.current_epoch = 0

    # Print model info
    trainer._print_model_info()

    return trainer


def extract_last_val_metrics(
    history: Dict[str, Any], primary_horizon: int = 1
) -> Dict[str, float]:
    """Extract last epoch validation metrics from BaseTrainer history.

    Với multi-horizon, lấy metrics cho primary_horizon (mặc định horizon đầu tiên).
    """
    val_hist = history.get("val", [])
    if not val_hist:
        return {}
    last = val_hist[-1] or {}

    # Lấy metrics cho primary horizon
    precision_key = f"precision_h{primary_horizon}"
    recall_key = f"recall_h{primary_horizon}"

    return {
        "val_loss": float(last.get("loss", 0.0)),
        "val_accuracy": float(last.get("accuracy", 0.0)),
        "val_f1_macro": float(last.get("f1_macro", 0.0)),
        "val_f1_weighted": float(last.get("f1_weighted", 0.0)),
        "val_precision_macro": float(
            last.get(precision_key, last.get("precision", 0.0))
        ),
        "val_recall_macro": float(last.get(recall_key, last.get("recall", 0.0))),
    }


def scan_rnn(
    config_path: Path,
    output_root: Path,
    seq_range: range,
    horizon_range: range,
    csv_path: Path,
    fieldnames: List[str],
    no_weather: bool = False,
) -> None:
    # RNN không phải attention-based: ẩn tầng 64/128/256, layers 1/2/3
    hidden_dims = [128, 256]
    num_layers_list = [2, 3]
    emb_dims = [8, 16, 32]  # dùng cho cả time & segment embedding nếu có

    # Load base config để lấy csv_path và các tham số khác
    base_config = load_rnn_config(config_path)
    if no_weather:
        remove_weather_features(base_config)

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Multi-horizon prediction: 1h, 2h, 3h ahead
            prediction_horizons = [1, 2, 3]

            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="sequential",
                config=base_config,
            )

            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== RNN | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config = load_rnn_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = prediction_horizons
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

                        # Create trainer với cached datasets
                        trainer = create_trainer_with_cached_data(
                            RNNTrainer, config, cached_datasets, "sequential"
                        )
                        results = trainer.train()

                        # Lấy metrics cho primary horizon (horizon đầu tiên = 1)
                        metrics = extract_last_val_metrics(
                            results["history"], primary_horizon=1
                        )
                        row: Dict[str, Any] = {
                            "model": "RNN",
                            "primary_horizon": 1,  # Primary horizon luôn là 1
                            "sequence_length": seq_len,
                            "prediction_horizon": horizon,  # Giữ lại để track scan parameter
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
    no_weather: bool = False,
) -> None:
    # GNN (RNN+GNN) không hoàn toàn attention-based: cho phép 64/128/256
    hidden_dims = [128, 256]
    num_layers_list = [2, 3]
    emb_dims = [8, 16, 32]

    base_config = load_gnn_config(config_path)
    if no_weather:
        remove_weather_features(base_config)
    prediction_horizons = [1, 2, 3]  # Multi-horizon prediction

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="graph",
                config=base_config,
            )

            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== GNN | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: GraphTrainingConfig = load_gnn_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = prediction_horizons
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

                        trainer = create_trainer_with_cached_data(
                            GraphTrainer, config, cached_datasets, "graph"
                        )
                        results = trainer.train()

                        metrics = extract_last_val_metrics(
                            results["history"], primary_horizon=1
                        )
                        row: Dict[str, Any] = {
                            "model": "GNN",
                            "primary_horizon": 1,
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
    no_weather: bool = False,
) -> None:
    # Attention-based: chỉ 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]  # số layer Transformer
    emb_dims = [8, 16, 32]

    base_config = load_transformer_config(config_path)
    if no_weather:
        remove_weather_features(base_config)
    prediction_horizons = [1, 2, 3]  # Multi-horizon prediction

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="graph",
                config=base_config,
            )

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
                        config.prediction_horizons = prediction_horizons
                        config.hidden_dim = hidden_dim
                        config.num_transformer_layers = num_layers
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "transformer", seq_len, horizon
                        )

                        trainer = create_trainer_with_cached_data(
                            TransformerTrainer, config, cached_datasets, "graph"
                        )
                        results = trainer.train()

                        metrics = extract_last_val_metrics(
                            results["history"], primary_horizon=1
                        )
                        row: Dict[str, Any] = {
                            "model": "Transformer",
                            "primary_horizon": 1,
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
    no_weather: bool = False,
) -> None:
    # TCN có thể dùng attention, coi như attention-based: 64,128
    hidden_dims = [64, 128]
    # Không có tham số layer rõ ràng, dùng num_layers=0 như placeholder để log
    num_layers_list = [1, 2, 3]
    emb_dims = [8, 16, 32]

    base_config = load_tcn_config(config_path)
    if no_weather:
        remove_weather_features(base_config)
    prediction_horizons = [1, 2, 3]  # Multi-horizon prediction

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="graph",
                config=base_config,
            )

            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== TCN | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: TCNTrainingConfig = load_tcn_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = prediction_horizons
                        config.hidden_dim = hidden_dim
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "tcn", seq_len, horizon
                        )

                        trainer = create_trainer_with_cached_data(
                            TCNTrainer, config, cached_datasets, "graph"
                        )
                        results = trainer.train()

                        metrics = extract_last_val_metrics(
                            results["history"], primary_horizon=1
                        )
                        row: Dict[str, Any] = {
                            "model": "TCN",
                            "primary_horizon": 1,
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
    no_weather: bool = False,
) -> None:
    # Attention-based: 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]  # dùng cho e_layers/d_layers
    emb_dims = [8, 16, 32]

    base_config = load_informer_config(config_path)
    if no_weather:
        remove_weather_features(base_config)
    prediction_horizons = [1, 2, 3]  # Multi-horizon prediction

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="graph",
                config=base_config,
            )

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
                        config.out_len = horizon
                        config.prediction_horizons = prediction_horizons
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

                        trainer = create_trainer_with_cached_data(
                            InformerTrainer, config, cached_datasets, "graph"
                        )
                        results = trainer.train()

                        metrics = extract_last_val_metrics(
                            results["history"], primary_horizon=1
                        )
                        row: Dict[str, Any] = {
                            "model": "Informer",
                            "primary_horizon": 1,
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
    no_weather: bool = False,
) -> None:
    # Attention-based: 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]  # e_layers

    base_config = load_timesnet_config(config_path)
    if no_weather:
        remove_weather_features(base_config)
    prediction_horizons = [1, 2, 3]  # Multi-horizon prediction

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="graph",
                config=base_config,
            )

            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    print(
                        f"\n=== TimesNet | seq_len={seq_len}, horizon={horizon}, "
                        f"hidden_dim={hidden_dim}, layers={num_layers} ==="
                    )
                    config: TimesNetTrainingConfig = load_timesnet_config(config_path)
                    config.sequence_length = seq_len
                    config.prediction_horizons = prediction_horizons
                    # pred_len cho TimesNet nên >= max horizon
                    config.pred_len = horizon
                    config.hidden_dim = hidden_dim
                    config.e_layers = num_layers
                    config.output_dir = make_run_output_dir(
                        output_root, "timesnet", seq_len, horizon
                    )

                    trainer = create_trainer_with_cached_data(
                        TimesNetTrainer, config, cached_datasets, "graph"
                    )
                    results = trainer.train()

                    metrics = extract_last_val_metrics(
                        results["history"], primary_horizon=1
                    )
                    row: Dict[str, Any] = {
                        "model": "TimesNet",
                        "primary_horizon": 1,
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
    no_weather: bool = False,
) -> None:
    # GMAN là attention-based: 64,128
    hidden_dims = [64, 128]
    num_layers_list = [1, 2, 3]
    emb_dims = [8, 16, 32]

    base_config = load_sota_config(config_path)
    if no_weather:
        remove_weather_features(base_config)
    prediction_horizons = [1, 2, 3]  # Multi-horizon prediction

    for seq_len in seq_range:
        for horizon in horizon_range:
            # Load dataset một lần cho combination này
            cached_datasets = get_or_load_dataset(
                csv_path=base_config.csv_path,
                seq_len=seq_len,
                prediction_horizons=prediction_horizons,
                model_type="graph",
                config=base_config,
            )

            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for emb_dim in emb_dims:
                        print(
                            f"\n=== GMAN++ | seq_len={seq_len}, horizon={horizon}, "
                            f"hidden_dim={hidden_dim}, layers={num_layers}, emb_dim={emb_dim} ==="
                        )
                        config: SOTATrainingConfig = load_sota_config(config_path)
                        config.sequence_length = seq_len
                        config.prediction_horizons = prediction_horizons
                        config.hidden_dim = hidden_dim
                        config.num_layers = num_layers
                        if hasattr(config, "time_embedding_dim"):
                            config.time_embedding_dim = emb_dim
                        if hasattr(config, "segment_embedding_dim"):
                            config.segment_embedding_dim = emb_dim

                        config.output_dir = make_run_output_dir(
                            output_root, "gman", seq_len, horizon
                        )

                        trainer = create_trainer_with_cached_data(
                            SOTATrainer, config, cached_datasets, "graph"
                        )
                        results = trainer.train()

                        metrics = extract_last_val_metrics(
                            results["history"], primary_horizon=1
                        )
                        row: Dict[str, Any] = {
                            "model": "GMAN++",
                            "primary_horizon": 1,
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
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Remove weather features from numerical_features (only use traffic features)",
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
        "val_precision_macro",
        "val_recall_macro",
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
            no_weather=args.no_weather,
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
            no_weather=args.no_weather,
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
            no_weather=args.no_weather,
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
            no_weather=args.no_weather,
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
    #         no_weather=args.no_weather,
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
            no_weather=args.no_weather,
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
            no_weather=args.no_weather,
        )
    except Exception as e:
        print(f"[WARN] SOTA scan failed: {e}")

    print(f"\n✅ Deep model scan complete. Results saved to {args.results_csv}")


if __name__ == "__main__":
    main()
