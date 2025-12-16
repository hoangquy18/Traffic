"""Grid search tuning for traditional ML models (tree, XGBoost).

Chạy một lần để tune nhiều mô hình:
- Decision Tree
- XGBoost

Kết quả sẽ được ghi vào file CSV, mỗi dòng là một combination hyperparameters.
"""

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from traffic_trainer.trainers import (
    DecisionTreeTrainer,
    DecisionTreeTrainingConfig,
    XGBoostTrainer,
    XGBoostTrainingConfig,
)
from traffic_trainer.trainers.decision_tree_trainer import (
    load_config as load_dt_config,
)
from traffic_trainer.trainers.xgboost_trainer import load_config as load_xgb_config
from traffic_trainer.data import load_dataset

# Global dataset cache: key = (seq_len, tuple(prediction_horizons), csv_path, ...)
_ML_DATASET_CACHE: Dict[tuple, Any] = {}

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
        original_count = len(config.numerical_features)
        # Filter out weather features
        config.numerical_features = [
            f for f in config.numerical_features if f not in WEATHER_FEATURES
        ]
        removed_count = original_count - len(config.numerical_features)
        if removed_count > 0:
            print(
                f"  [NO-WEATHER] Removed {removed_count} weather features. "
                f"Remaining {len(config.numerical_features)} features: {config.numerical_features}"
            )


def get_or_load_ml_dataset(
    csv_path: Path,
    seq_len: int,
    prediction_horizons: List[int],
    config: Any,  # Config object để lấy các tham số khác
) -> tuple:
    """Load dataset từ cache hoặc load mới nếu chưa có cho ML models.

    Returns:
        (train_ds, val_ds, test_ds, feature_names, scaler, metadata)
    """
    cache_key = (
        seq_len,
        tuple(prediction_horizons),
        str(csv_path),
        config.train_ratio,
        config.val_ratio,
        config.resample_rule,
        tuple(sorted(config.numerical_features or [])),
        tuple(sorted(config.categorical_features or [])),
    )

    if cache_key in _ML_DATASET_CACHE:
        print(
            f"  [CACHE HIT] Reusing dataset for seq_len={seq_len}, horizons={prediction_horizons}"
        )
        return _ML_DATASET_CACHE[cache_key]

    print(
        f"  [CACHE MISS] Loading dataset for seq_len={seq_len}, horizons={prediction_horizons}..."
    )

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
        use_time_embedding=False,
        use_segment_embedding=False,
    )

    _ML_DATASET_CACHE[cache_key] = datasets
    return datasets


def create_ml_trainer_with_cached_data(trainer_class, config, cached_datasets):
    """Create ML trainer instance với cached datasets.

    Args:
        trainer_class: Trainer class (DecisionTreeTrainer, XGBoostTrainer, etc.)
        config: Config object
        cached_datasets: Pre-loaded datasets tuple

    Returns:
        Trainer instance với datasets đã được inject
    """
    from traffic_trainer.data import LOS_LEVELS

    # Create trainer instance nhưng skip __init__
    trainer = trainer_class.__new__(trainer_class)

    # Setup basic attributes (copy from MLBaseTrainer.__init__)
    trainer.config = config
    trainer.prediction_horizons = sorted({int(h) for h in config.prediction_horizons})
    trainer.num_horizons = len(trainer.prediction_horizons)
    trainer.num_classes = len(LOS_LEVELS)

    # Inject cached datasets thay vì gọi _load_data()
    train_ds, val_ds, test_ds, feature_names, scaler, metadata = cached_datasets

    trainer.feature_names = feature_names
    trainer.scaler = scaler
    trainer.metadata = metadata

    # Convert to numpy arrays (flatten sequences) - need to call method from instance
    # We'll create a temporary instance to get the method, or define it inline
    def dataset_to_arrays(dataset):
        """Convert dataset to feature matrix and target arrays."""
        import numpy as np

        features_list = []
        targets_list = []
        for i in range(len(dataset)):
            sample, targets = dataset[i]
            # Use the last timestep of the sequence as features
            features = sample["features"][-1].numpy()  # [num_features]
            features_list.append(features)
            targets_list.append(targets.numpy())  # [num_horizons]
        X = np.array(features_list)  # [num_samples, num_features]
        y = np.array(targets_list)  # [num_samples, num_horizons]
        return X, y

    trainer.X_train, trainer.y_train = dataset_to_arrays(train_ds)
    trainer.X_val, trainer.y_val = dataset_to_arrays(val_ds)
    trainer.X_test, trainer.y_test = dataset_to_arrays(test_ds)

    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    trainer.models = {}  # One model per horizon
    trainer.history = {"train": [], "val": []}

    return trainer


def cartesian_product(param_grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Generate list of param combinations from a grid dict."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values_product = list(product(*(param_grid[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in values_product]


def make_run_output_dir(
    base_dir: Path, model_name: str, params: Dict[str, Any], run_idx: int
) -> Path:
    """Create a readable, unique output directory for a given param combo."""
    # Short param signature for folder name
    parts: List[str] = []
    for k in sorted(params.keys()):
        v = params[k]
        parts.append(f"{k}={v}")
    sig = "__".join(parts) if parts else "default"
    # Avoid very long paths
    if len(sig) > 100:
        sig = sig[:100]
    out_dir = base_dir / model_name / f"run_{run_idx:03d}_{sig}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_ml_val_metrics(
    history: Dict[str, Any], primary_horizon: int
) -> Dict[str, float]:
    """Get main classification metrics on validation set for a given horizon.

    Trong MLBaseTrainer, history["val"] là list, mỗi phần tử tương ứng với 1 horizon,
    chứ không phải 1 epoch. Vì vậy ta cần chọn đúng dict có chứa các key *_h{h}.
    """
    val_history = history.get("val", [])
    if not val_history:
        return {}

    h = primary_horizon
    key_prefix = f"accuracy_h{h}"

    # Tìm entry nào thực sự chứa metrics cho horizon này
    metrics_for_h = None
    for entry in val_history:
        if key_prefix in entry:
            metrics_for_h = entry
            break

    if metrics_for_h is None:
        # fallback: không tìm được, trả về rỗng
        return {}

    return {
        "val_accuracy": float(metrics_for_h.get(f"accuracy_h{h}", 0.0)),
        "val_f1_macro": float(metrics_for_h.get(f"f1_macro_h{h}", 0.0)),
        "val_f1_weighted": float(metrics_for_h.get(f"f1_weighted_h{h}", 0.0)),
        "val_precision_macro": float(metrics_for_h.get(f"precision_h{h}", 0.0)),
        "val_recall_macro": float(metrics_for_h.get(f"recall_h{h}", 0.0)),
    }


def run_grid_search_decision_tree(
    base_config_path: Path,
    base_output_dir: Path,
    param_grid: Dict[str, Iterable[Any]],
    sequence_length: int,
    prediction_horizon: int,
    results_csv: Path | None = None,
    csv_fieldnames: List[str] | None = None,
    no_weather: bool = False,
) -> List[Dict[str, Any]]:
    combos = cartesian_product(param_grid)
    results: List[Dict[str, Any]] = []

    # Load base config để lấy csv_path và các tham số khác
    base_config = load_dt_config(base_config_path)
    if no_weather:
        remove_weather_features(base_config)

    # Multi-horizon prediction: 1h, 2h, 3h ahead (giống DL scan)
    prediction_horizons = [1, 2, 3]

    # Load dataset một lần cho combination này
    cached_datasets = get_or_load_ml_dataset(
        csv_path=base_config.csv_path,
        seq_len=sequence_length,
        prediction_horizons=prediction_horizons,
        config=base_config,
    )

    for idx, params in enumerate(combos, start=1):
        print(f"\n=== DecisionTree grid search {idx}/{len(combos)} ===")
        print(f"Params: {params}")

        config: DecisionTreeTrainingConfig = load_dt_config(base_config_path)
        if no_weather:
            remove_weather_features(config)
        # Override data-related parameters
        config.sequence_length = sequence_length
        config.prediction_horizons = prediction_horizons

        # Override model hyperparameters
        for k, v in params.items():
            setattr(config, k, v)

        # Output dir for this run
        config.output_dir = make_run_output_dir(
            base_output_dir, "decision_tree", params, idx
        )

        # Create trainer với cached datasets
        trainer = create_ml_trainer_with_cached_data(
            DecisionTreeTrainer, config, cached_datasets
        )
        train_results = trainer.train()

        # Lấy metrics cho primary horizon (horizon đầu tiên = 1)
        primary_h = 1
        metrics = extract_ml_val_metrics(train_results["history"], primary_h)

        row: Dict[str, Any] = {
            "model": "DecisionTree",
            "primary_horizon": 1,  # Primary horizon luôn là 1 (giống DL scan)
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,  # Giữ lại để track scan parameter
            # giữ lại field score_f1_macro cho tiện sort
            "score_f1_macro": metrics.get("val_f1_macro", 0.0),
            "output_dir": str(config.output_dir),
            **metrics,
        }
        row.update(params)
        results.append(row)

        # Ghi incremental vào CSV ngay sau khi xong mỗi combination
        if results_csv is not None and csv_fieldnames is not None:
            append_row_csv(results_csv, csv_fieldnames, row)

    return results


def run_grid_search_xgboost(
    base_config_path: Path,
    base_output_dir: Path,
    param_grid: Dict[str, Iterable[Any]],
    sequence_length: int,
    prediction_horizon: int,
    results_csv: Path | None = None,
    csv_fieldnames: List[str] | None = None,
    no_weather: bool = False,
) -> List[Dict[str, Any]]:
    combos = cartesian_product(param_grid)
    results: List[Dict[str, Any]] = []

    # Load base config để lấy csv_path và các tham số khác
    base_config = load_xgb_config(base_config_path)
    if no_weather:
        remove_weather_features(base_config)

    # Multi-horizon prediction: 1h, 2h, 3h ahead (giống DL scan)
    prediction_horizons = [1, 2, 3]

    # Load dataset một lần cho combination này
    cached_datasets = get_or_load_ml_dataset(
        csv_path=base_config.csv_path,
        seq_len=sequence_length,
        prediction_horizons=prediction_horizons,
        config=base_config,
    )

    for idx, params in enumerate(combos, start=1):
        print(f"\n=== XGBoost grid search {idx}/{len(combos)} ===")
        print(f"Params: {params}")

        config: XGBoostTrainingConfig = load_xgb_config(base_config_path)
        if no_weather:
            remove_weather_features(config)
        # Override data-related parameters
        config.sequence_length = sequence_length
        config.prediction_horizons = prediction_horizons

        # Override model hyperparameters
        for k, v in params.items():
            setattr(config, k, v)

        # Output dir for this run
        config.output_dir = make_run_output_dir(base_output_dir, "xgboost", params, idx)

        # Create trainer với cached datasets
        trainer = create_ml_trainer_with_cached_data(
            XGBoostTrainer, config, cached_datasets
        )
        train_results = trainer.train()

        # Lấy metrics cho primary horizon (horizon đầu tiên = 1)
        primary_h = 1
        metrics = extract_ml_val_metrics(train_results["history"], primary_h)

        row: Dict[str, Any] = {
            "model": "XGBoost",
            "primary_horizon": 1,  # Primary horizon luôn là 1 (giống DL scan)
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,  # Giữ lại để track scan parameter
            "score_f1_macro": metrics.get("val_f1_macro", 0.0),
            "output_dir": str(config.output_dir),
            **metrics,
        }
        row.update(params)
        results.append(row)

        # Ghi incremental vào CSV
        if results_csv is not None and csv_fieldnames is not None:
            append_row_csv(results_csv, csv_fieldnames, row)

    return results


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write all results (different models) into a single CSV."""
    if not rows:
        print("No results to write.")
        return

    # Collect all keys that appear in any row
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n✅ Grid search results written to: {path}")


def append_row_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Append a single row to CSV, tạo file + header nếu chưa tồn tại.

    Giúp lưu kết quả ngay sau mỗi run; nếu script lỗi giữa chừng thì vẫn
    không mất các run đã hoàn thành.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Bảo đảm đủ cột, thiếu thì để rỗng
        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search tuning for ML models (DecisionTree, XGBoost)"
    )
    parser.add_argument(
        "--dt-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "decision_tree_config.yaml",
        help="YAML config path cho Decision Tree",
    )
    parser.add_argument(
        "--xgb-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "xgboost_config.yaml",
        help="YAML config path cho XGBoost",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments") / "ml_grid_search",
        help="Thư mục gốc để lưu toàn bộ models/experiments của grid search",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "ml_grid_search" / "grid_search_results.csv",
        help="Đường dẫn file CSV lưu tổng hợp kết quả",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Remove weather features from numerical_features (only use traffic features)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-adjust CSV filename if --no-weather is used
    if args.no_weather:
        csv_path = args.results_csv
        if csv_path.suffix == ".csv":
            # Insert "_no_weather" before .csv extension
            csv_path = csv_path.parent / f"{csv_path.stem}_no_weather{csv_path.suffix}"
        else:
            csv_path = csv_path.parent / f"{csv_path.name}_no_weather"
        args.results_csv = csv_path
        print(
            f"\n🌤️  [NO-WEATHER MODE] Weather features will be removed from all models"
        )
        print(f"   Results will be saved to: {args.results_csv}")
    else:
        print(f"\n🌦️  [WITH-WEATHER MODE] Using all features including weather data")
        print(f"   Results will be saved to: {args.results_csv}")

    print(f"📊 [MULTI-HORIZON] All models will use prediction_horizons = [1, 2, 3]")
    print(f"   Metrics will be reported for primary horizon = 1 (giống DL scan)")

    all_results: List[Dict[str, Any]] = []

    # Định nghĩa danh sách cột cố định cho CSV để có thể ghi incremental
    base_metric_fields: List[str] = [
        "model",
        "primary_horizon",
        "sequence_length",
        "prediction_horizon",
        "score_f1_macro",
        "output_dir",
        "val_accuracy",
        "val_f1_macro",
        "val_f1_weighted",
        "val_precision_macro",
        "val_recall_macro",
    ]

    # ============================
    # 1) Decision Tree grid search
    # ============================
    dt_param_grid: Dict[str, Iterable[Any]] = {
        # Các giá trị dưới đây chỉ là ví dụ, bạn có thể chỉnh lại cho phù hợp
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }
    dt_param_fields = list(dt_param_grid.keys())

    try:
        for seq_len in range(1, 7):  # 1..6
            for horizon in range(1, 6):  # 1..5
                dt_results = run_grid_search_decision_tree(
                    args.dt_config,
                    args.output_root,
                    dt_param_grid,
                    sequence_length=seq_len,
                    prediction_horizon=horizon,
                    results_csv=args.results_csv,
                    csv_fieldnames=base_metric_fields + dt_param_fields,
                    no_weather=args.no_weather,
                )
                all_results.extend(dt_results)
    except Exception as e:
        print(f"[WARN] DecisionTree grid search failed: {e}")

    # ============================
    # 2) XGBoost grid search
    # ============================
    xgb_param_grid: Dict[str, Iterable[Any]] = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 5],
        "gamma": [0.0, 1.0],
        "reg_alpha": [0.0],
        "reg_lambda": [1.0],
    }
    xgb_param_fields = list(xgb_param_grid.keys())

    try:
        for seq_len in range(1, 7):
            for horizon in range(1, 6):
                xgb_results = run_grid_search_xgboost(
                    args.xgb_config,
                    args.output_root,
                    xgb_param_grid,
                    sequence_length=seq_len,
                    prediction_horizon=horizon,
                    results_csv=args.results_csv,
                    csv_fieldnames=base_metric_fields + xgb_param_fields,
                    no_weather=args.no_weather,
                )
                all_results.extend(xgb_results)
    except Exception as e:
        print(f"[WARN] XGBoost grid search failed: {e}")

    # Ghi toàn bộ kết quả ra CSV (nếu có kết quả mới)
    if all_results:
        write_results_csv(args.results_csv, all_results)
        print(f"\n✅ Grid search complete!")
        if args.no_weather:
            print(f"   Mode: NO-WEATHER (only traffic features)")
        else:
            print(f"   Mode: WITH-WEATHER (all features)")
        print(f"   Total runs: {len(all_results)}")
        print(f"   Results saved to: {args.results_csv}")
    else:
        print("\n⚠️  No results to save. Check for errors above.")


if __name__ == "__main__":
    main()
