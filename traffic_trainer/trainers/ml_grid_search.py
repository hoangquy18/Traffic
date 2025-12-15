"""Grid search tuning for traditional ML models (tree, XGBoost, ARIMA, SARIMA).

Chạy một lần để tune nhiều mô hình:
- Decision Tree
- XGBoost
- ARIMA
- SARIMA

Kết quả sẽ được ghi vào file CSV, mỗi dòng là một combination hyperparameters.
"""

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from traffic_trainer.trainers import (
    ARIMATrainer,
    ARIMATrainingConfig,
    DecisionTreeTrainer,
    DecisionTreeTrainingConfig,
    SARIMATrainer,
    SARIMATrainingConfig,
    XGBoostTrainer,
    XGBoostTrainingConfig,
)
from traffic_trainer.trainers.arima_trainer import load_config as load_arima_config
from traffic_trainer.trainers.decision_tree_trainer import (
    load_config as load_dt_config,
)
from traffic_trainer.trainers.sarima_trainer import load_config as load_sarima_config
from traffic_trainer.trainers.xgboost_trainer import load_config as load_xgb_config


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


def score_ts_results(test_results: Dict[str, Any], primary_horizon: int) -> float:
    """Get F1 macro from ARIMA/SARIMA test results."""
    horizon_key = f"horizon_{primary_horizon}"
    h_res = test_results.get(horizon_key, {})
    return float(h_res.get("f1_macro", 0.0))


def run_grid_search_decision_tree(
    base_config_path: Path,
    base_output_dir: Path,
    param_grid: Dict[str, Iterable[Any]],
    sequence_length: int,
    prediction_horizon: int,
    results_csv: Path | None = None,
    csv_fieldnames: List[str] | None = None,
) -> List[Dict[str, Any]]:
    combos = cartesian_product(param_grid)
    results: List[Dict[str, Any]] = []

    for idx, params in enumerate(combos, start=1):
        print(f"\n=== DecisionTree grid search {idx}/{len(combos)} ===")
        print(f"Params: {params}")

        config: DecisionTreeTrainingConfig = load_dt_config(base_config_path)
        # Override data-related parameters
        config.sequence_length = sequence_length
        config.prediction_horizons = [prediction_horizon]

        # Override model hyperparameters
        for k, v in params.items():
            setattr(config, k, v)

        # Output dir for this run
        config.output_dir = make_run_output_dir(
            base_output_dir, "decision_tree", params, idx
        )

        trainer = DecisionTreeTrainer(config)
        train_results = trainer.train()

        primary_h = config.prediction_horizons[0]
        metrics = extract_ml_val_metrics(train_results["history"], primary_h)

        row: Dict[str, Any] = {
            "model": "DecisionTree",
            "primary_horizon": primary_h,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
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
) -> List[Dict[str, Any]]:
    combos = cartesian_product(param_grid)
    results: List[Dict[str, Any]] = []

    for idx, params in enumerate(combos, start=1):
        print(f"\n=== XGBoost grid search {idx}/{len(combos)} ===")
        print(f"Params: {params}")

        config: XGBoostTrainingConfig = load_xgb_config(base_config_path)
        # Override data-related parameters
        config.sequence_length = sequence_length
        config.prediction_horizons = [prediction_horizon]

        # Override model hyperparameters
        for k, v in params.items():
            setattr(config, k, v)

        # Output dir for this run
        config.output_dir = make_run_output_dir(base_output_dir, "xgboost", params, idx)

        trainer = XGBoostTrainer(config)
        train_results = trainer.train()

        primary_h = config.prediction_horizons[0]
        metrics = extract_ml_val_metrics(train_results["history"], primary_h)

        row: Dict[str, Any] = {
            "model": "XGBoost",
            "primary_horizon": primary_h,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
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


def run_grid_search_arima(
    base_config_path: Path,
    base_output_dir: Path,
    param_grid: Dict[str, Iterable[Any]],
    sequence_length: int,
    prediction_horizon: int,
    results_csv: Path | None = None,
    csv_fieldnames: List[str] | None = None,
) -> List[Dict[str, Any]]:
    combos = cartesian_product(param_grid)
    results: List[Dict[str, Any]] = []

    for idx, params in enumerate(combos, start=1):
        print(f"\n=== ARIMA grid search {idx}/{len(combos)} ===")
        print(f"Params: {params}")

        config: ARIMATrainingConfig = load_arima_config(base_config_path)
        # Override data-related parameters
        config.sequence_length = sequence_length
        config.prediction_horizons = [prediction_horizon]
        for k, v in params.items():
            setattr(config, k, v)

        config.output_dir = make_run_output_dir(base_output_dir, "arima", params, idx)

        trainer = ARIMATrainer(config)
        _ = trainer.train()
        test_results = trainer.evaluate()

        primary_h = config.prediction_horizons[0]
        score = score_ts_results(test_results, primary_h)

        row: Dict[str, Any] = {
            "model": "ARIMA",
            "primary_horizon": primary_h,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
            "score_f1_macro": score,
            "output_dir": str(config.output_dir),
        }
        row.update(params)
        results.append(row)

        if results_csv is not None and csv_fieldnames is not None:
            append_row_csv(results_csv, csv_fieldnames, row)

    return results


def run_grid_search_sarima(
    base_config_path: Path,
    base_output_dir: Path,
    param_grid: Dict[str, Iterable[Any]],
    sequence_length: int,
    prediction_horizon: int,
    results_csv: Path | None = None,
    csv_fieldnames: List[str] | None = None,
) -> List[Dict[str, Any]]:
    combos = cartesian_product(param_grid)
    results: List[Dict[str, Any]] = []

    for idx, params in enumerate(combos, start=1):
        print(f"\n=== SARIMA grid search {idx}/{len(combos)} ===")
        print(f"Params: {params}")

        config: SARIMATrainingConfig = load_sarima_config(base_config_path)
        # Override data-related parameters
        config.sequence_length = sequence_length
        config.prediction_horizons = [prediction_horizon]
        for k, v in params.items():
            setattr(config, k, v)

        config.output_dir = make_run_output_dir(base_output_dir, "sarima", params, idx)

        trainer = SARIMATrainer(config)
        _ = trainer.train()
        test_results = trainer.evaluate()

        primary_h = config.prediction_horizons[0]
        score = score_ts_results(test_results, primary_h)

        row: Dict[str, Any] = {
            "model": "SARIMA",
            "primary_horizon": primary_h,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
            "score_f1_macro": score,
            "output_dir": str(config.output_dir),
        }
        row.update(params)
        results.append(row)

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
        description="Grid search tuning for ML models (DecisionTree, XGBoost, ARIMA, SARIMA)"
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
        "--arima-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "arima_config.yaml",
        help="YAML config path cho ARIMA",
    )
    parser.add_argument(
        "--sarima-config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "sarima_config.yaml",
        help="YAML config path cho SARIMA",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
                )
                all_results.extend(xgb_results)
    except Exception as e:
        print(f"[WARN] XGBoost grid search failed: {e}")

    # ============================
    # 3) ARIMA grid search
    # ============================
    arima_param_grid: Dict[str, Iterable[Any]] = {
        # (p, d, q)
        "order": [(1, 1, 1), (2, 1, 2)],
        "max_iter": [50],
    }
    arima_param_fields = list(arima_param_grid.keys())

    try:
        for seq_len in range(1, 7):
            for horizon in range(1, 6):
                arima_results = run_grid_search_arima(
                    args.arima_config,
                    args.output_root,
                    arima_param_grid,
                    sequence_length=seq_len,
                    prediction_horizon=horizon,
                    results_csv=args.results_csv,
                    csv_fieldnames=base_metric_fields + arima_param_fields,
                )
                all_results.extend(arima_results)
    except Exception as e:
        print(f"[WARN] ARIMA grid search failed: {e}")

    # ============================
    # 4) SARIMA grid search
    # ============================
    sarima_param_grid: Dict[str, Iterable[Any]] = {
        # (p, d, q)
        "order": [(1, 1, 1)],
        # (P, D, Q, s) với s=24 cho dữ liệu theo giờ
        "seasonal_order": [(1, 1, 1, 24), (2, 1, 1, 24)],
        "max_iter": [50],
    }
    sarima_param_fields = list(sarima_param_grid.keys())

    try:
        for seq_len in range(1, 7):
            for horizon in range(1, 6):
                sarima_results = run_grid_search_sarima(
                    args.sarima_config,
                    args.output_root,
                    sarima_param_grid,
                    sequence_length=seq_len,
                    prediction_horizon=horizon,
                    results_csv=args.results_csv,
                    csv_fieldnames=base_metric_fields + sarima_param_fields,
                )
                all_results.extend(sarima_results)
    except Exception as e:
        print(f"[WARN] SARIMA grid search failed: {e}")

    # Ghi toàn bộ kết quả ra CSV
    write_results_csv(args.results_csv, all_results)


if __name__ == "__main__":
    main()
