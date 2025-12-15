"""Base trainer for traditional ML models (XGBoost, Decision Tree, etc.)."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

import yaml
from traffic_trainer.data import LOS_LEVELS, load_dataset


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and flatten YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


@dataclass
class MLBaseConfig:
    """Base configuration for ML models."""

    # Paths
    csv_path: Path = Path("data.csv")
    output_dir: Path = Path("experiments/ml_run01")

    # Data
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    prediction_horizons: List[int] = field(default_factory=lambda: [1])
    sequence_length: int = 8
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    resample_rule: Optional[str] = "1H"

    # Training
    random_state: int = 42


class MLBaseTrainer(ABC):
    """Base trainer for traditional ML models."""

    def __init__(self, config: MLBaseConfig) -> None:
        self.config = config
        self.prediction_horizons = sorted({int(h) for h in config.prediction_horizons})
        self.num_horizons = len(self.prediction_horizons)
        self.num_classes = len(LOS_LEVELS)

        # Load data
        self._load_data()

        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.models = {}  # One model per horizon
        self.history = {"train": [], "val": []}

    def _load_data(self) -> None:
        """Load and prepare data for ML models."""
        print(f"Loading dataset from {self.config.csv_path}...")

        train_ds, val_ds, test_ds, feature_names, scaler, metadata = load_dataset(
            csv_path=self.config.csv_path,
            sequence_length=self.config.sequence_length,
            feature_columns={
                "numerical": self.config.numerical_features,
                "categorical": self.config.categorical_features,
            },
            prediction_horizons=tuple(self.config.prediction_horizons),
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            resample_rule=self.config.resample_rule,
            normalize=True,
            use_time_embedding=False,
            use_segment_embedding=False,
        )

        self.feature_names = feature_names
        self.scaler = scaler
        self.metadata = metadata

        # Convert to numpy arrays (flatten sequences)
        self.X_train, self.y_train = self._dataset_to_arrays(train_ds)
        self.X_val, self.y_val = self._dataset_to_arrays(val_ds)
        self.X_test, self.y_test = self._dataset_to_arrays(test_ds)

        print(f"Train samples: {len(self.X_train)}")
        print(f"Val samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {len(self.feature_names)}")

    def _dataset_to_arrays(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Convert dataset to feature matrix and target arrays."""
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

    @abstractmethod
    def _create_model(self, horizon: int) -> Any:
        """Create a model for a specific horizon."""
        pass

    @abstractmethod
    def _train_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Train a model."""
        pass

    @abstractmethod
    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def _get_model_type(self) -> str:
        """Return model type string."""
        pass

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int,
    ) -> Dict[str, float]:
        """Compute metrics for a specific horizon."""
        metrics = {}

        # Filter valid samples (target >= 0)
        valid_mask = y_true >= 0
        if valid_mask.sum() == 0:
            return metrics

        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        metrics[f"accuracy_h{horizon}"] = float(
            accuracy_score(y_true_valid, y_pred_valid)
        )
        metrics[f"f1_macro_h{horizon}"] = float(
            f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        )
        metrics[f"f1_weighted_h{horizon}"] = float(
            f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
        )
        metrics[f"precision_h{horizon}"] = float(
            precision_score(
                y_true_valid, y_pred_valid, average="macro", zero_division=0
            )
        )
        metrics[f"recall_h{horizon}"] = float(
            recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        )

        # Per-class F1
        f1_per_class = f1_score(
            y_true_valid, y_pred_valid, average=None, zero_division=0
        )
        for cls_idx, cls_name in enumerate(LOS_LEVELS.keys()):
            if cls_idx < len(f1_per_class):
                metrics[f"f1_{cls_name}_h{horizon}"] = float(f1_per_class[cls_idx])

        return metrics

    def train(self) -> Dict[str, Any]:
        """Train models for all horizons."""
        print(f"\n{'='*60}")
        print(
            f"Training {self._get_model_type()} for horizons {self.prediction_horizons}"
        )
        print(f"{'='*60}\n")

        for horizon_idx, horizon in enumerate(self.prediction_horizons):
            print(f"\n--- Training for horizon +{horizon} ---")

            # Create model
            model = self._create_model(horizon)

            # Get targets for this horizon
            y_train_h = self.y_train[:, horizon_idx]
            y_val_h = self.y_val[:, horizon_idx]

            # Train
            self._train_model(model, self.X_train, y_train_h)

            # Evaluate on train
            y_pred_train = self._predict(model, self.X_train)
            train_metrics = self._compute_metrics(y_train_h, y_pred_train, horizon)

            # Evaluate on val
            y_pred_val = self._predict(model, self.X_val)
            val_metrics = self._compute_metrics(y_val_h, y_pred_val, horizon)

            # Store model
            self.models[horizon] = model

            # Store metrics
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            print(
                f"Train - Acc: {train_metrics.get(f'accuracy_h{horizon}', 0):.4f}, "
                f"F1: {train_metrics.get(f'f1_macro_h{horizon}', 0):.4f}"
            )
            print(
                f"Val   - Acc: {val_metrics.get(f'accuracy_h{horizon}', 0):.4f}, "
                f"F1: {val_metrics.get(f'f1_macro_h{horizon}', 0):.4f}"
            )

        # Save history
        with open(self.config.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return {"history": self.history}

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set."""
        print(f"\n{'='*60}")
        print("Evaluating on test set...")
        print(f"{'='*60}")

        results = {"model_type": self._get_model_type()}
        los_names = list(LOS_LEVELS.keys())

        for horizon_idx, horizon in enumerate(self.prediction_horizons):
            model = self.models[horizon]
            y_test_h = self.y_test[:, horizon_idx]

            # Predict
            y_pred_test = self._predict(model, self.X_test)

            # Filter valid samples
            valid_mask = y_test_h >= 0
            y_test_valid = y_test_h[valid_mask]
            y_pred_valid = y_pred_test[valid_mask]

            if len(y_test_valid) > 0:
                results[f"horizon_{horizon}"] = {
                    "accuracy": float(accuracy_score(y_test_valid, y_pred_valid)),
                    "f1_macro": float(
                        f1_score(
                            y_test_valid, y_pred_valid, average="macro", zero_division=0
                        )
                    ),
                    "f1_weighted": float(
                        f1_score(
                            y_test_valid,
                            y_pred_valid,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "precision_macro": float(
                        precision_score(
                            y_test_valid, y_pred_valid, average="macro", zero_division=0
                        )
                    ),
                    "recall_macro": float(
                        recall_score(
                            y_test_valid, y_pred_valid, average="macro", zero_division=0
                        )
                    ),
                }

                # Per-class F1
                f1_per_class = f1_score(
                    y_test_valid, y_pred_valid, average=None, zero_division=0
                )
                for cls_idx, cls_name in enumerate(los_names):
                    if cls_idx < len(f1_per_class):
                        results[f"horizon_{horizon}"][f"f1_{cls_name}"] = float(
                            f1_per_class[cls_idx]
                        )

                # Print results
                print(f"\n=== Test Results (Horizon +{horizon}) ===")
                print(f"Accuracy:    {results[f'horizon_{horizon}']['accuracy']:.4f}")
                print(f"F1 Macro:    {results[f'horizon_{horizon}']['f1_macro']:.4f}")
                print(
                    f"F1 Weighted: {results[f'horizon_{horizon}']['f1_weighted']:.4f}"
                )
                print("\nClassification Report:")
                print(
                    classification_report(
                        y_test_valid,
                        y_pred_valid,
                        target_names=los_names,
                        zero_division=0,
                    )
                )

        # Save results
        results_path = self.config.output_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def save_artifacts(self) -> None:
        """Save scaler, feature names, and metadata."""
        import joblib

        joblib.dump(self.scaler, self.config.output_dir / "scaler.joblib")

        with open(self.config.output_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        if self.metadata:
            with open(self.config.output_dir / "metadata.json", "w") as f:
                metadata_json = {}
                for k, v in self.metadata.items():
                    if isinstance(v, dict):
                        metadata_json[k] = {str(kk): vv for kk, vv in v.items()}
                    else:
                        metadata_json[k] = v
                json.dump(metadata_json, f, indent=2)

        # Save models
        for horizon, model in self.models.items():
            joblib.dump(model, self.config.output_dir / f"model_h{horizon}.joblib")

    def run(self) -> Dict[str, Any]:
        """Run full pipeline: train, evaluate, save artifacts."""
        train_results = self.train()
        test_results = self.evaluate()
        self.save_artifacts()
        print(f"\n✅ Training complete! Results saved to {self.config.output_dir}")
        return {"train": train_results, "test": test_results}
