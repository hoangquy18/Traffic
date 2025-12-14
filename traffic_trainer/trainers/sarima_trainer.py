"""SARIMA trainer for traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from traffic_trainer.data import LOS_LEVELS, load_dataset


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and flatten YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


@dataclass
class SARIMATrainingConfig:
    """Configuration for SARIMA training."""
    
    # Paths
    csv_path: Path = Path("data.csv")
    output_dir: Path = Path("experiments/sarima_run01")
    
    # Data
    numerical_features: List[str] = None
    categorical_features: List[str] = None
    prediction_horizons: List[int] = None
    sequence_length: int = 8
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    resample_rule: Optional[str] = "1H"
    
    # SARIMA parameters
    order: Tuple[int, int, int] = (1, 1, 1)  # (p, d, q)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)  # (P, D, Q, s) - s=24 for hourly data
    max_iter: int = 50


def load_config(config_path: Path) -> SARIMATrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    sarima_cfg = cfg.get("sarima", {})
    
    order_tuple = tuple(sarima_cfg.get("order", [1, 1, 1]))
    if len(order_tuple) != 3:
        order_tuple = (1, 1, 1)
    
    seasonal_order_tuple = tuple(sarima_cfg.get("seasonal_order", [1, 1, 1, 24]))
    if len(seasonal_order_tuple) != 4:
        seasonal_order_tuple = (1, 1, 1, 24)
    
    return SARIMATrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/sarima_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        order=order_tuple,
        seasonal_order=seasonal_order_tuple,
        max_iter=sarima_cfg.get("max_iter", 50),
    )


class SARIMATrainer:
    """Trainer for SARIMA model."""
    
    def __init__(self, config: SARIMATrainingConfig) -> None:
        self.config = config
        self.prediction_horizons = sorted({int(h) for h in config.prediction_horizons})
        self.num_horizons = len(self.prediction_horizons)
        self.num_classes = len(LOS_LEVELS)
        
        # Load data
        self._load_data()
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.models = {}  # One model per segment and horizon
        self.history = {"train": [], "val": []}
    
    def _load_data(self) -> None:
        """Load and prepare data for SARIMA."""
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
        
        # Convert to time series format (per segment)
        self.train_data = self._dataset_to_timeseries(train_ds)
        self.val_data = self._dataset_to_timeseries(val_ds)
        self.test_data = self._dataset_to_timeseries(test_ds)
        
        print(f"Segments: {len(self.train_data)}")
    
    def _dataset_to_timeseries(self, dataset) -> Dict[int, Dict[str, np.ndarray]]:
        """Convert dataset to time series format per segment."""
        segment_data = {}
        
        for i in range(len(dataset)):
            sample, targets = dataset[i]
            segment_id = sample.get("segment_ids", None)
            if segment_id is None:
                segment_id = 0
            
            segment_id = int(segment_id.item()) if hasattr(segment_id, 'item') else int(segment_id)
            
            if segment_id not in segment_data:
                segment_data[segment_id] = {
                    "features": [],
                    "targets": [],
                }
            
            # Use the last timestep
            features = sample["features"][-1].numpy()
            segment_data[segment_id]["features"].append(features)
            segment_data[segment_id]["targets"].append(targets.numpy())
        
        # Convert to arrays
        for seg_id in segment_data:
            segment_data[seg_id]["features"] = np.array(segment_data[seg_id]["features"])
            segment_data[seg_id]["targets"] = np.array(segment_data[seg_id]["targets"])
        
        return segment_data
    
    def _create_sarima_model(self, ts: np.ndarray) -> Optional[SARIMAX]:
        """Create and fit SARIMA model for a time series."""
        try:
            # Use the mean of features as the time series (or first feature)
            if ts.ndim > 1:
                ts = ts.mean(axis=1) if ts.shape[1] > 0 else ts[:, 0]
            
            # Ensure enough data points
            min_length = max(
                self.config.order[0] + self.config.order[2],
                self.config.seasonal_order[0] + self.config.seasonal_order[2],
                self.config.seasonal_order[3] * 2
            )
            if len(ts) < min_length:
                return None
            
            model = SARIMAX(
                ts,
                order=self.config.order,
                seasonal_order=self.config.seasonal_order,
            )
            fitted_model = model.fit(maxiter=self.config.max_iter, disp=0)
            return fitted_model
        except Exception as e:
            print(f"Warning: SARIMA fitting failed: {e}")
            return None
    
    def _predict_sarima(self, model: SARIMAX, steps: int) -> np.ndarray:
        """Predict using SARIMA model."""
        try:
            forecast = model.forecast(steps=steps)
            # Convert to class predictions (simple mapping)
            forecast_int = np.clip(np.round(forecast).astype(int), 0, self.num_classes - 1)
            return forecast_int
        except Exception:
            # Return default prediction
            return np.array([0] * steps)
    
    def train(self) -> Dict[str, Any]:
        """Train SARIMA models."""
        print(f"\n{'='*60}")
        print(f"Training SARIMA for horizons {self.prediction_horizons}")
        print(f"{'='*60}\n")
        
        print("Note: SARIMA works best with univariate time series.")
        print("Using aggregated approach across all segments.")
        
        # Aggregate all segments into one time series
        all_train_features = []
        all_train_targets = []
        
        for seg_id, data in self.train_data.items():
            all_train_features.append(data["features"])
            all_train_targets.append(data["targets"])
        
        if not all_train_features:
            print("No training data available!")
            return {"history": self.history}
        
        # Concatenate and use mean across segments
        train_features = np.concatenate(all_train_features, axis=0)
        train_targets = np.concatenate(all_train_targets, axis=0)
        
        # Use mean of features as time series
        train_ts = train_features.mean(axis=1)
        
        # Train one model per horizon
        for horizon_idx, horizon in enumerate(self.prediction_horizons):
            print(f"\n--- Training SARIMA for horizon +{horizon} ---")
            
            # Create target time series (LOS values)
            target_ts = train_targets[:, horizon_idx]
            
            # Filter valid targets
            valid_mask = target_ts >= 0
            if valid_mask.sum() < self.config.seasonal_order[3] * 2:
                print(f"Not enough valid data for horizon {horizon} (need at least {self.config.seasonal_order[3] * 2} samples)")
                continue
            
            train_ts_valid = train_ts[valid_mask]
            target_ts_valid = target_ts[valid_mask]
            
            # Fit SARIMA on target time series
            model = self._create_sarima_model(target_ts_valid)
            if model is not None:
                self.models[horizon] = model
                print(f"SARIMA model fitted for horizon {horizon}")
            else:
                print(f"Failed to fit SARIMA for horizon {horizon}")
        
        return {"history": self.history}
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set."""
        print(f"\n{'='*60}")
        print("Evaluating SARIMA on test set...")
        print(f"{'='*60}")
        
        results = {"model_type": "SARIMA"}
        los_names = list(LOS_LEVELS.keys())
        
        # Aggregate test data
        all_test_features = []
        all_test_targets = []
        
        for seg_id, data in self.test_data.items():
            all_test_features.append(data["features"])
            all_test_targets.append(data["targets"])
        
        if not all_test_features:
            print("No test data available!")
            return results
        
        test_features = np.concatenate(all_test_features, axis=0)
        test_targets = np.concatenate(all_test_targets, axis=0)
        
        for horizon_idx, horizon in enumerate(self.prediction_horizons):
            if horizon not in self.models:
                continue
            
            model = self.models[horizon]
            y_test_h = test_targets[:, horizon_idx]
            
            # Predict (SARIMA predicts future values)
            y_pred_test = []
            for i in range(len(y_test_h)):
                if y_test_h[i] >= 0:
                    pred = self._predict_sarima(model, steps=1)
                    y_pred_test.append(pred[0] if len(pred) > 0 else 0)
                else:
                    y_pred_test.append(-1)
            
            y_pred_test = np.array(y_pred_test)
            
            # Filter valid samples
            valid_mask = y_test_h >= 0
            y_test_valid = y_test_h[valid_mask]
            y_pred_valid = y_pred_test[valid_mask]
            
            if len(y_test_valid) > 0:
                results[f"horizon_{horizon}"] = {
                    "accuracy": float(accuracy_score(y_test_valid, y_pred_valid)),
                    "f1_macro": float(
                        f1_score(y_test_valid, y_pred_valid, average="macro", zero_division=0)
                    ),
                    "f1_weighted": float(
                        f1_score(y_test_valid, y_pred_valid, average="weighted", zero_division=0)
                    ),
                    "precision_macro": float(
                        precision_score(y_test_valid, y_pred_valid, average="macro", zero_division=0)
                    ),
                    "recall_macro": float(
                        recall_score(y_test_valid, y_pred_valid, average="macro", zero_division=0)
                    ),
                }
                
                print(f"\n=== Test Results (Horizon +{horizon}) ===")
                print(f"Accuracy:    {results[f'horizon_{horizon}']['accuracy']:.4f}")
                print(f"F1 Macro:    {results[f'horizon_{horizon}']['f1_macro']:.4f}")
                print(f"F1 Weighted: {results[f'horizon_{horizon}']['f1_weighted']:.4f}")
        
        # Save results
        results_path = self.config.output_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            import json
            json.dump(results, f, indent=2)
        
        return results
    
    def save_artifacts(self) -> None:
        """Save artifacts."""
        import joblib
        
        joblib.dump(self.scaler, self.config.output_dir / "scaler.joblib")
        
        with open(self.config.output_dir / "feature_names.json", "w") as f:
            import json
            json.dump(self.feature_names, f)
        
        # Save models
        for horizon, model in self.models.items():
            joblib.dump(model, self.config.output_dir / f"model_h{horizon}.joblib")
    
    def run(self) -> Dict[str, Any]:
        """Run full pipeline."""
        train_results = self.train()
        test_results = self.evaluate()
        self.save_artifacts()
        print(f"\n✅ Training complete! Results saved to {self.config.output_dir}")
        return {"train": train_results, "test": test_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SARIMA for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "sarima_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = SARIMATrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

