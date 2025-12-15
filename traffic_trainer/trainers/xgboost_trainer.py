"""XGBoost trainer for traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from xgboost import XGBClassifier

from traffic_trainer.trainers.ml_base import MLBaseConfig, MLBaseTrainer, load_yaml_config


@dataclass
class XGBoostTrainingConfig(MLBaseConfig):
    """Configuration for XGBoost training."""
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0


def load_config(config_path: Path) -> XGBoostTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    xgboost_cfg = cfg.get("xgboost", {})
    
    return XGBoostTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/xgboost_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        random_state=data.get("random_state", 42),
        n_estimators=xgboost_cfg.get("n_estimators", 100),
        max_depth=xgboost_cfg.get("max_depth", 6),
        learning_rate=xgboost_cfg.get("learning_rate", 0.1),
        subsample=xgboost_cfg.get("subsample", 0.8),
        colsample_bytree=xgboost_cfg.get("colsample_bytree", 0.8),
        min_child_weight=xgboost_cfg.get("min_child_weight", 1),
        gamma=xgboost_cfg.get("gamma", 0.0),
        reg_alpha=xgboost_cfg.get("reg_alpha", 0.0),
        reg_lambda=xgboost_cfg.get("reg_lambda", 1.0),
    )


class XGBoostTrainer(MLBaseTrainer):
    """Trainer for XGBoost model."""
    
    config: XGBoostTrainingConfig
    
    def _create_model(self, horizon: int) -> XGBClassifier:
        """Create XGBoost model for a specific horizon."""
        return XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            objective="multi:softprob",
            num_class=self.num_classes,
            eval_metric="mlogloss",
            n_jobs=-1,
            verbosity=0,
        )
    
    def _train_model(self, model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost model."""
        # Filter out invalid samples (target < 0)
        valid_mask = y >= 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        model.fit(X_valid, y_valid)
    
    def _predict(self, model: XGBClassifier, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return model.predict(X)
    
    def _get_model_type(self) -> str:
        return "XGBoost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "xgboost_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = XGBoostTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

