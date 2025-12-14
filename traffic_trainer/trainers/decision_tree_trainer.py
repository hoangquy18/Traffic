"""Decision Tree trainer for traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from sklearn.tree import DecisionTreeClassifier

from traffic_trainer.trainers.ml_base import MLBaseConfig, MLBaseTrainer, load_yaml_config


@dataclass
class DecisionTreeTrainingConfig(MLBaseConfig):
    """Configuration for Decision Tree training."""
    
    # Decision Tree parameters
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    criterion: str = "gini"


def load_config(config_path: Path) -> DecisionTreeTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    dt_cfg = cfg.get("decision_tree", {})
    
    return DecisionTreeTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/decision_tree_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        random_state=data.get("random_state", 42),
        max_depth=dt_cfg.get("max_depth"),
        min_samples_split=dt_cfg.get("min_samples_split", 2),
        min_samples_leaf=dt_cfg.get("min_samples_leaf", 1),
        max_features=dt_cfg.get("max_features", "sqrt"),
        criterion=dt_cfg.get("criterion", "gini"),
    )


class DecisionTreeTrainer(MLBaseTrainer):
    """Trainer for Decision Tree model."""
    
    config: DecisionTreeTrainingConfig
    
    def _create_model(self, horizon: int) -> DecisionTreeClassifier:
        """Create Decision Tree model for a specific horizon."""
        return DecisionTreeClassifier(
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            criterion=self.config.criterion,
            random_state=self.config.random_state,
        )
    
    def _train_model(self, model: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray) -> None:
        """Train Decision Tree model."""
        # Filter out invalid samples (target < 0)
        valid_mask = y >= 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        model.fit(X_valid, y_valid)
    
    def _predict(self, model: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return model.predict(X)
    
    def _get_model_type(self) -> str:
        return "DecisionTree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Decision Tree for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "decision_tree_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = DecisionTreeTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

