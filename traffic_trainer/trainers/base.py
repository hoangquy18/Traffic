"""Abstract base trainer for traffic prediction models."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from traffic_trainer.data import LOS_LEVELS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class BaseConfig:
    """Base configuration shared by all trainers."""
    
    # Paths
    csv_path: Path = Path("data.csv")
    output_dir: Path = Path("experiments/run01")
    
    # Data
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    prediction_horizons: List[int] = field(default_factory=lambda: [1])
    sequence_length: int = 8
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    resample_rule: Optional[str] = "1H"
    
    # Model
    hidden_dim: int = 128
    dropout: float = 0.3
    
    # Optimization
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    
    # Training
    epochs: int = 50
    num_workers: int = 0
    device: str = "cuda"
    
    # Scheduler
    scheduler_type: str = "plateau"  # "plateau" or "cosine"
    
    # Logging
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "run"
    wandb_mode: str = "disabled"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.0
    
    # Checkpointing
    checkpoint_every: int = 5
    save_optimizer_state: bool = True


class BaseTrainer(ABC):
    """
    Abstract base trainer for traffic prediction models.
    
    Subclasses must implement:
    - _load_data(): Load and return datasets
    - _create_model(): Create and return the model
    - _forward_batch(): Run forward pass on a batch
    - _get_model_type(): Return model type string for logging
    """
    
    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        
        # Load data (implemented by subclass)
        self._load_data()
        
        # Derived attributes
        self.prediction_horizons = sorted({int(h) for h in config.prediction_horizons})
        self.num_horizons = len(self.prediction_horizons)
        self.num_classes = len(LOS_LEVELS)
        
        # Create data loaders
        self._create_dataloaders()
        
        # Create model (implemented by subclass)
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Loss function (can be overridden)
        self.criterion = self._create_criterion()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        self._init_wandb()
        
        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Print model info
        self._print_model_info()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load datasets. Must set self.train_dataset, self.val_dataset, self.test_dataset, 
        self.feature_names, self.scaler, and self.metadata."""
        pass
    
    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the model."""
        pass
    
    @abstractmethod
    def _forward_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass on a batch.
        
        Returns:
            logits: Model output [batch, *, num_horizons, num_classes]
            targets: Ground truth [batch, *, num_horizons]
            mask: Valid samples mask [batch, *] or None
        """
        pass
    
    @abstractmethod
    def _get_model_type(self) -> str:
        """Return model type string for logging."""
        pass
    
    def _get_collate_fn(self) -> Optional[callable]:
        """Return collate function for DataLoader. Override if needed."""
        return None
    
    def _create_dataloaders(self) -> None:
        """Create train/val/test data loaders."""
        collate_fn = self._get_collate_fn()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function. Override for custom losses."""
        return nn.CrossEntropyLoss(ignore_index=-1)
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            return ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
            )
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        self.use_wandb = WANDB_AVAILABLE and self.config.wandb_mode != "disabled"
        if self.use_wandb and self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                mode=self.config.wandb_mode,
                config=self._get_wandb_config(),
            )
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Get config dict for W&B. Override to add model-specific info."""
        return {
            "model_type": self._get_model_type(),
            "num_features": len(self.feature_names),
            **{k: str(v) if isinstance(v, Path) else v 
               for k, v in vars(self.config).items()},
        }
    
    def _print_model_info(self) -> None:
        """Print model information."""
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self._get_model_type()}")
        print(f"Parameters: {num_params:,} ({trainable_params:,} trainable)")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def _compute_metrics(
        self,
        all_preds: np.ndarray,
        all_targets: np.ndarray,
        all_masks: np.ndarray,
        all_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute metrics from predictions."""
        metrics = {}
        
        for h_idx, horizon in enumerate(self.prediction_horizons):
            # Handle different tensor shapes
            if all_preds.ndim == 3:  # [batch, nodes, horizons]
                preds_h = all_preds[:, :, h_idx]
                targets_h = all_targets[:, :, h_idx]
                probs_h = all_probs[:, :, h_idx, :] if all_probs is not None else None
            else:  # [batch, horizons]
                preds_h = all_preds[:, h_idx]
                targets_h = all_targets[:, h_idx]
                probs_h = all_probs[:, h_idx, :] if all_probs is not None else None
            
            # Apply mask
            if all_masks.ndim == 2 and all_preds.ndim == 3:  # Graph data
                valid_mask = all_masks & (targets_h >= 0)
            else:
                valid_mask = targets_h >= 0
            
            preds_valid = preds_h[valid_mask]
            targets_valid = targets_h[valid_mask]
            probs_valid = probs_h[valid_mask] if probs_h is not None else None
            
            if len(targets_valid) > 0:
                metrics[f"accuracy_h{horizon}"] = float(accuracy_score(targets_valid, preds_valid))
                metrics[f"f1_macro_h{horizon}"] = float(f1_score(
                    targets_valid, preds_valid, average="macro", zero_division=0
                ))
                metrics[f"f1_weighted_h{horizon}"] = float(f1_score(
                    targets_valid, preds_valid, average="weighted", zero_division=0
                ))
                metrics[f"precision_h{horizon}"] = float(precision_score(
                    targets_valid, preds_valid, average="macro", zero_division=0
                ))
                metrics[f"recall_h{horizon}"] = float(recall_score(
                    targets_valid, preds_valid, average="macro", zero_division=0
                ))
                
                # Per-class F1
                f1_per_class = f1_score(targets_valid, preds_valid, average=None, zero_division=0)
                for cls_idx, cls_name in enumerate(LOS_LEVELS.keys()):
                    if cls_idx < len(f1_per_class):
                        metrics[f"f1_{cls_name}_h{horizon}"] = float(f1_per_class[cls_idx])
        
        # Primary metrics (first horizon)
        primary_horizon = self.prediction_horizons[0]
        metrics["f1_macro"] = metrics.get(f"f1_macro_h{primary_horizon}", 0.0)
        metrics["f1_weighted"] = metrics.get(f"f1_weighted_h{primary_horizon}", 0.0)
        metrics["accuracy"] = metrics.get(f"accuracy_h{primary_horizon}", 0.0)
        
        return metrics
    
    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Dict[str, float]:
        """Run one epoch of training or validation."""
        self.model.train() if train else self.model.eval()
        
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        
        context = torch.no_grad() if not train else torch.enable_grad()
        
        with context:
            for batch in loader:
                # Forward pass (implemented by subclass)
                logits, targets, mask = self._forward_batch(batch)
                
                # Flatten for loss computation
                num_classes = logits.shape[-1]
                logits_flat = logits.view(-1, num_classes)
                targets_flat = targets.view(-1)
                
                # Apply mask if present
                if mask is not None:
                    num_horizons = logits.shape[-2]
                    mask_expanded = mask.unsqueeze(-1).expand(*mask.shape, num_horizons)
                    mask_flat = mask_expanded.reshape(-1)
                    targets_masked = targets_flat.clone()
                    targets_masked[~mask_flat] = -1
                else:
                    targets_masked = targets_flat
                
                loss = self.criterion(logits_flat, targets_masked)
                
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config.gradient_clip_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm,
                        )
                    self.optimizer.step()
                    
                    if self.config.scheduler_type == "cosine":
                        self.scheduler.step()
                
                total_loss += loss.item()
                
                # Collect predictions
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())
                if mask is not None:
                    all_masks.append(mask.cpu().numpy())
                else:
                    all_masks.append(np.ones(preds.shape[:-1], dtype=bool))
        
        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_targets, all_masks, all_probs)
        metrics["loss"] = total_loss / len(loader)
        
        return metrics
    
    def _save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_type": self._get_model_type(),
            "model_state": self.model.state_dict(),
            "best_val_f1": self.best_val_f1,
            "config": {k: str(v) if isinstance(v, Path) else v 
                      for k, v in vars(self.config).items()},
            "feature_names": self.feature_names,
            "metadata": getattr(self, "metadata", {}),
        }
        if self.config.save_optimizer_state:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        if is_best:
            torch.save(checkpoint, self.config.output_dir / "best_model.pt")
    
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log epoch metrics."""
        print(
            f"Epoch {self.current_epoch:3d}/{self.config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
        if self.use_wandb:
            log_dict = {
                "epoch": self.current_epoch,
                "train/loss": train_metrics["loss"],
                "train/f1_macro": train_metrics.get("f1_macro", 0),
                "train/accuracy": train_metrics.get("accuracy", 0),
                "val/loss": val_metrics["loss"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/f1_weighted": val_metrics.get("f1_weighted", 0),
                "val/accuracy": val_metrics["accuracy"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            # Per-class F1
            for cls_name in LOS_LEVELS.keys():
                key = f"f1_{cls_name}_h{self.prediction_horizons[0]}"
                if key in val_metrics:
                    log_dict[f"val/f1_{cls_name}"] = val_metrics[key]
            wandb.log(log_dict)
    
    def train(self) -> Dict[str, Any]:
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"Training {self._get_model_type()} for {self.config.epochs} epochs")
        print(f"{'='*60}\n")
        
        history = {"train": [], "val": []}
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch + 1
            
            # Training
            train_metrics = self._run_epoch(self.train_loader, train=True)
            history["train"].append(train_metrics)
            
            # Validation
            val_metrics = self._run_epoch(self.val_loader, train=False)
            history["val"].append(val_metrics)
            
            # Update scheduler (plateau only)
            if self.config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics["f1_macro"])
            
            # Logging
            self._log_epoch(train_metrics, val_metrics)
            
            # Checkpointing
            is_best = val_metrics["f1_macro"] > self.best_val_f1 + self.config.early_stopping_delta
            if is_best:
                self.best_val_f1 = val_metrics["f1_macro"]
                self.patience_counter = 0
                self._save_checkpoint(self.config.output_dir / "best_model.pt", is_best=True)
                print(f"  ★ New best model! F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
            
            if self.current_epoch % self.config.checkpoint_every == 0:
                self._save_checkpoint(
                    self.config.output_dir / f"checkpoint_epoch{self.current_epoch}.pt"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n⚠ Early stopping at epoch {self.current_epoch}")
                break
        
        # Save history
        with open(self.config.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        return {"history": history, "best_val_f1": self.best_val_f1}
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set."""
        print(f"\n{'='*60}")
        print("Evaluating on test set...")
        print(f"{'='*60}")
        
        # Load best model
        best_path = self.config.output_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                logits, targets, mask = self._forward_batch(batch)
                probs = torch.softmax(logits, dim=-1)
                
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                if mask is not None:
                    all_masks.append(mask.cpu().numpy())
                else:
                    all_masks.append(np.ones(targets.shape[:-1], dtype=bool))
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        results = {"model_type": self._get_model_type()}
        los_names = list(LOS_LEVELS.keys())
        
        for h_idx, horizon in enumerate(self.prediction_horizons):
            # Handle different tensor shapes
            if all_preds.ndim == 3:
                preds_h = all_preds[:, :, h_idx]
                targets_h = all_targets[:, :, h_idx]
                probs_h = all_probs[:, :, h_idx, :]
                valid_mask = all_masks & (targets_h >= 0)
            else:
                preds_h = all_preds[:, h_idx]
                targets_h = all_targets[:, h_idx]
                probs_h = all_probs[:, h_idx, :]
                valid_mask = targets_h >= 0
            
            preds_valid = preds_h[valid_mask]
            targets_valid = targets_h[valid_mask]
            probs_valid = probs_h[valid_mask]
            
            if len(targets_valid) > 0:
                results[f"horizon_{horizon}"] = {
                    "accuracy": float(accuracy_score(targets_valid, preds_valid)),
                    "f1_macro": float(f1_score(targets_valid, preds_valid, average="macro", zero_division=0)),
                    "f1_weighted": float(f1_score(targets_valid, preds_valid, average="weighted", zero_division=0)),
                    "precision_macro": float(precision_score(targets_valid, preds_valid, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(targets_valid, preds_valid, average="macro", zero_division=0)),
                }
                
                # Per-class F1
                f1_per_class = f1_score(targets_valid, preds_valid, average=None, zero_division=0)
                for cls_idx, cls_name in enumerate(los_names):
                    if cls_idx < len(f1_per_class):
                        results[f"horizon_{horizon}"][f"f1_{cls_name}"] = float(f1_per_class[cls_idx])
                
                # ROC-AUC
                try:
                    results[f"horizon_{horizon}"]["roc_auc_ovr"] = float(
                        roc_auc_score(targets_valid, probs_valid, multi_class="ovr", average="macro")
                    )
                except ValueError:
                    pass
                
                # Print results
                print(f"\n=== Test Results (Horizon +{horizon}) ===")
                print(f"Accuracy:    {results[f'horizon_{horizon}']['accuracy']:.4f}")
                print(f"F1 Macro:    {results[f'horizon_{horizon}']['f1_macro']:.4f}")
                print(f"F1 Weighted: {results[f'horizon_{horizon}']['f1_weighted']:.4f}")
                print("\nClassification Report:")
                print(classification_report(targets_valid, preds_valid, target_names=los_names, zero_division=0))
        
        # Save results
        results_path = self.config.output_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        if self.use_wandb:
            wandb.log({"test": results})
        
        return results
    
    def save_artifacts(self) -> None:
        """Save scaler, feature names, and metadata."""
        import joblib
        
        joblib.dump(self.scaler, self.config.output_dir / "scaler.joblib")
        
        with open(self.config.output_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        
        if hasattr(self, "metadata") and self.metadata:
            with open(self.config.output_dir / "metadata.json", "w") as f:
                # Convert keys to strings for JSON
                metadata_json = {}
                for k, v in self.metadata.items():
                    if isinstance(v, dict):
                        metadata_json[k] = {str(kk): vv for kk, vv in v.items()}
                    else:
                        metadata_json[k] = v
                json.dump(metadata_json, f, indent=2)
    
    def finish(self) -> None:
        """Cleanup."""
        if self.use_wandb:
            wandb.finish()
    
    def run(self) -> Dict[str, Any]:
        """Run full training pipeline: train, evaluate, save artifacts."""
        train_results = self.train()
        test_results = self.evaluate()
        self.save_artifacts()
        self.finish()
        print(f"\n✅ Training complete! Results saved to {self.config.output_dir}")
        return {"train": train_results, "test": test_results}


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and flatten YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

