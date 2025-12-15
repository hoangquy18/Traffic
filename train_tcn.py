#!/usr/bin/env python3
"""
Script để train TCN model cho dự đoán giao thông.

Usage:
    python train_tcn.py --config traffic_trainer/configs/tcn_config.yaml
    
    # Hoặc sử dụng config mặc định:
    python train_tcn.py
"""

import argparse
from pathlib import Path

from traffic_trainer.trainers import TCNTrainer, TCNTrainingConfig
from traffic_trainer.trainers.base import load_yaml_config


def load_config(config_path: Path) -> TCNTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    tcn_cfg = cfg.get("tcn", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    return TCNTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/tcn_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 12),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.2),
        model_type=tcn_cfg.get("model_type", "standard"),
        num_channels=tcn_cfg.get("num_channels"),
        kernel_size=tcn_cfg.get("kernel_size", 3),
        scales=tcn_cfg.get("scales"),
        use_temporal_attention=tcn_cfg.get("use_temporal_attention", True),
        use_spatial_attention=tcn_cfg.get("use_spatial_attention", False),
        num_attention_heads=tcn_cfg.get("num_attention_heads", 4),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        batch_size=optim.get("batch_size", 32),
        learning_rate=optim.get("learning_rate", 0.001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 4),
        device=training.get("device", "cuda"),
        scheduler_type=training.get("scheduler_type", "plateau"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "tcn-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TCN model for traffic prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train với config mặc định
  python train_tcn.py
  
  # Train với custom config
  python train_tcn.py --config my_config.yaml
  
  # Train với multiscale TCN
  python train_tcn.py --config traffic_trainer/configs/tcn_multiscale_config.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("traffic_trainer/configs/tcn_config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    print("="*80)
    print("TCN TRAINING FOR TRAFFIC PREDICTION")
    print("="*80)
    
    args = parse_args()
    
    # Load config
    print(f"\n📋 Loading config from: {args.config}")
    config = load_config(args.config)
    
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📊 Dataset: {config.csv_path}")
    print(f"🔧 TCN Type: {config.model_type}")
    print(f"💾 Batch size: {config.batch_size}")
    print(f"📈 Epochs: {config.epochs}")
    print(f"🎯 Prediction horizons: {config.prediction_horizons}")
    
    # Create trainer
    print(f"\n🚀 Initializing TCN trainer...")
    trainer = TCNTrainer(config)
    
    # Run training
    print(f"\n🏃 Starting training...")
    results = trainer.run()
    
    # Print final results
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Validation F1: {results['train']['best_val_f1']:.4f}")
    print(f"Results saved to: {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

