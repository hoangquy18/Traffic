#!/usr/bin/env python3
"""
Script để train Autoformer model cho dự đoán giao thông.

Autoformer là mô hình state-of-the-art (NeurIPS 2021) sử dụng Auto-Correlation 
và Series Decomposition cho long sequence forecasting.

Usage:
    python train_autoformer.py --config traffic_trainer/configs/autoformer_config.yaml
    
    # Hoặc sử dụng config mặc định:
    python train_autoformer.py
"""

import argparse
from pathlib import Path

from traffic_trainer.trainers import AutoformerTrainer, AutoformerTrainingConfig
from traffic_trainer.trainers.base import load_yaml_config


def load_config(config_path: Path) -> AutoformerTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    autoformer_cfg = cfg.get("autoformer", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    seq_len = data.get("sequence_length", 96)
    
    return AutoformerTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/autoformer_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=seq_len,
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 256),
        dropout=model.get("dropout", 0.1),
        seq_len=autoformer_cfg.get("seq_len", seq_len),
        label_len=autoformer_cfg.get("label_len", seq_len // 2),
        out_len=autoformer_cfg.get("out_len", seq_len // 4),
        factor=autoformer_cfg.get("factor", 1),
        d_ff=autoformer_cfg.get("d_ff", 2048),
        n_heads=autoformer_cfg.get("n_heads", 8),
        e_layers=autoformer_cfg.get("e_layers", 2),
        d_layers=autoformer_cfg.get("d_layers", 1),
        moving_avg=autoformer_cfg.get("moving_avg", 25),
        activation=autoformer_cfg.get("activation", "gelu"),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        batch_size=optim.get("batch_size", 16),
        learning_rate=optim.get("learning_rate", 0.0001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 4),
        device=training.get("device", "cuda"),
        scheduler_type=training.get("scheduler_type", "plateau"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "autoformer-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Autoformer model for traffic prediction (SOTA - NeurIPS 2021)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Autoformer Features:
  • Auto-Correlation: Tìm periodic patterns tự động
  • Series Decomposition: Tách trend và seasonal components
  • O(L log L) complexity: Hiệu quả cho chuỗi dài
  • Interpretable: Có thể phân tích trend/seasonal riêng biệt
  • Đặc biệt tốt cho dữ liệu có tính chu kỳ (traffic, weather)

Examples:
  # Train với config mặc định
  python train_autoformer.py
  
  # Train với custom config
  python train_autoformer.py --config my_autoformer_config.yaml
  
  # So sánh với Informer
  python train_autoformer.py  # Autoformer
  python train_informer.py    # Informer
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("traffic_trainer/configs/autoformer_config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    print("="*80)
    print("🏆 AUTOFORMER TRAINING FOR TRAFFIC PREDICTION (SOTA - NeurIPS 2021)")
    print("="*80)
    
    args = parse_args()
    
    # Load config
    print(f"\n📋 Loading config from: {args.config}")
    config = load_config(args.config)
    
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📊 Dataset: {config.csv_path}")
    print(f"📏 Sequence Length: {config.seq_len}")
    print(f"🎯 Label Length: {config.label_len}")
    print(f"📈 Output Length: {config.out_len}")
    print(f"🧠 Encoder Layers: {config.e_layers}")
    print(f"🎭 Decoder Layers: {config.d_layers}")
    print(f"👁️  Attention Heads: {config.n_heads}")
    print(f"📊 Moving Average Window: {config.moving_avg}")
    print(f"⚡ Auto-Correlation Factor: {config.factor}")
    print(f"💾 Batch size: {config.batch_size}")
    print(f"📈 Epochs: {config.epochs}")
    print(f"🎯 Prediction horizons: {config.prediction_horizons}")
    
    # Create trainer
    print(f"\n🚀 Initializing Autoformer trainer...")
    trainer = AutoformerTrainer(config)
    
    # Run training
    print(f"\n🏃 Starting training...")
    print("⚠️  Note: Autoformer excels at capturing periodic patterns in traffic data!")
    results = trainer.run()
    
    # Print final results
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Best Validation F1: {results['train']['best_val_f1']:.4f}")
    print(f"📁 Results saved to: {config.output_dir}")
    print("="*80)
    print("\n💡 Autoformer Advantages:")
    print("  ✓ Auto-Correlation: Tự động phát hiện chu kỳ")
    print("  ✓ Series Decomposition: Tách trend và seasonal")
    print("  ✓ Interpretable: Dễ hiểu và debug")
    print("  ✓ Stable Training: Ít hyperparameters cần tune")
    print("  ✓ Perfect for Traffic: Dữ liệu giao thông có chu kỳ rõ ràng")
    print("="*80)


if __name__ == "__main__":
    main()
