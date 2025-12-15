#!/usr/bin/env python3
"""
Script để train PatchTST model cho dự đoán giao thông.

PatchTST là mô hình SOTA (ICLR 2023) sử dụng patching như Vision Transformer.
Đạt SOTA performance với efficiency cao hơn vanilla Transformers.

Usage:
    python train_patchtst.py --config traffic_trainer/configs/patchtst_config.yaml
    
    # Hoặc sử dụng config mặc định:
    python train_patchtst.py
"""

import argparse
from pathlib import Path

from traffic_trainer.trainers.patchtst_trainer import PatchTSTTrainer, PatchTSTTrainingConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PatchTST model for traffic prediction (SOTA - ICLR 2023)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PatchTST Features:
  • Patching: Chia time series thành patches (như ViT)
  • Channel Independence: Xử lý mỗi feature riêng biệt
  • Efficiency: Giảm sequence length (96 → 6 patches)
  • SOTA Performance: Top trên nhiều benchmarks
  • Nhẹ hơn Autoformer/Informer

Examples:
  # Train với config mặc định
  python train_patchtst.py
  
  # Train với custom config
  python train_patchtst.py --config my_patchtst_config.yaml
  
  # So sánh với models khác
  python train_dlinear.py     # DLinear: 15-30 min, 0.5-1 GB
  python train_patchtst.py    # PatchTST: 1-1.5 hours, 1-2 GB
  python train_autoformer.py  # Autoformer: 2-3 hours, 2-3 GB
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("traffic_trainer/configs/patchtst_config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    print("="*80)
    print("🏆 PATCHTST TRAINING FOR TRAFFIC PREDICTION (SOTA - ICLR 2023)")
    print("="*80)
    print("💡 PatchTST: A Time Series is Worth 64 Words!")
    print("   Patching + Channel Independence = SOTA! 🎯")
    print("="*80)
    
    args = parse_args()
    
    # Load config
    print(f"\n📋 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Calculate num_patches
    num_patches = (config.seq_len - config.patch_len) // config.stride + 1
    
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📊 Dataset: {config.csv_path}")
    print(f"📏 Sequence Length: {config.seq_len}")
    print(f"🔲 Patch Length: {config.patch_len}")
    print(f"👣 Stride: {config.stride}")
    print(f"📦 Number of Patches: {num_patches} (reduced from {config.seq_len}!)")
    print(f"🧠 Transformer Layers: {config.n_layers}")
    print(f"👁️  Attention Heads: {config.n_heads}")
    print(f"📊 Feed-Forward Dim: {config.d_ff}")
    print(f"🔀 Channel Independence: {config.channel_independence}")
    print(f"💾 Batch size: {config.batch_size}")
    print(f"📈 Epochs: {config.epochs}")
    print(f"🎯 Prediction horizons: {config.prediction_horizons}")
    
    # Create trainer
    print(f"\n🚀 Initializing PatchTST trainer...")
    trainer = PatchTSTTrainer(config)
    
    # Run training
    print(f"\n🏃 Starting training...")
    print("⚡ PatchTST is efficient - expect 1-1.5 hours!")
    print("🎯 Patching reduces sequence length → faster training!")
    results = trainer.run()
    
    # Print final results
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Best Validation F1: {results['train']['best_val_f1']:.4f}")
    print(f"📁 Results saved to: {config.output_dir}")
    print("="*80)
    print("\n💡 PatchTST Advantages:")
    print("  ✓ SOTA Performance: Top trên nhiều benchmarks")
    print("  ✓ Patching: Giảm sequence length hiệu quả")
    print("  ✓ Channel Independence: Đơn giản hơn, hiệu quả hơn")
    print("  ✓ Efficient: Nhẹ hơn Autoformer/Informer")
    print("  ✓ Modern: ICLR 2023 - mới nhất!")
    print("="*80)
    print("\n🎯 PatchTST Philosophy:")
    print("  'A Time Series is Worth 64 Words'")
    print("  Patching + Channel Independence → SOTA!")
    print("="*80)


if __name__ == "__main__":
    main()


