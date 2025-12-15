#!/usr/bin/env python3
"""
Train PatchTST với Multi-GPU support (DataParallel).

Usage:
    # Tự động dùng tất cả GPUs
    python train_patchtst_multigpu.py
    
    # Chỉ định GPUs cụ thể
    CUDA_VISIBLE_DEVICES=0,1 python train_patchtst_multigpu.py
"""

import torch
import torch.nn as nn
from pathlib import Path

from traffic_trainer.trainers.patchtst_trainer import (
    PatchTSTTrainer,
    PatchTSTTrainingConfig,
    load_config,
    parse_args
)


def main():
    """Main training function with multi-GPU support."""
    print("="*80)
    print("🏆 PATCHTST MULTI-GPU TRAINING")
    print("="*80)
    
    args = parse_args()
    
    # Check GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n🔍 Detected {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load config
    print(f"\n📋 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Create trainer
    print(f"\n🚀 Initializing PatchTST trainer...")
    trainer = PatchTSTTrainer(config)
    
    # Wrap with DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"\n🚀 Using DataParallel with {num_gpus} GPUs!")
        print(f"   Effective batch size: {config.batch_size} ({config.batch_size//num_gpus} per GPU)")
        trainer.model = nn.DataParallel(trainer.model)
    else:
        print(f"\n⚠️  Using single GPU")
    
    # Run training
    print(f"\n🏃 Starting training...")
    results = trainer.run()
    
    # Print final results
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Best Validation F1: {results['train']['best_val_f1']:.4f}")
    print(f"📁 Results saved to: {config.output_dir}")
    if num_gpus > 1:
        print(f"🚀 Trained on {num_gpus} GPUs with DataParallel")
    print("="*80)


if __name__ == "__main__":
    main()


