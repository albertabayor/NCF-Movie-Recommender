#!/usr/bin/env python3
"""
Memory-efficient local training script for NCF models.

Designed for systems with limited GPU VRAM (4GB) and system RAM.
Uses smaller batch size and gradient accumulation to maintain
training quality while staying within memory limits.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import torch

from src.config import config
from src.models.neumf import NeuMF
from src.train import train_model
from src.negative_sampling import build_user_history


def check_gpu():
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detected: {gpu_name}")
        print(f"  VRAM: {gpu_memory_gb:.1f} GB")

        # Recommend batch size based on VRAM
        if gpu_memory_gb < 6:
            recommended_batch = 64
            print(f"  Recommended batch size: {recommended_batch} (4GB GPU)")
        elif gpu_memory_gb < 12:
            recommended_batch = 128
            print(f"  Recommended batch size: {recommended_batch} (8GB GPU)")
        else:
            recommended_batch = 256
            print(f"  Recommended batch size: {recommended_batch} (12GB+ GPU)")
        return device, recommended_batch
    else:
        print("⚠️  No GPU detected. Training on CPU will be very slow.")
        return "cpu", 32


def print_memory_usage():
    """Print current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()

        print(f"\nMemory Usage:")
        print(f"  Process: {mem_info.rss / 1e9:.2f} GB")
        print(f"  System:  {(system_mem.total - system_mem.available) / 1e9:.2f} GB / {system_mem.total / 1e9:.2f} GB")
        print(f"  System %: {system_mem.percent:.1f}%")
    except ImportError:
        print("\nMemory usage monitoring unavailable (psutil not installed)")
        print("  Install with: pip install psutil")


def main():
    """Run training with memory-efficient settings."""
    print("=" * 60)
    print("NCF MOVIE RECOMMENDER - LOCAL TRAINING")
    print("=" * 60)

    # Check GPU and get recommended batch size
    device, recommended_batch = check_gpu()

    # Load preprocessed data
    print("\n[1/5] Loading data...")
    train_df = pd.read_pickle(config.paths.train_path)
    val_df = pd.read_pickle(config.paths.val_path)
    test_df = pd.read_pickle(config.paths.test_path)

    with open(config.paths.mappings_path, "rb") as f:
        mappings = pickle.load(f)

    num_users = mappings["num_users"]
    num_items = mappings["num_items"]

    train_users = train_df["userId"].values
    train_items = train_df["movieId"].values
    val_users = val_df["userId"].values
    val_items = val_df["movieId"].values

    # Build user history
    print("\n[2/5] Building user history for negative sampling...")
    user_history = build_user_history(train_users, train_items)

    # Print memory usage
    print_memory_usage()

    # Create model
    print("\n[3/5] Creating NeuMF model...")
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    print(f"  Estimated model size: {param_count * 4 / 1e6:.1f} MB")

    # Memory-efficient training settings
    print("\n[4/5] Configuring training for 4GB GPU...")
    batch_size = 128  # Larger batch for 4GB VRAM (try 256 if OOM)
    learning_rate = 1e-3
    num_epochs = 30
    num_negatives = 4
    num_workers = 4  # Parallel data loading

    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Negatives per positive: {num_negatives}")
    print(f"  Data loading workers: {num_workers}")

    # Estimate training time
    if device == "cuda":
        # With num_workers=4, expect ~30-50 it/s on RTX 3050
        it_per_sec = 40
        total_steps = len(train_users) // batch_size
        time_per_epoch = total_steps / it_per_sec / 60  # minutes
        print(f"\n  Estimated time per epoch: {time_per_epoch:.0f} minutes")
        print(f"  Estimated total time: {time_per_epoch * 5:.0f} minutes (assuming ~5 epochs)")

    # Train
    print("\n[5/5] Starting training...")
    print("  Models will save to: experiments/trained_models/")
    print("  Best model saves automatically (based on HR@10)")
    print("\n" + "=" * 60)

    history = train_model(
        model=model,
        train_users=train_users,
        train_items=train_items,
        val_data={
            "users": val_users,
            "items": val_items,
        },
        num_items=num_items,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        num_negatives=num_negatives,
        device=device,
        num_workers=num_workers,
        save_dir=str(config.paths.TRAINED_MODELS_DIR),
        early_stopping_patience=5,
        early_stopping_metric="hr@10",
        lr_scheduler_patience=3,
        lr_scheduler_factor=0.5,
        log_dir=str(config.paths.TENSORBOARD_LOG_DIR),
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest HR@10: {max([m.get('hr@10', 0) for m in history['val_metrics']]):.4f}")
    print(f"\nModel saved to: {config.paths.TRAINED_MODELS_DIR}")


if __name__ == "__main__":
    main()
