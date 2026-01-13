#!/usr/bin/env python3
"""
Training script for NeuMF+ model (Neural Collaborative Filtering + Content Features).

This script trains the advanced NeuMF+ model with:
- Genre features (multi-hot encoding)
- Gated fusion mechanism
- Optional data sampling for faster training
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import torch

from src.config import config
from src.models.neumf_plus import NeuMFPlus
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
            recommended_batch = 128
        elif gpu_memory_gb < 12:
            recommended_batch = 256
        else:
            recommended_batch = 512
        print(f"  Recommended batch size: {recommended_batch}")
        return device, recommended_batch
    else:
        print("⚠️  No GPU detected. Training on CPU will be very slow.")
        return "cpu", 32


def main():
    """Run NeuMF+ training."""
    print("=" * 70)
    print("NeuMF+ TRAINING - COLLABORATIVE FILTERING + CONTENT FEATURES")
    print("=" * 70)

    # Check GPU
    device, recommended_batch = check_gpu()

    # Load preprocessed data
    print("\n[1/6] Loading preprocessed data...")
    train_df = pd.read_pickle(config.paths.train_path)
    val_df = pd.read_pickle(config.paths.val_path)
    test_df = pd.read_pickle(config.paths.test_path)

    with open(config.paths.mappings_path, "rb") as f:
        mappings = pickle.load(f)

    num_users = mappings["num_users"]
    num_items = mappings["num_items"]
    num_genres = mappings["num_genres"]

    # Optional: Sample subset for faster training
    SAMPLE_RATIO = float(os.environ.get("SAMPLE_RATIO", "0.1"))  # Default 10%

    if SAMPLE_RATIO < 1.0:
        print(f"\n[2/6] Sampling {SAMPLE_RATIO*100:.0f}% of training data...")
        np.random.seed(42)
        sample_idx = np.random.choice(
            len(train_df),
            int(len(train_df) * SAMPLE_RATIO),
            replace=False
        )
        train_df = train_df.iloc[sample_idx].copy()
        print(f"  Sampled: {len(train_df):,} ratings")

    train_users = train_df["userId"].values
    train_items = train_df["movieId"].values
    val_users = val_df["userId"].values
    val_items = val_df["movieId"].values

    # Extract genre features
    print("\n[3/6] Extracting genre features...")
    train_genre_features = np.stack(train_df["genre_features"].values)
    val_genre_features = np.stack(val_df["genre_features"].values)

    print(f"  Train genre shape: {train_genre_features.shape}")
    print(f"  Val genre shape: {val_genre_features.shape}")

    # Build user history for negative sampling
    print("\n[4/6] Building user history for negative sampling...")
    user_history = build_user_history(train_users, train_items)

    # Create NeuMF+ model
    print("\n[5/6] Creating NeuMF+ model...")
    model = NeuMFPlus(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        # Content encoder settings
        genre_embed_dim=64,
        content_embed_dim=256,
        content_encoder_dropout=0.1,
        # Gated fusion settings
        gated_fusion_hidden_dim=64,
        gated_fusion_dropout=0.1,
        # Output settings
        output_hidden_dim=64,
        output_dropout=0.2,
        # Ablation study flags
        use_genre=True,         # Enable genre features
        use_synopsis=False,     # Disable synopsis (need SBERT embeddings)
        use_gated_fusion=True,  # Enable gated fusion
    )

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: NeuMF+ (Neural Collaborative Filtering + Content)")
    print(f"  Parameters: {param_count:,}")
    print(f"  Estimated size: {param_count * 4 / 1e6:.1f} MB")
    print(f"\nArchitecture:")
    print(f"  ✓ CF branch (NeuMF: GMF + MLP)")
    print(f"  ✓ Content encoder (Genre features)")
    print(f"  ✓ Gated fusion (CF + Content)")
    print(f"\n  Ablation flags:")
    print(f"    - use_genre: True")
    print(f"    - use_synopsis: False (need SBERT embeddings)")
    print(f"    - use_gated_fusion: True")

    # Training configuration
    print("\n[6/6] Configuring training...")
    batch_size = min(recommended_batch, 256)
    learning_rate = 1e-3
    num_epochs = 30
    num_negatives = 4
    num_workers = 0  # Single worker to save RAM
    use_amp = (device == "cuda")

    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Negatives per positive: {num_negatives}")
    print(f"  Data workers: {num_workers}")
    print(f"  Mixed precision (FP16): {use_amp}")

    # Estimate training time
    if device == "cuda":
        it_per_sec = 4  # Conservative estimate
        total_steps = len(train_users) // batch_size
        time_per_epoch = total_steps / it_per_sec / 60
        print(f"\n  Estimated time per epoch: {time_per_epoch:.0f} minutes")
        print(f"  Estimated total time: {time_per_epoch * 5:.0f} minutes (~{time_per_epoch * 5 / 60:.1f} hours)")

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    # Prepare validation data with genre features
    val_data = {
        "users": val_users,
        "items": val_items,
        "genre_features": val_genre_features,
    }

    history = train_model(
        model=model,
        train_users=train_users,
        train_items=train_items,
        val_data=val_data,
        num_items=num_items,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        num_negatives=num_negatives,
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
        save_dir=str(config.paths.TRAINED_MODELS_DIR),
        early_stopping_patience=5,
        early_stopping_metric="hr@10",
        lr_scheduler_patience=3,
        lr_scheduler_factor=0.5,
        log_dir=str(config.paths.TENSORBOARD_LOG_DIR),
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest HR@10: {max([m.get('hr@10', 0) for m in history['val_metrics']]):.4f}")
    print(f"\nModel saved to: {config.paths.TRAINED_MODELS_DIR}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
