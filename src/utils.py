"""
Utility functions for NCF Movie Recommender.

This module provides helper functions for:
- Loading and saving processed data
- Computing synopsis embeddings with Sentence-BERT
- Setting random seeds
- Logging and monitoring
"""

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_mappings(data_dir: str = "data") -> Dict:
    """
    Load saved mappings from preprocessing.

    Args:
        data_dir: Path to the data directory

    Returns:
        Dict with mappings and metadata
    """
    mappings_path = Path(data_dir) / "mappings.pkl"

    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    return mappings


def load_statistics(data_dir: str = "data") -> Dict:
    """
    Load statistics from preprocessing.

    Args:
        data_dir: Path to the data directory

    Returns:
        Dict with statistics
    """
    stats_path = Path(data_dir) / "statistics.json"

    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    return stats


def load_data_splits(data_dir: str = "data") -> Dict[str, Any]:
    """
    Load train/val/test data splits.

    Args:
        data_dir: Path to the data directory

    Returns:
        Dict with train, val, test, cold_start_test DataFrames
    """
    import pandas as pd

    splits = {}

    for split_name in ["train", "val", "test", "cold_start_test"]:
        file_path = Path(data_dir) / f"{split_name}.pkl"
        if file_path.exists():
            splits[split_name] = pd.read_pickle(file_path)
        else:
            print(f"Warning: {split_name} split not found at {file_path}")

    return splits


def load_genre_features(data_dir: str = "data") -> np.ndarray:
    """
    Load genre features from preprocessed data.

    Args:
        data_dir: Path to the data directory

    Returns:
        Genre features array (num_items, num_genres)
    """
    import pandas as pd

    # Load from any split that has genre_features
    for split_name in ["train", "val", "test"]:
        file_path = Path(data_dir) / f"{split_name}.pkl"
        if file_path.exists():
            df = pd.read_pickle(file_path)
            if "genre_features" in df.columns:
                # Get unique genre features
                num_items = load_mappings(data_dir)["num_items"]
                genre_features = np.zeros((num_items, len(df["genre_features"].iloc[0])))

                # Collect from all splits
                for split in ["train", "val", "test"]:
                    split_path = Path(data_dir) / f"{split}.pkl"
                    if split_path.exists():
                        split_df = pd.read_pickle(split_path)
                        for _, row in split_df.iterrows():
                            genre_features[row["movieId"]] = row["genre_features"]

                return genre_features

    raise ValueError("Genre features not found in preprocessed data")


def compute_synopsis_embeddings(
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = "data",
    batch_size: int = 32,
    save_path: str = None,
) -> np.ndarray:
    """
    Compute synopsis embeddings using Sentence-BERT.

    This should be run after preprocessing to generate embeddings
    for all movie overviews.

    Args:
        model_name: Sentence-BERT model name
        data_dir: Path to the data directory
        batch_size: Batch size for encoding
        save_path: Path to save embeddings (default: data/synopsis_embeddings.npy)

    Returns:
        Synopsis embeddings array (num_items, embed_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        )

    # Load data
    splits = load_data_splits(data_dir)
    mappings = load_mappings(data_dir)

    num_items = mappings["num_items"]

    # Collect all unique synopses
    from collections import defaultdict
    item_synopses = {}

    for split_name, df in splits.items():
        if split_name == "cold_start_test":
            continue
        for _, row in df.iterrows():
            item_id = row["movieId"]
            if item_id not in item_synopses:
                # We need to get the original overview from metadata
                # For now, skip (this should be integrated into preprocessing)
                pass

    raise NotImplementedError(
        "Synopsis embeddings should be computed during preprocessing. "
        "See preprocessing.py for integration."
    )


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: str = "cuda") -> torch.device:
    """
    Get the appropriate device for training/inference.

    Args:
        device: Preferred device ("cuda" or "cpu")

    Returns:
        torch.device object
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class EarlyStopping:
    """
    Early stopping utility for training.

    Stops training when a monitored metric stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "max" or "min" for the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dict of metric names to values
        prefix: Prefix to add to each line
    """
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{name}: {value:.4f}")
        else:
            print(f"{prefix}{name}: {value}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dict of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> Dict:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Dict with epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {filepath}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


if __name__ == "__main__":
    print("Utility functions for NCF Movie Recommender")
    print("Import this module to use the utility functions")
