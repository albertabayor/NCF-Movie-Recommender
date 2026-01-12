"""
Configuration file for NCF Movie Recommender.

This module contains all hyperparameters, paths, and settings
for the Neural Collaborative Filtering implementation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# PATHS
# ============================================================================

@dataclass
class Paths:
    """Data and model paths."""

    # Raw data paths
    DATASETS_DIR: str = "datasets"
    RATINGS_FILE: str = "ratings.csv"
    MOVIES_METADATA_FILE: str = "movies_metadata.csv"
    LINKS_FILE: str = "links.csv"
    KEYWORDS_FILE: str = "keywords.csv"

    # Processed data paths
    DATA_DIR: str = "data"
    TRAIN_FILE: str = "train.pkl"
    VAL_FILE: str = "val.pkl"
    TEST_FILE: str = "test.pkl"
    COLD_START_TEST_FILE: str = "cold_start_test.pkl"
    MAPPINGS_FILE: str = "mappings.pkl"
    STATISTICS_FILE: str = "statistics.json"
    SYNOPSIS_EMBEDDINGS_FILE: str = "synopsis_embeddings.npy"

    # Experiment paths
    EXPERIMENTS_DIR: str = "experiments"
    TRAINED_MODELS_DIR: str = "experiments/trained_models"
    LOGS_DIR: str = "experiments/logs"

    # Tensorboard log dir
    TENSORBOARD_LOG_DIR: str = "experiments/logs/tensorboard"

    @property
    def ratings_path(self) -> str:
        return os.path.join(self.DATASETS_DIR, self.RATINGS_FILE)

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.DATASETS_DIR, self.MOVIES_METADATA_FILE)

    @property
    def links_path(self) -> str:
        return os.path.join(self.DATASETS_DIR, self.LINKS_FILE)

    @property
    def keywords_path(self) -> str:
        return os.path.join(self.DATASETS_DIR, self.KEYWORDS_FILE)

    @property
    def train_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.TRAIN_FILE)

    @property
    def val_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.VAL_FILE)

    @property
    def test_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.TEST_FILE)

    @property
    def cold_start_test_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.COLD_START_TEST_FILE)

    @property
    def mappings_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.MAPPINGS_FILE)

    @property
    def statistics_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.STATISTICS_FILE)

    @property
    def synopsis_embeddings_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.SYNOPSIS_EMBEDDINGS_FILE)

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.TRAINED_MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_DIR, exist_ok=True)


# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    # Embedding dimensions
    USER_EMBEDDING_DIM: int = 32
    ITEM_EMBEDDING_DIM: int = 32

    # GMF architecture
    GMF_HIDDEN_DIM: int = 8

    # MLP architecture
    MLP_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128, 64, 32])
    MLP_DROPOUT: float = 0.2

    # NeuMF architecture
    NEUMF_FUSION_DIM: int = 32

    # Content encoder
    GENRE_EMBEDDING_DIM: int = 64
    SYNOPSIS_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # 384-dim
    SYNOPSIS_EMBEDDING_DIM: int = 384

    # Gated fusion
    GATED_FUSION_HIDDEN_DIM: int = 64
    GATED_FUSION_DROPOUT: float = 0.1

    # Output layer
    OUTPUT_HIDDEN_DIM: int = 64
    OUTPUT_DROPOUT: float = 0.2


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Batch settings
    BATCH_SIZE: int = 256
    NUM_NEGATIVES: int = 4

    # Optimizer settings
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    BETAS: tuple = (0.9, 0.999)

    # Learning rate scheduler
    LR_SCHEDULER_PATIENCE: int = 3
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_MIN: float = 1e-6

    # Training settings
    NUM_EPOCHS: int = 30
    GRADIENT_CLIP_MAX_NORM: float = 5.0

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_METRIC: str = "hr@10"

    # Device
    DEVICE: str = "cuda"  # Will be set to 'cuda' if available, else 'cpu'

    # Random seed
    SEED: int = 42

    # Negative sampling strategy
    NEGATIVE_SAMPLING_STRATEGY: str = "uniform"  # 'uniform' or 'popularity'


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation hyperparameters."""

    # Top-K metrics
    K_VALUES: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Cold-start thresholds
    COLD_START_USER_THRESHOLD: int = 10
    COLD_START_ITEM_THRESHOLD: int = 10

    # Evaluation settings
    NUM_NEGATIVES_TEST: int = 99  # For ranking metrics (1 positive + 99 negatives)


# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    # Filtering thresholds
    MIN_USER_RATINGS: int = 5
    MIN_ITEM_RATINGS: int = 5

    # Train/Val/Test split ratios (time-based, per user)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Rating validation
    MIN_RATING: float = 0.5
    MAX_RATING: float = 5.0


# ============================================================================
# ABSTUDY STUDY CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Ablation study configuration."""

    # Model variants to train
    MODELS: List[str] = field(default_factory=lambda: [
        "gmf",
        "mlp",
        "neumf",
        "neumf_genre",
        "neumf_synopsis",
        "neumf_genre_synopsis",
        "neumf_plus",  # Final model with gated fusion
    ])

    # Description of each variant
    MODEL_DESCRIPTIONS: dict = field(default_factory=lambda: {
        "gmf": "Baseline: Generalized Matrix Factorization",
        "mlp": "Baseline: Multi-Layer Perceptron",
        "neumf": "Baseline: Neural Matrix Factorization (GMF + MLP)",
        "neumf_genre": "NeuMF + Genre features (simple concat)",
        "neumf_synopsis": "NeuMF + Synopsis features (simple concat)",
        "neumf_genre_synopsis": "NeuMF + Genre + Synopsis (simple concat)",
        "neumf_plus": "NeuMF + Genre + Synopsis + Gated Fusion (FINAL)",
    })


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config:
    """Main configuration class combining all configs."""

    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self):
        """Initialize paths and device."""
        self.paths.ensure_dirs()
        self._set_device()

    def _set_device(self) -> None:
        """Set device to CUDA if available."""
        import torch
        if torch.cuda.is_available():
            self.train.DEVICE = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.train.DEVICE = "cpu"
            print("CUDA not available, using CPU")

    def set_seed(self) -> None:
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        import torch

        random.seed(self.train.SEED)
        np.random.seed(self.train.SEED)
        torch.manual_seed(self.train.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.train.SEED)
            torch.cuda.manual_seed_all(self.train.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global config instance
config = Config()
