"""
Memory-efficient configuration for 4GB VRAM (RTX 3050 Laptop).

This configuration is optimized for training on GPUs with limited VRAM.
Use this by importing: from src.config_4gb import config_4gb
"""

from dataclasses import dataclass, field
from typing import List

# Import from base config
from src.config import Paths, ModelConfig, TrainConfig, EvalConfig, DataConfig, Config


@dataclass
class TrainConfig4GB(TrainConfig):
    """Training config optimized for 4GB VRAM."""

    # Reduced batch size for memory constraints
    BATCH_SIZE: int = 128  # Reduced from 256

    # Fewer negatives to save memory
    NUM_NEGATIVES: int = 3  # Reduced from 4

    # Same learning rate (batch size normalized)
    LEARNING_RATE: float = 1e-3


@dataclass
class ModelConfig4GB(ModelConfig):
    """Model config optimized for 4GB VRAM."""

    # Same embedding dimensions (these don't consume much VRAM)
    USER_EMBEDDING_DIM: int = 32
    ITEM_EMBEDDING_DIM: int = 32

    # GMF architecture
    GMF_HIDDEN_DIM: int = 8

    # Slightly smaller MLP to save memory
    MLP_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [96, 48, 24])
    MLP_DROPOUT: float = 0.2

    # NeuMF architecture
    NEUMF_FUSION_DIM: int = 24

    # Content encoder
    GENRE_EMBEDDING_DIM: int = 48  # Reduced from 64
    SYNOPSIS_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SYNOPSIS_EMBEDDING_DIM: int = 384

    # Gated fusion
    GATED_FUSION_HIDDEN_DIM: int = 48  # Reduced from 64
    GATED_FUSION_DROPOUT: float = 0.1

    # Output layer
    OUTPUT_HIDDEN_DIM: int = 48  # Reduced from 64
    OUTPUT_DROPOUT: float = 0.2


@dataclass
class Config4GB(Config):
    """Complete config optimized for 4GB VRAM."""

    # Override with 4GB-optimized configs
    model: ModelConfig = field(default_factory=ModelConfig4GB)
    train: TrainConfig = field(default_factory=TrainConfig4GB)


# Global instance
config_4gb = Config4GB()


def get_config(use_4gb: bool = True) -> Config:
    """
    Get the appropriate config based on available VRAM.

    Args:
        use_4gb: If True, return 4GB-optimized config

    Returns:
        Config instance
    """
    if use_4gb:
        return config_4gb
    return Config()


if __name__ == "__main__":
    print("4GB VRAM Optimized Configuration")
    print("=" * 50)
    cfg = config_4gb
    print(f"Batch Size: {cfg.train.BATCH_SIZE}")
    print(f"Negatives: {cfg.train.NUM_NEGATIVES}")
    print(f"MLP Hidden Dims: {cfg.model.MLP_HIDDEN_DIMS}")
    print(f"Genre Embed Dim: {cfg.model.GENRE_EMBEDDING_DIM}")
    print(f"Fusion Dim: {cfg.model.GATED_FUSION_HIDDEN_DIM}")
