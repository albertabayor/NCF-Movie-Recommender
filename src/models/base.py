"""
Base model class for NCF models.

This module defines the abstract base class that all NCF models inherit from.
It provides common functionality like saving, loading, and parameter counting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all NCF models.

    All models (GMF, MLP, NeuMF, NeuMF+) should inherit from this class
    and implement the required methods.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
    ):
        """
        Initialize the base model.

        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of user/item embeddings
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings (shared by most models)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    @abstractmethod
    def forward(self, user_ids, item_ids, **kwargs):
        """
        Forward pass of the model.

        Each model should implement its own forward pass.

        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            **kwargs: Additional model-specific arguments (e.g., genre_features)

        Returns:
            Predicted scores/logits
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def predict(self, user_ids, item_ids, **kwargs):
        """
        Make predictions for user-item pairs.

        This is an alias for forward() that can be used during inference.

        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            **kwargs: Additional model-specific arguments

        Returns:
            Predicted scores
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids, item_ids, **kwargs)
        return scores

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """
        Get the embedding vector for a specific user.

        Args:
            user_id: User ID

        Returns:
            Embedding vector
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            embed = self.user_embedding(user_tensor)
        return embed.squeeze(0)

    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """
        Get the embedding vector for a specific item.

        Args:
            item_id: Item ID

        Returns:
            Embedding vector
        """
        self.eval()
        with torch.no_grad():
            item_tensor = torch.LongTensor([item_id]).to(self.device)
            embed = self.item_embedding(item_tensor)
        return embed.squeeze(0)

    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_info(self) -> Dict[str, int]:
        """
        Get detailed parameter information for each layer.

        Returns:
            Dict mapping layer names to parameter counts
        """
        info = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                info[name] = param.numel()
        return info

    def save(self, filepath: str, **kwargs) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save the checkpoint
            **kwargs: Additional information to save (e.g., epoch, metrics)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "num_parameters": self.count_parameters(),
            **kwargs,
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, model_cls: Optional[type] = None, **kwargs) -> "BaseModel":
        """
        Load model from checkpoint.

        Args:
            filepath: Path to the checkpoint
            model_cls: Model class to instantiate (uses cls if None)
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model
        """
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location="cpu")

        # Get model class
        model_cls = model_cls or cls

        # Extract saved parameters
        num_users = checkpoint.get("num_users", kwargs.get("num_users"))
        num_items = checkpoint.get("num_items", kwargs.get("num_items"))
        embedding_dim = checkpoint.get("embedding_dim", kwargs.get("embedding_dim", 32))

        # Create model instance
        model = model_cls(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            **kwargs
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Model loaded from {filepath}")
        print(f"  Parameters: {model.count_parameters():,}")

        # Print additional saved info
        for key, value in checkpoint.items():
            if key not in ["model_state_dict", "num_users", "num_items", "embedding_dim", "num_parameters"]:
                print(f"  {key}: {value}")

        return model

    def summary(self) -> str:
        """
        Print a summary of the model architecture.

        Returns:
            Summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append("=" * 60)
        lines.append(f"Num users: {self.num_users:,}")
        lines.append(f"Num items: {self.num_items:,}")
        lines.append(f"Embedding dim: {self.embedding_dim}")
        lines.append(f"Total parameters: {self.count_parameters():,}")
        lines.append("-" * 60)

        for name, count in self.get_parameter_info().items():
            lines.append(f"{name:50s} {count:>12,}")

        lines.append("=" * 60)

        return "\n".join(lines)


def create_model(
    model_type: str,
    num_users: int,
    num_items: int,
    **kwargs
) -> BaseModel:
    """
    Factory function to create models by type.

    Args:
        model_type: Type of model ('gmf', 'mlp', 'neumf', 'neumf_plus')
        num_users: Number of users
        num_items: Number of items
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    from .gmf import GMF
    from .mlp import MLP
    from .neumf import NeuMF
    from .neumf_plus import NeuMFPlus

    model_classes = {
        "gmf": GMF,
        "mlp": MLP,
        "neumf": NeuMF,
        "neumf_plus": NeuMFPlus,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(model_classes.keys())}"
        )

    return model_classes[model_type](
        num_users=num_users,
        num_items=num_items,
        **kwargs
    )
