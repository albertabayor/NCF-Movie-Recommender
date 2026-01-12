"""
Neural Matrix Factorization (NeuMF) model.

NeuMF combines GMF and MLP branches to capture both linear and non-linear
user-item interactions. The two branches are concatenated and passed through
a final fusion layer.

Architecture:
    User Embedding ──┬──> GMF Branch (element-wise multiply) ─┐
                    │                                        ├─> Concat -> FC -> Output
                    └──> MLP Branch (concat + hidden layers) ─┘

Reference:
    He, X., et al. (2017). "Neural Collaborative Filtering." WWW.
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .gmf import GMF
from .mlp import MLP


class NeuMF(BaseModel):
    """
    Neural Matrix Factorization model.

    NeuMF is an ensemble of GMF and MLP that combines their strengths:
    - GMF captures linear interactions (like traditional MF)
    - MLP captures non-linear, complex patterns

    The two branches are fused in a final layer that learns to balance
    their contributions.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of user/item embeddings (default: 32)
        gmf_hidden_dim: Hidden dim for GMF branch (default: 8)
        mlp_hidden_dims: Hidden dims for MLP branch (default: [128, 64, 32])
        mlp_dropout: Dropout for MLP branch (default: 0.2)
        fusion_dim: Dimension of the fusion layer (default: 32)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        gmf_hidden_dim: int = 8,
        mlp_hidden_dims: list = None,
        mlp_dropout: float = 0.2,
        fusion_dim: int = 32,
    ):
        # Don't call super().__init__() because we'll create our own embeddings
        nn.Module.__init__(self)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.gmf_hidden_dim = gmf_hidden_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [128, 64, 32]
        self.mlp_dropout = mlp_dropout
        self.fusion_dim = fusion_dim

        # Separate embeddings for GMF and MLP (as per original paper)
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)

        # GMF branch
        self.gmf_fc = nn.Linear(embedding_dim, gmf_hidden_dim)

        # MLP branch
        mlp_input_dim = 2 * embedding_dim
        self.mlp_layers = nn.ModuleList()
        self.mlp_dropout_layers = nn.ModuleList()

        prev_dim = mlp_input_dim
        for hidden_dim in self.mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_dropout_layers.append(nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim

        # Fusion layer
        self.fusion_fc = nn.Linear(gmf_hidden_dim + prev_dim, fusion_dim)
        self.fusion_dropout = nn.Dropout(0.1)
        self.output_fc = nn.Linear(fusion_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        # GMF
        nn.init.xavier_uniform_(self.gmf_fc.weight)
        nn.init.zeros_(self.gmf_fc.bias)

        # MLP
        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Fusion
        nn.init.xavier_uniform_(self.fusion_fc.weight)
        nn.init.zeros_(self.fusion_fc.bias)
        nn.init.xavier_uniform_(self.output_fc.weight)
        nn.init.zeros_(self.output_fc.bias)

    def forward(self, user_ids, item_ids, **kwargs):
        """
        Forward pass of NeuMF.

        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
            item_ids: Tensor of item IDs, shape (batch_size,)
            **kwargs: Unused (for interface consistency)

        Returns:
            Predicted scores, shape (batch_size, 1)
        """
        # GMF branch
        gmf_user_embed = self.gmf_user_embedding(user_ids)
        gmf_item_embed = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_embed * gmf_item_embed  # Element-wise
        gmf_hidden = self.gmf_fc(gmf_output)  # (batch, gmf_hidden_dim)

        # MLP branch
        mlp_user_embed = self.mlp_user_embedding(user_ids)
        mlp_item_embed = self.mlp_item_embedding(item_ids)
        mlp_concat = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)

        x = mlp_concat
        for layer, dropout in zip(self.mlp_layers, self.mlp_dropout_layers):
            x = layer(x)
            x = torch.relu(x)
            x = dropout(x)

        mlp_hidden = x  # (batch, mlp_final_hidden_dim)

        # Fusion
        fusion_input = torch.cat([gmf_hidden, mlp_hidden], dim=-1)
        fusion_hidden = self.fusion_fc(fusion_input)
        fusion_hidden = torch.relu(fusion_hidden)
        fusion_hidden = self.fusion_dropout(fusion_hidden)

        # Output
        output = self.output_fc(fusion_hidden)

        return output

    def predict_scores(self, user_ids, item_ids) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Wrapper that applies sigmoid to get probabilities.

        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs

        Returns:
            Predicted probabilities (0-1), shape (batch_size,)
        """
        logits = self.forward(user_ids, item_ids)
        return torch.sigmoid(logits).squeeze(-1)

    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device


if __name__ == "__main__":
    # Test NeuMF model
    print("Testing NeuMF model...")

    num_users = 1000
    num_items = 5000
    batch_size = 64

    model = NeuMF(
        num_users,
        num_items,
        embedding_dim=32,
        gmf_hidden_dim=8,
        mlp_hidden_dims=[128, 64, 32],
        fusion_dim=32,
    )

    # Print model summary
    print(model.summary())

    # Test forward pass
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))

    output = model(user_ids, item_ids)
    print(f"\nOutput shape: {output.shape}")
    print(f"Sample outputs: {output[:5].squeeze(-1)}")

    # Test prediction
    probs = model.predict_scores(user_ids, item_ids)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Sample probs: {probs[:5]}")

    print("\n✓ NeuMF model test passed!")
