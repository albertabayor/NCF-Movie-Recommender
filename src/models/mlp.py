"""
Multi-Layer Perceptron (MLP) model for collaborative filtering.

MLP uses concatenation of user and item embeddings, followed by
multiple hidden layers with non-linear activations.

Architecture:
    User ID -> Embedding(d) ---┐
                             ├-> Concat -> FC(h1) -> ReLU -> Dropout -> FC(h2) -> ... -> Output
    Item ID -> Embedding(d) ---┘

Reference:
    He, X., et al. (2017). "Neural Collaborative Filtering." WWW.
"""

import torch
import torch.nn as nn

from .base import BaseModel


class MLP(BaseModel):
    """
    Multi-Layer Perceptron model for collaborative filtering.

    Unlike GMF which uses element-wise product, MLP concatenates the
    user and item embeddings and passes them through multiple hidden
    layers to learn non-linear interactions.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of user/item embeddings (default: 32)
        hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__(num_users, num_items, embedding_dim)

        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Build MLP layers
        # Input: concatenated user + item embeddings
        input_dim = 2 * embedding_dim

        self.mlp_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        self.fc_output = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.fc_output.weight)
        nn.init.zeros_(self.fc_output.bias)

    def forward(self, user_ids, item_ids, **kwargs):
        """
        Forward pass of MLP.

        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
            item_ids: Tensor of item IDs, shape (batch_size,)
            **kwargs: Unused (for interface consistency)

        Returns:
            Predicted scores, shape (batch_size, 1)
        """
        # Get embeddings
        user_embed = self.user_embedding(user_ids)  # (batch, embedding_dim)
        item_embed = self.item_embedding(item_ids)  # (batch, embedding_dim)

        # Concatenate embeddings
        concat = torch.cat([user_embed, item_embed], dim=-1)  # (batch, 2*embedding_dim)

        # Pass through MLP layers
        x = concat
        for layer, dropout in zip(self.mlp_layers, self.dropout_layers):
            x = layer(x)
            x = torch.relu(x)
            x = dropout(x)

        # Output layer
        output = self.fc_output(x)  # (batch, 1)

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


if __name__ == "__main__":
    # Test MLP model
    print("Testing MLP model...")

    num_users = 1000
    num_items = 5000
    batch_size = 64

    model = MLP(num_users, num_items, embedding_dim=32, hidden_dims=[128, 64, 32])

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

    print("\n✓ MLP model test passed!")
