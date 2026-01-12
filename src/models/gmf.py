"""
Generalized Matrix Factorization (GMF) model.

GMF is the neural network equivalent of traditional matrix factorization.
It uses element-wise product of user and item embeddings, followed by
a fully connected layer to produce the final prediction.

Architecture:
    User ID -> Embedding(d) ---┐
                             ├-> Element-wise Multiply -> FC(h) -> Output
    Item ID -> Embedding(d) ---┘

Reference:
    He, X., et al. (2017). "Neural Collaborative Filtering." WWW.
"""

import torch
import torch.nn as nn

from .base import BaseModel


class GMF(BaseModel):
    """
    Generalized Matrix Factorization model.

    This is the neural network equivalent of traditional matrix factorization.
    The key difference is the use of element-wise multiplication instead of
    dot product, followed by a learnable fully-connected layer.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of user/item embeddings (default: 32)
        hidden_dim: Dimension of the hidden FC layer (default: 8)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dim: int = 8,
    ):
        super().__init__(num_users, num_items, embedding_dim)
        self.hidden_dim = hidden_dim

        # GMF-specific layers
        # The element-wise product is computed in forward()
        # Then passed through a hidden FC layer
        self.fc_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.fc_hidden.weight)
        nn.init.xavier_uniform_(self.fc_output.weight)
        nn.init.zeros_(self.fc_hidden.bias)
        nn.init.zeros_(self.fc_output.bias)

    def forward(self, user_ids, item_ids, **kwargs):
        """
        Forward pass of GMF.

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

        # Element-wise product (GMF interaction)
        gmf_embed = user_embed * item_embed  # (batch, embedding_dim)

        # Hidden layer
        hidden = self.fc_hidden(gmf_embed)  # (batch, hidden_dim)

        # Output layer
        output = self.fc_output(hidden)  # (batch, 1)

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
    # Test GMF model
    print("Testing GMF model...")

    num_users = 1000
    num_items = 5000
    batch_size = 64

    model = GMF(num_users, num_items, embedding_dim=32, hidden_dim=8)

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

    print("\n✓ GMF model test passed!")
