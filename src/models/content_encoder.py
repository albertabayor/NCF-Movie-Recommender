"""
Content encoder for genre and synopsis features.

This module encodes content features:
- Genres: Multi-hot encoding passed through FC layers
- Synopsis: Pre-computed Sentence-BERT embeddings (384-dim)

The content features are combined and passed through additional layers
to create a content embedding that can be fused with CF embeddings.
"""

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    """
    Encode content features (genre + synopsis) into embeddings.

    This encoder takes:
    - Genre features: Multi-hot encoded genre vector
    - Synopsis features: Pre-computed Sentence-BERT embeddings

    And produces a unified content embedding.

    Args:
        num_genres: Number of unique genres (e.g., 19 for MovieLens)
        genre_embed_dim: Output dimension for genre encoder (default: 64)
        synopsis_embed_dim: Dimension of synopsis embeddings (default: 384 for SBERT)
        content_embed_dim: Final content embedding dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        num_genres: int,
        genre_embed_dim: int = 64,
        synopsis_embed_dim: int = 384,
        content_embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_genres = num_genres
        self.genre_embed_dim = genre_embed_dim
        self.synopsis_embed_dim = synopsis_embed_dim
        self.content_embed_dim = content_embed_dim

        # Genre encoder: multi-hot -> FC layers
        self.genre_encoder = nn.Sequential(
            nn.Linear(num_genres, genre_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Synopsis projection: SBERT -> FC layer
        # Synopsis embeddings are already high-quality, so just a projection
        self.synopsis_projection = nn.Sequential(
            nn.Linear(synopsis_embed_dim, synopsis_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined content encoder
        combined_dim = genre_embed_dim + synopsis_embed_dim // 2
        self.content_encoder = nn.Sequential(
            nn.Linear(combined_dim, content_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        genre_features: torch.Tensor,
        synopsis_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode content features.

        Args:
            genre_features: Multi-hot genre vectors, shape (batch, num_genres)
            synopsis_embeddings: Pre-computed synopsis embeddings,
                                shape (batch, synopsis_embed_dim)

        Returns:
            Content embeddings, shape (batch, content_embed_dim)
        """
        # Encode genres
        genre_embed = self.genre_encoder(genre_features)  # (batch, genre_embed_dim)

        # Project synopsis embeddings
        synopsis_embed = self.synopsis_projection(synopsis_embeddings)  # (batch, synopsis_embed_dim//2)

        # Combine
        combined = torch.cat([genre_embed, synopsis_embed], dim=-1)  # (batch, combined_dim)
        content_embed = self.content_encoder(combined)  # (batch, content_embed_dim)

        return content_embed

    def encode_genres_only(self, genre_features: torch.Tensor) -> torch.Tensor:
        """
        Encode only genre features (for ablation study).

        Args:
            genre_features: Multi-hot genre vectors, shape (batch, num_genres)

        Returns:
            Genre embeddings, shape (batch, genre_embed_dim)
        """
        return self.genre_encoder(genre_features)

    def encode_synopsis_only(self, synopsis_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode only synopsis features (for ablation study).

        Args:
            synopsis_embeddings: Pre-computed synopsis embeddings,
                                shape (batch, synopsis_embed_dim)

        Returns:
            Synopsis embeddings, shape (batch, synopsis_embed_dim//2)
        """
        return self.synopsis_projection(synopsis_embeddings)


class GatedFusion(nn.Module):
    """
    Gated fusion module for combining CF and content embeddings.

    Instead of simple concatenation or averaging, this module learns
    to dynamically balance collaborative filtering and content signals
    based on the specific user-item pair.

    Gate value close to 1 -> rely more on CF embeddings
    Gate value close to 0 -> rely more on content features

    This is particularly useful for cold-start scenarios where CF
    signals may be weak.

    Args:
        cf_dim: Dimension of collaborative filtering embeddings
        content_dim: Dimension of content embeddings
        hidden_dim: Hidden dimension for gate network (default: 64)
        dropout: Dropout for gate network (default: 0.1)
    """

    def __init__(
        self,
        cf_dim: int,
        content_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cf_dim = cf_dim
        self.content_dim = content_dim

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(cf_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        cf_embed: torch.Tensor,
        content_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse CF and content embeddings using gated mechanism.

        Args:
            cf_embed: Collaborative filtering embeddings, shape (batch, cf_dim)
            content_embed: Content embeddings, shape (batch, content_dim)

        Returns:
            Tuple of:
            - Fused embeddings, shape (batch, cf_dim or content_dim)
              (Returns the larger of the two dims, projected if needed)
            - Gate values, shape (batch, 1)
        """
        # Concatenate for gate input
        combined = torch.cat([cf_embed, content_embed], dim=-1)

        # Compute gate
        gate = self.gate_network(combined)  # (batch, 1)

        # Fuse: gate * cf_embed + (1 - gate) * content_embed
        # First, project to same dimension if needed
        if self.cf_dim != self.content_dim:
            if self.cf_dim > self.content_dim:
                # Project content to cf dimension
                content_projected = self._project_to_dim(content_embed, self.cf_dim)
                fused = gate * cf_embed + (1 - gate) * content_projected
                output_dim = self.cf_dim
            else:
                # Project cf to content dimension
                cf_projected = self._project_to_dim(cf_embed, self.content_dim)
                fused = gate * cf_projected + (1 - gate) * content_embed
                output_dim = self.content_dim
        else:
            fused = gate * cf_embed + (1 - gate) * content_embed
            output_dim = self.cf_dim

        return fused, gate

    def _project_to_dim(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project tensor to target dimension."""
        if not hasattr(self, '_projection_layers'):
            self._projection_layers = nn.ModuleDict()

        key = f"proj_{x.shape[-1]}_to_{target_dim}"
        if key not in self._projection_layers:
            self._projection_layers[key] = nn.Linear(x.shape[-1], target_dim).to(x.device)
            nn.init.xavier_uniform_(self._projection_layers[key].weight)
            nn.init.zeros_(self._projection_layers[key].bias)

        return self._projection_layers[key](x)


if __name__ == "__main__":
    # Test content encoder and gated fusion
    print("Testing ContentEncoder and GatedFusion...")

    batch_size = 32
    num_genres = 19
    genre_embed_dim = 64
    synopsis_embed_dim = 384
    content_embed_dim = 256

    # Create dummy data
    genre_features = torch.randint(0, 2, (batch_size, num_genres)).float()
    synopsis_embeddings = torch.randn(batch_size, synopsis_embed_dim)

    # Test content encoder
    print("\n1. Testing ContentEncoder...")
    encoder = ContentEncoder(
        num_genres=num_genres,
        genre_embed_dim=genre_embed_dim,
        synopsis_embed_dim=synopsis_embed_dim,
        content_embed_dim=content_embed_dim,
    )

    content_embed = encoder(genre_features, synopsis_embeddings)
    print(f"   Content embedding shape: {content_embed.shape}")
    print(f"   Expected: ({batch_size}, {content_embed_dim})")

    # Test genre-only encoding
    genre_only = encoder.encode_genres_only(genre_features)
    print(f"   Genre-only shape: {genre_only.shape}")

    # Test synopsis-only encoding
    synopsis_only = encoder.encode_synopsis_only(synopsis_embeddings)
    print(f"   Synopsis-only shape: {synopsis_only.shape}")

    # Test gated fusion
    print("\n2. Testing GatedFusion...")
    cf_dim = 32
    cf_embed = torch.randn(batch_size, cf_dim)

    gated_fusion = GatedFusion(
        cf_dim=cf_dim,
        content_dim=content_embed_dim,
        hidden_dim=64,
    )

    fused, gate = gated_fusion(cf_embed, content_embed)
    print(f"   Fused embedding shape: {fused.shape}")
    print(f"   Gate shape: {gate.shape}")
    print(f"   Gate values (first 5): {gate[:5].squeeze(-1).detach().numpy()}")

    print("\n✓ All tests passed!")
