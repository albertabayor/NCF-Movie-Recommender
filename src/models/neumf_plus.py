"""
NeuMF+ model with Genre, Synopsis, and Gated Fusion.

This is the final proposed model that combines:
1. NeuMF (GMF + MLP) for collaborative filtering
2. Content encoder for genre + synopsis features
3. Gated fusion to dynamically balance CF and content signals

Architecture:
                    ┌─────────────────────────────────────┐
                    │          USER & ITEM EMBEDDINGS      │
                    ├─────────────────────────────────────┤
  User ID -> Embed -> ┐                                 │
                     ├──> GMF (elem-mult) ──┐           │
  Item ID -> Embed -> ┘                     │           │
                                       ┌────┤           │
                                       │    └──> NeuMF ─┼──┐
                                       │                │  │
                    ┌──────────────────┴─────┐          │  │
                    │      CONTENT FEATURES   │          │  │
                    ├──────────────────────────┤          │  │
  Genres (multi-hot) ──> FC ──┐               │          │  │
                              ├──> Content ───┼─────┐    │  │
  Synopsis (SBERT 384-d) ─> FC─┘              │     │    │  │
                                              │     │    │  │
                    ┌──────────────────────────┴─────┐    │  │
                    │         GATED FUSION            │    │  │
                    │  CF embed + Content embed ──>   │    │  │
                    │  Gate Network ──> Weighted Sum │◄───┴──┘
                    └────────────────────────────────┘
                                 │
                                 ▼
                           FC(64) -> Dropout -> FC(1) -> Sigmoid

Reference:
    Extended from He, X., et al. (2017). "Neural Collaborative Filtering." WWW
"""

import torch
import torch.nn as nn

from .content_encoder import ContentEncoder, GatedFusion


class NeuMFPlus(nn.Module):
    """
    NeuMF+ with Genre, Synopsis, and Gated Fusion.

    This model extends NeuMF by adding content features (genre + synopsis)
    and a gated fusion mechanism to dynamically balance CF and content signals.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        num_genres: Number of unique genres
        embedding_dim: Dimension of CF embeddings (default: 32)
        gmf_hidden_dim: Hidden dim for GMF branch (default: 8)
        mlp_hidden_dims: Hidden dims for MLP branch (default: [128, 64, 32])
        mlp_dropout: Dropout for MLP branch (default: 0.2)
        fusion_dim: Dimension of NeuMF fusion layer (default: 32)
        genre_embed_dim: Genre encoder output dim (default: 64)
        synopsis_embed_dim: Synopsis embedding dim (default: 384)
        content_embed_dim: Content encoder output dim (default: 256)
        gated_fusion_hidden_dim: Hidden dim for gate network (default: 64)
        gated_fusion_dropout: Dropout for gate network (default: 0.1)
        output_hidden_dim: Hidden dim for output layer (default: 64)
        output_dropout: Dropout for output layer (default: 0.2)

    Model variants for ablation study:
    - use_genre: Include genre features
    - use_synopsis: Include synopsis features
    - use_gated_fusion: Use gated fusion (vs simple concat)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        # CF (NeuMF) parameters
        embedding_dim: int = 32,
        gmf_hidden_dim: int = 8,
        mlp_hidden_dims: list = None,
        mlp_dropout: float = 0.2,
        fusion_dim: int = 32,
        # Content encoder parameters
        genre_embed_dim: int = 64,
        synopsis_embed_dim: int = 384,
        content_embed_dim: int = 256,
        content_encoder_dropout: float = 0.1,
        # Gated fusion parameters
        gated_fusion_hidden_dim: int = 64,
        gated_fusion_dropout: float = 0.1,
        # Output parameters
        output_hidden_dim: int = 64,
        output_dropout: float = 0.2,
        # Ablation study flags
        use_genre: bool = True,
        use_synopsis: bool = True,
        use_gated_fusion: bool = True,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.use_genre = use_genre
        self.use_synopsis = use_synopsis
        self.use_gated_fusion = use_gated_fusion

        # Calculate content dimensions
        if use_genre and use_synopsis:
            # Full content encoder
            self.content_encoder = ContentEncoder(
                num_genres=num_genres,
                genre_embed_dim=genre_embed_dim,
                synopsis_embed_dim=synopsis_embed_dim,
                content_embed_dim=content_embed_dim,
                dropout=content_encoder_dropout,
            )
            actual_content_dim = content_embed_dim
        elif use_genre:
            # Genre only
            self.genre_encoder = nn.Sequential(
                nn.Linear(num_genres, genre_embed_dim),
                nn.ReLU(),
                nn.Dropout(content_encoder_dropout),
            )
            actual_content_dim = genre_embed_dim
        elif use_synopsis:
            # Synopsis only
            self.synopsis_projection = nn.Sequential(
                nn.Linear(synopsis_embed_dim, synopsis_embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(content_encoder_dropout),
            )
            actual_content_dim = synopsis_embed_dim // 2
        else:
            # No content features
            actual_content_dim = 0

        # CF (NeuMF) branch
        # Separate embeddings for GMF and MLP
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
        mlp_hidden_dims = mlp_hidden_dims or [128, 64, 32]
        mlp_input_dim = 2 * embedding_dim
        self.mlp_layers = nn.ModuleList()
        self.mlp_dropout_layers = nn.ModuleList()

        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_dropout_layers.append(nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim

        # NeuMF fusion layer
        self.neumf_fusion_fc = nn.Linear(gmf_hidden_dim + prev_dim, fusion_dim)
        self.neumf_fusion_dropout = nn.Dropout(0.1)
        self.cf_output_dim = fusion_dim

        # Content fusion
        if actual_content_dim > 0:
            if use_gated_fusion:
                # Use gated fusion
                self.gated_fusion = GatedFusion(
                    cf_dim=self.cf_output_dim,
                    content_dim=actual_content_dim,
                    hidden_dim=gated_fusion_hidden_dim,
                    dropout=gated_fusion_dropout,
                )
                # Gated fusion outputs the max of cf_dim and content_dim
                self.final_input_dim = max(self.cf_output_dim, actual_content_dim)
            else:
                # Simple concatenation
                self.final_input_dim = self.cf_output_dim + actual_content_dim
        else:
            # No content features
            self.final_input_dim = self.cf_output_dim

        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(self.final_input_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(output_dropout),
            nn.Linear(output_hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        # Genre-only encoder
        if self.use_genre and not self.use_synopsis:
            if hasattr(self, 'genre_encoder'):
                for module in self.genre_encoder:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        # Synopsis-only projection
        if self.use_synopsis and not self.use_genre:
            if hasattr(self, 'synopsis_projection'):
                for module in self.synopsis_projection:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        # NeuMF components
        nn.init.xavier_uniform_(self.gmf_fc.weight)
        nn.init.zeros_(self.gmf_fc.bias)

        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.neumf_fusion_fc.weight)
        nn.init.zeros_(self.neumf_fusion_fc.bias)

        # Output layers
        for module in self.output_fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor = None,
        synopsis_embeddings: torch.Tensor = None,
        return_gate: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of NeuMF+.

        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
            item_ids: Tensor of item IDs, shape (batch_size,)
            genre_features: Multi-hot genre vectors, shape (batch, num_genres)
                           Can be None if use_genre=False
            synopsis_embeddings: Synopsis embeddings, shape (batch, 384)
                                Can be None if use_synopsis=False
            return_gate: If True, return gate values along with output

        Returns:
            If return_gate=False: Predicted scores, shape (batch, 1)
            If return_gate=True: Tuple of (scores, gate_values)
        """
        # ===== CF (NeuMF) Branch =====
        # GMF
        gmf_user_embed = self.gmf_user_embedding(user_ids)
        gmf_item_embed = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_embed * gmf_item_embed
        gmf_hidden = self.gmf_fc(gmf_output)

        # MLP
        mlp_user_embed = self.mlp_user_embedding(user_ids)
        mlp_item_embed = self.mlp_item_embedding(item_ids)
        mlp_concat = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)

        x = mlp_concat
        for layer, dropout in zip(self.mlp_layers, self.mlp_dropout_layers):
            x = layer(x)
            x = torch.relu(x)
            x = dropout(x)

        mlp_hidden = x

        # NeuMF fusion
        neumf_input = torch.cat([gmf_hidden, mlp_hidden], dim=-1)
        cf_embed = self.neumf_fusion_fc(neumf_input)
        cf_embed = torch.relu(cf_embed)
        cf_embed = self.neumf_fusion_dropout(cf_embed)

        # ===== Content Branch =====
        content_embed = None
        if self.use_genre and self.use_synopsis:
            # Ensure genre_features and synopsis_embeddings are provided
            if genre_features is None or synopsis_embeddings is None:
                raise ValueError(
                    f"use_genre={self.use_genre} and use_synopsis={self.use_synopsis} "
                    f"but genre_features={genre_features is not None} and "
                    f"synopsis_embeddings={synopsis_embeddings is not None}"
                )
            content_embed = self.content_encoder(genre_features, synopsis_embeddings)
        elif self.use_genre:
            # Ensure genre_features is provided
            if genre_features is None:
                batch_size = user_ids.size(0)
                device = user_ids.device
                genre_features = torch.zeros(batch_size, self.num_genres, device=device)
            content_embed = self.genre_encoder(genre_features)
        elif self.use_synopsis:
            # Ensure synopsis_embeddings is provided
            if synopsis_embeddings is None:
                batch_size = user_ids.size(0)
                device = user_ids.device
                synopsis_embeddings = torch.zeros(batch_size, self.synopsis_embed_dim, device=device)
            content_embed = self.synopsis_projection(synopsis_embeddings)

        # ===== Fusion =====
        if content_embed is not None:
            if self.use_gated_fusion:
                final_embed, gate = self.gated_fusion(cf_embed, content_embed)
            else:
                # Simple concatenation
                final_embed = torch.cat([cf_embed, content_embed], dim=-1)
                gate = None
        else:
            final_embed = cf_embed
            gate = None

        # ===== Output =====
        output = self.output_fc(final_embed)

        if return_gate:
            return output, gate
        return output

    def predict_scores(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor = None,
        synopsis_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Wrapper that applies sigmoid to get probabilities.

        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            genre_features: Multi-hot genre vectors
            synopsis_embeddings: Synopsis embeddings

        Returns:
            Predicted probabilities (0-1), shape (batch_size,)
        """
        logits = self.forward(user_ids, item_ids, genre_features, synopsis_embeddings)
        return torch.sigmoid(logits).squeeze(-1)

    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device

    def save(self, path: str, epoch: int = None, metrics: dict = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number (optional)
            metrics: Validation metrics (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'num_genres': self.num_genres,
                'use_genre': self.use_genre,
                'use_synopsis': self.use_synopsis,
                'use_gated_fusion': self.use_gated_fusion,
            },
            'epoch': epoch,
            'metrics': metrics or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']

        model = cls(
            num_users=config['num_users'],
            num_items=config['num_items'],
            num_genres=config['num_genres'],
            use_genre=config['use_genre'],
            use_synopsis=config['use_synopsis'],
            use_gated_fusion=config['use_gated_fusion'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        return model, checkpoint


def create_neumf_plus_variant(
    variant: str,
    num_users: int,
    num_items: int,
    num_genres: int,
    **kwargs
) -> NeuMFPlus:
    """
    Factory function to create NeuMF+ variants for ablation study.

    Args:
        variant: One of 'neumf', 'neumf_genre', 'neumf_synopsis',
                'neumf_genre_synopsis', 'neumf_plus'
        num_users: Number of users
        num_items: Number of items
        num_genres: Number of genres
        **kwargs: Additional model parameters

    Returns:
        Configured NeuMFPlus model
    """
    variant_configs = {
        'neumf': {
            'use_genre': False,
            'use_synopsis': False,
            'use_gated_fusion': False,
        },
        'neumf_genre': {
            'use_genre': True,
            'use_synopsis': False,
            'use_gated_fusion': False,
        },
        'neumf_synopsis': {
            'use_genre': False,
            'use_synopsis': True,
            'use_gated_fusion': False,
        },
        'neumf_genre_synopsis': {
            'use_genre': True,
            'use_synopsis': True,
            'use_gated_fusion': False,
        },
        'neumf_plus': {
            'use_genre': True,
            'use_synopsis': True,
            'use_gated_fusion': True,
        },
    }

    if variant not in variant_configs:
        raise ValueError(f"Unknown variant: {variant}")

    config = variant_configs[variant]
    return NeuMFPlus(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        **config,
        **kwargs,
    )


if __name__ == "__main__":
    # Test NeuMF+ model
    print("Testing NeuMF+ model...")

    num_users = 1000
    num_items = 5000
    num_genres = 19
    batch_size = 32

    # Test full model
    print("\n1. Testing full NeuMF+ model...")
    model = NeuMFPlus(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
    )

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    genre_features = torch.randint(0, 2, (batch_size, num_genres)).float()
    synopsis_embeddings = torch.randn(batch_size, 384)

    # Forward pass
    output, gate = model(user_ids, item_ids, genre_features, synopsis_embeddings, return_gate=True)
    print(f"   Output shape: {output.shape}")
    print(f"   Gate shape: {gate.shape}")
    print(f"   Sample gates: {gate[:5].squeeze(-1).detach().numpy()}")

    # Test variants
    print("\n2. Testing ablation variants...")
    for variant in ['neumf', 'neumf_genre', 'neumf_synopsis', 'neumf_genre_synopsis', 'neumf_plus']:
        model_variant = create_neumf_plus_variant(
            variant, num_users, num_items, num_genres
        )
        output = model_variant(user_ids, item_ids, genre_features, synopsis_embeddings)
        print(f"   {variant:25s}: params={sum(p.numel() for p in model_variant.parameters()):,}, output={output.shape}")

    print("\n✓ All tests passed!")
