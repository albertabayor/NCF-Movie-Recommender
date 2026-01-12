"""
Inference module for NCF models.

This module provides functions for generating top-K recommendations
for users, using trained NCF models.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def recommend_top_k_for_user(
    model: nn.Module,
    user_id: int,
    k: int = 10,
    num_items: int = None,
    user_history: set = None,
    device: str = "cuda",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate top-K recommendations for a single user.

    Args:
        model: Trained NCF model
        user_id: User ID to generate recommendations for
        k: Number of recommendations to generate
        num_items: Total number of items
        user_history: Set of items the user has already interacted with
        device: Device to run inference on
        **kwargs: Additional model args (genre_features, synopsis_embeddings)

    Returns:
        Tuple of (recommended_item_ids, scores)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if num_items is None:
        raise ValueError("num_items must be specified")

    # All item IDs
    all_items = np.arange(num_items)

    # Optionally exclude items the user has already seen
    if user_history is not None:
        candidates = np.setdiff1d(all_items, list(user_history))
    else:
        candidates = all_items

    with torch.no_grad():
        # Create tensors
        user_tensor = torch.LongTensor([user_id] * len(candidates)).to(device)
        item_tensor = torch.LongTensor(candidates).to(device)

        # Handle additional kwargs
        model_kwargs = {}
        if "genre_features" in kwargs and kwargs["genre_features"] is not None:
            genre_features = kwargs["genre_features"][candidates]
            model_kwargs["genre_features"] = torch.FloatTensor(genre_features).to(device)

        if "synopsis_embeddings" in kwargs and kwargs["synopsis_embeddings"] is not None:
            synopsis_embeddings = kwargs["synopsis_embeddings"][candidates]
            model_kwargs["synopsis_embeddings"] = torch.FloatTensor(synopsis_embeddings).to(device)

        # Get scores
        scores = model(user_tensor, item_tensor, **model_kwargs).squeeze(-1)

        # Get top-K
        top_k_scores, top_k_indices = torch.topk(scores, min(k, len(candidates)))

        # Map back to original item IDs
        recommended_items = candidates[top_k_indices.cpu().numpy()]
        recommended_scores = top_k_scores.cpu().numpy()

    return recommended_items, recommended_scores


def recommend_top_k_batch(
    model: nn.Module,
    user_ids: List[int],
    k: int = 10,
    num_items: int = None,
    user_histories: Dict[int, set] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Generate top-K recommendations for multiple users.

    Args:
        model: Trained NCF model
        user_ids: List of user IDs
        k: Number of recommendations per user
        num_items: Total number of items
        user_histories: Dict mapping user_id -> set of interacted items
        device: Device to run inference on
        **kwargs: Additional model args

    Returns:
        Dict mapping user_id -> (recommended_items, scores)
    """
    recommendations = {}

    for user_id in tqdm(user_ids, desc="Generating recommendations"):
        user_history = user_histories.get(user_id) if user_histories else None
        items, scores = recommend_top_k_for_user(
            model=model,
            user_id=user_id,
            k=k,
            num_items=num_items,
            user_history=user_history,
            device=device,
            **kwargs,
        )
        recommendations[user_id] = (items, scores)

    return recommendations


def get_similar_items(
    model: nn.Module,
    item_id: int,
    k: int = 10,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find similar items using item embeddings.

    Args:
        model: Trained NCF model (must have item_embedding attribute)
        item_id: Item ID to find similar items for
        k: Number of similar items to return
        device: Device to run on

    Returns:
        Tuple of (similar_item_ids, similarities)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Get item embeddings
        if hasattr(model, 'item_embedding'):
            item_embeddings = model.item_embedding.weight  # (num_items, embedding_dim)
        else:
            raise ValueError("Model must have item_embedding attribute")

        # Get target item embedding
        target_tensor = torch.LongTensor([item_id]).to(device)
        target_embed = model.item_embedding(target_tensor)  # (1, embedding_dim)

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            target_embed, item_embeddings, dim=-1
        )

        # Get top-K (excluding the item itself)
        similarities[0, item_id] = -1  # Exclude self
        top_k_similarities, top_k_indices = torch.topk(similarities, k)

        similar_items = top_k_indices.cpu().numpy()[0]
        similarity_scores = top_k_similarities.cpu().numpy()[0]

    return similar_items, similarity_scores


def explain_prediction(
    model: nn.Module,
    user_id: int,
    item_id: int,
    num_items: int,
    device: str = "cuda",
    **kwargs,
) -> Dict:
    """
    Explain a prediction by analyzing feature contributions.

    This is a simple heuristic explanation - for more sophisticated
    explanations, consider using SHAP or LIME.

    Args:
        model: Trained NCF model
        user_id: User ID
        item_id: Item ID
        num_items: Total number of items
        device: Device to run on
        **kwargs: Additional model args

    Returns:
        Dict with explanation information
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id]).to(device)
        item_tensor = torch.LongTensor([item_id]).to(device)

        model_kwargs = {}
        if "genre_features" in kwargs and kwargs["genre_features"] is not None:
            model_kwargs["genre_features"] = torch.FloatTensor(
                kwargs["genre_features"][item_id:item_id+1]
            ).to(device)

        if "synopsis_embeddings" in kwargs and kwargs["synopsis_embeddings"] is not None:
            model_kwargs["synopsis_embeddings"] = torch.FloatTensor(
                kwargs["synopsis_embeddings"][item_id:item_id+1]
            ).to(device)

        # Get prediction score
        score = model(user_tensor, item_tensor, **model_kwargs).item()
        probability = torch.sigmoid(torch.tensor(score)).item()

    explanation = {
        "user_id": user_id,
        "item_id": item_id,
        "score": score,
        "probability": probability,
    }

    # Add gate value if using NeuMF+
    if hasattr(model, 'use_gated_fusion') and model.use_gated_fusion:
        with torch.no_grad():
            _, gate = model(user_tensor, item_tensor, **model_kwargs, return_gate=True)
            gate_value = gate.item()
            explanation["gate_value"] = gate_value
            explanation["cf_weight"] = gate_value
            explanation["content_weight"] = 1 - gate_value
            explanation["dominant_signal"] = "collaborative filtering" if gate_value > 0.5 else "content features"

    return explanation


class Recommender:
    """
    High-level recommender class for inference.

    This class wraps a trained model and provides convenient methods
    for generating recommendations.
    """

    def __init__(
        self,
        model: nn.Module,
        num_items: int,
        genre_features: np.ndarray = None,
        synopsis_embeddings: np.ndarray = None,
        user_histories: Dict[int, set] = None,
        device: str = "cuda",
    ):
        """
        Initialize the recommender.

        Args:
            model: Trained NCF model
            num_items: Total number of items
            genre_features: Genre features array (num_items, num_genres)
            synopsis_embeddings: Synopsis embeddings array (num_items, embed_dim)
            user_histories: Dict mapping user_id -> interacted items
            device: Device to run on
        """
        self.model = model
        self.num_items = num_items
        self.genre_features = genre_features
        self.synopsis_embeddings = synopsis_embeddings
        self.user_histories = user_histories or {}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate top-K recommendations for a user.

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude items the user has already seen

        Returns:
            Tuple of (recommended_items, scores)
        """
        user_history = self.user_histories.get(user_id) if exclude_seen else None

        kwargs = {}
        if self.genre_features is not None:
            kwargs["genre_features"] = self.genre_features
        if self.synopsis_embeddings is not None:
            kwargs["synopsis_embeddings"] = self.synopsis_embeddings

        return recommend_top_k_for_user(
            model=self.model,
            user_id=user_id,
            k=k,
            num_items=self.num_items,
            user_history=user_history,
            device=self.device,
            **kwargs,
        )

    def explain(
        self,
        user_id: int,
        item_id: int,
    ) -> Dict:
        """
        Explain a prediction.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Dict with explanation information
        """
        kwargs = {}
        if self.genre_features is not None:
            kwargs["genre_features"] = self.genre_features
        if self.synopsis_embeddings is not None:
            kwargs["synopsis_embeddings"] = self.synopsis_embeddings

        return explain_prediction(
            model=self.model,
            user_id=user_id,
            item_id=item_id,
            num_items=self.num_items,
            device=self.device,
            **kwargs,
        )


if __name__ == "__main__":
    print("Inference module for NCF models")
    print("Import this module and use the Recommender class for inference")
