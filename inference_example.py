#!/usr/bin/env python3
"""
Example inference script for trained NCF model.

This script demonstrates how to:
1. Load a trained model
2. Get recommendations for a specific user
3. Get similar movies (using item embeddings)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch

from src.models.neumf import NeuMF


def load_trained_model(model_path, mappings_path, device='cuda'):
    """Load a trained model and mappings."""
    print(f"Loading model from {model_path}...")

    # Load mappings
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    num_users = mappings['num_users']
    num_items = mappings['num_items']

    # Load model
    model = NeuMF.load(
        model_path,
        NeuMF,
        num_users=num_users,
        num_items=num_items,
    )
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")

    return model, mappings


def get_user_recommendations(model, user_id, mappings, top_k=10, device='cuda'):
    """
    Get top-K movie recommendations for a specific user.

    Args:
        model: Trained NCF model
        user_id: Internal user ID (0 to num_users-1)
        mappings: Dictionary with user/item mappings
        top_k: Number of recommendations to return
        device: Device to run inference on

    Returns:
        List of (movie_id, score) tuples
    """
    model.eval()
    num_items = mappings['num_items']

    # Get all item IDs
    all_items = np.arange(num_items)

    # Create user tensor
    user_tensor = torch.tensor([user_id] * num_items, dtype=torch.long).to(device)
    item_tensor = torch.tensor(all_items, dtype=torch.long).to(device)

    # Get predictions
    with torch.no_grad():
        scores = model(user_tensor, item_tensor).squeeze()

    # Get top-K items
    top_scores, top_indices = torch.topk(scores, top_k)

    # Convert to original movie IDs
    reverse_item_map = mappings['reverse_item_map']
    recommendations = []
    for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
        original_movie_id = reverse_item_map[int(idx)]
        recommendations.append((original_movie_id, float(score)))

    return recommendations


def get_similar_movies(model, item_id, mappings, top_k=10, device='cuda'):
    """
    Get similar movies using item embeddings.

    Args:
        model: Trained NCF model
        item_id: Original movie ID from MovieLens
        mappings: Dictionary with user/item mappings
        top_k: Number of similar movies to return
        device: Device to run inference on

    Returns:
        List of (movie_id, similarity) tuples
    """
    model.eval()

    # Get item embedding
    if hasattr(model, 'item_embedding_gmf'):
        item_embedding = model.item_embedding_gmf.weight
    elif hasattr(model, 'item_embedding_mlp'):
        item_embedding = model.item_embedding_mlp.weight
    else:
        raise ValueError("Model doesn't have item embeddings")

    # Map original item ID to internal ID
    item_map = mappings['item_map']
    reverse_item_map = mappings['reverse_item_map']

    if item_id not in item_map:
        print(f"Item {item_id} not found in mappings")
        return []

    internal_id = item_map[item_id]

    # Get embedding for the target item
    target_embedding = item_embedding[internal_id].unsqueeze(0)

    # Compute cosine similarity with all items
    with torch.no_grad():
        embeddings = item_embedding.to(device)
        target = target_embedding.to(device)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(target, embeddings, dim=1)

    # Get top-K (excluding the item itself)
    similarity[internal_id] = -1  # Exclude self
    top_similarities, top_indices = torch.topk(similarity, top_k)

    # Convert to original movie IDs
    similar_items = []
    for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        original_id = reverse_item_map[int(idx)]
        similar_items.append((original_id, float(sim)))

    return similar_items


def print_recommendations(recommendations, movies_df=None):
    """Print recommendations in a nice format."""
    print("\n" + "="*60)
    print("TOP RECOMMENDATIONS")
    print("="*60)

    if movies_df is not None:
        # Try to show movie titles if available
        for i, (movie_id, score) in enumerate(recommendations, 1):
            match = movies_df[movies_df['movieId'] == movie_id]
            if not match.empty:
                title = match.iloc[0]['title']
                print(f"{i}. {title} (ID: {movie_id}) - Score: {score:.4f}")
            else:
                print(f"{i}. Movie ID {movie_id} - Score: {score:.4f}")
    else:
        for i, (movie_id, score) in enumerate(recommendations, 1):
            print(f"{i}. Movie ID {movie_id} - Score: {score:.4f}")


def main():
    """Run inference example."""
    print("="*60)
    print("NCF MOVIE RECOMMENDER - INFERENCE EXAMPLE")
    print("="*60)

    # Paths
    model_path = "experiments/trained_models/NeuMF_best.pt"
    mappings_path = "data/mappings.pkl"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please train a model first using: python train_local.py")
        return

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, mappings = load_trained_model(model_path, mappings_path, device)

    # Example 1: Get recommendations for a user
    print("\n" + "="*60)
    print("EXAMPLE 1: User Recommendations")
    print("="*60)

    # Pick a random user
    num_users = mappings['num_users']
    test_user_id = np.random.randint(0, num_users)

    print(f"\nGetting recommendations for User ID: {test_user_id}")
    recommendations = get_user_recommendations(
        model, test_user_id, mappings, top_k=10, device=device
    )

    print_recommendations(recommendations)

    # Example 2: Get similar movies
    print("\n" + "="*60)
    print("EXAMPLE 2: Similar Movies")
    print("="*60)

    # Pick a random movie
    num_items = mappings['num_items']
    test_item_id = mappings['reverse_item_map'][np.random.randint(0, num_items)]

    print(f"\nFinding movies similar to Movie ID: {test_item_id}")
    similar = get_similar_movies(
        model, test_item_id, mappings, top_k=10, device=device
    )

    print(f"\nMovies similar to Movie ID {test_item_id}:")
    for i, (movie_id, sim) in enumerate(similar, 1):
        print(f"{i}. Movie ID {movie_id} - Similarity: {sim:.4f}")

    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
