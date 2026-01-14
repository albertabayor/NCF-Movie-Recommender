#!/usr/bin/env python3
"""
Inference script for trained NCF/NeuMF+ models.

Supports:
- Loading trained models from checkpoint
- Predicting scores for user-item pairs
- Recommending top-K items for a user
- Various input options (CF only, genre, synopsis, or both)

Usage:
    # Load model
    model, _ = NeuMFPlus.load('experiments/trained_models/NeuMFPlus_best.pt')

    # Predict single user-item
    score = predict_score(model, user_id=123, item_id=456)

    # Recommend top-K items
    recommendations = recommend(model, user_id=123, k=10)
"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


def load_model(
    checkpoint_path: str,
    device: str = 'cpu'
) -> Tuple:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    from src.models.neumf_plus import NeuMFPlus

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Use NeuMFPlus.load() method
    model, checkpoint = NeuMFPlus.load(checkpoint_path, device=device)
    model.eval()

    print(f"✅ Model loaded from: {checkpoint_path}")
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"   use_genre: {config.get('use_genre')}")
        print(f"   use_synopsis: {config.get('use_synopsis')}")
        print(f"   use_gated_fusion: {config.get('use_gated_fusion')}")

    return model, checkpoint


def load_mappings(data_dir: str = 'data') -> Dict:
    """
    Load mappings and metadata.

    Args:
        data_dir: Path to data directory

    Returns:
        Dict with mappings and metadata
    """
    mappings_path = os.path.join(data_dir, 'mappings.pkl')

    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    # Load movie metadata for display
    movies_path = os.path.join(data_dir, 'movies_metadata.csv')
    movies_df = None
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)

    return {
        'mappings': mappings,
        'movies_df': movies_df,
        'num_users': mappings['num_users'],
        'num_items': mappings['num_items'],
        'num_genres': mappings['num_genres'],
        'user_id_map': mappings.get('user_id_map'),
        'item_id_map': mappings.get('item_id_map'),
        'genre_names': mappings.get('genre_names', []),
    }


def load_features(data_dir: str = 'data') -> Dict:
    """
    Load item features (genre, synopsis) for inference.

    Args:
        data_dir: Path to data directory

    Returns:
        Dict with feature arrays
    """
    features = {}

    # Load genre features (item-to-genre mapping)
    genre_path = os.path.join(data_dir, 'item_genre_features.npy')
    if os.path.exists(genre_path):
        features['genre_features'] = np.load(genre_path)
        print(f"✅ Genre features loaded: {features['genre_features'].shape}")

    # Load synopsis embeddings
    synopsis_path = os.path.join(data_dir, 'item_synopsis_embeddings.npy')
    if os.path.exists(synopsis_path):
        features['synopsis_embeddings'] = np.load(synopsis_path)
        print(f"✅ Synopsis embeddings loaded: {features['synopsis_embeddings'].shape}")

    return features


def encode_genre(
    genres: Union[str, List[str]],
    genre_names: List[str],
    num_genres: int = None,
) -> np.ndarray:
    """
    Encode genres to multi-hot vector.

    Args:
        genres: Genre string (comma-separated) or list of genres
        genre_names: List of all genre names in correct order
        num_genres: Number of genres (used if genre_names is empty)

    Returns:
        Multi-hot genre vector (num_genres,)

    Examples:
        >>> encode_genre("Action,Comedy", genre_names)
        array([1., 0., 0., 1., 0., ...])

        >>> encode_genre(["Action", "Comedy"], genre_names)
        array([1., 0., 0., 1., 0., ...])
    """
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(',')]

    # Use num_genres if genre_names is empty
    if num_genres is not None:
        vector_size = num_genres
    elif genre_names:
        vector_size = len(genre_names)
    else:
        raise ValueError("Either genre_names or num_genres must be provided")

    genre_vector = np.zeros(vector_size, dtype=np.float32)

    # Only encode if we have genre_names
    if genre_names:
        for genre in genres:
            if genre in genre_names:
                idx = genre_names.index(genre)
                genre_vector[idx] = 1.0

    return genre_vector


def encode_synopsis(
    synopsis: str,
    sbert_model=None,
) -> np.ndarray:
    """
    Encode synopsis text to embedding vector.

    Args:
        synopsis: Movie synopsis text
        sbert_model: Sentence-BERT model (will load if None)

    Returns:
        Synopsis embedding vector (384,)

    Examples:
        >>> encode_synopsis("A group of astronauts discover...")
        array([0.123, -0.456, ...])
    """
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    embedding = sbert_model.encode(synopsis, show_progress_bar=False)
    return np.array(embedding, dtype=np.float32)


def predict_score(
    model: torch.nn.Module,
    user_id: int,
    item_id: int,
    genre_vector: Optional[np.ndarray] = None,
    synopsis_embedding: Optional[np.ndarray] = None,
    device: str = 'cpu',
) -> float:
    """
    Predict score for a user-item pair.

    Args:
        model: Trained NCF/NeuMF+ model
        user_id: User ID (already remapped to 0..num_users-1)
        item_id: Item ID (already remapped to 0..num_items-1)
        genre_vector: Genre features (num_genres,) - Optional
        synopsis_embedding: Synopsis embedding (384,) - Optional
        device: Device to run on

    Returns:
        Predicted score (0-1 probability)

    Examples:
        >>> score = predict_score(model, user_id=123, item_id=456)
        >>> print(f"Score: {score:.4f}")
        Score: 0.8532

        >>> # With genre only
        >>> score = predict_score(model, user_id=123, item_id=456,
        ...                      genre_vector=genre_features[456])

        >>> # With genre + synopsis
        >>> score = predict_score(model, user_id=123, item_id=456,
        ...                      genre_vector=genre_features[456],
        ...                      synopsis_embedding=synopsis_embeddings[456])
    """
    model.eval()

    # Prepare inputs
    user_tensor = torch.LongTensor([user_id]).to(device)
    item_tensor = torch.LongTensor([item_id]).to(device)

    kwargs = {}
    if genre_vector is not None:
        kwargs['genre_features'] = torch.FloatTensor([genre_vector]).to(device)
    if synopsis_embedding is not None:
        kwargs['synopsis_embeddings'] = torch.FloatTensor([synopsis_embedding]).to(device)

    # Predict
    with torch.no_grad():
        logits = model(user_tensor, item_tensor, **kwargs)
        score = torch.sigmoid(logits).squeeze(-1).item()

    return score


def recommend(
    model: torch.nn.Module,
    user_id: int,
    k: int = 10,
    item_genre_features: Optional[np.ndarray] = None,
    item_synopsis_embeddings: Optional[np.ndarray] = None,
    seen_items: Optional[List[int]] = None,
    device: str = 'cpu',
) -> List[Dict]:
    """
    Recommend top-K items for a user.

    Args:
        model: Trained NCF/NeuMF+ model
        user_id: User ID
        k: Number of recommendations
        item_genre_features: Genre features for all items (num_items, num_genres)
        item_synopsis_embeddings: Synopsis embeddings for all items (num_items, 384)
        seen_items: List of items the user has already seen (will be excluded)
        device: Device to run on

    Returns:
        List of dicts with item_id, score, and optional metadata

    Examples:
        >>> recommendations = recommend(model, user_id=123, k=10,
        ...                            item_genre_features=genre_feats,
        ...                            item_synopsis_embeddings=synopsis_feats)
        >>> for rec in recommendations:
        ...     print(f"Item {rec['item_id']}: {rec['score']:.4f}")
    """
    model.eval()
    num_items = model.num_items

    # Exclude seen items
    candidate_items = list(range(num_items))
    if seen_items is not None:
        candidate_items = [item for item in candidate_items if item not in seen_items]

    # Prepare batch inputs
    user_tensor = torch.LongTensor([user_id] * len(candidate_items)).to(device)
    item_tensor = torch.LongTensor(candidate_items).to(device)

    kwargs = {}
    if item_genre_features is not None:
        kwargs['genre_features'] = torch.FloatTensor(item_genre_features[candidate_items]).to(device)
    if item_synopsis_embeddings is not None:
        kwargs['synopsis_embeddings'] = torch.FloatTensor(item_synopsis_embeddings[candidate_items]).to(device)

    # Predict scores
    with torch.no_grad():
        logits = model(user_tensor, item_tensor, **kwargs)
        scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

    # Sort by score (descending) and get top-k
    top_indices = np.argsort(scores)[::-1][:k]

    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'item_id': int(candidate_items[idx]),
            'score': float(scores[idx]),
            'rank': int(idx) + 1,
        })

    return recommendations


def recommend_for_new_movie(
    model: torch.nn.Module,
    user_id: int,
    movie_genres: Union[str, List[str]],
    movie_synopsis: Optional[str] = None,
    k: int = 1,
    genre_names: List[str] = None,
    num_genres: int = None,
    sbert_model=None,
    item_genre_features: Optional[np.ndarray] = None,
    item_synopsis_embeddings: Optional[np.ndarray] = None,
    device: str = 'cpu',
) -> Dict:
    """
    Predict score for a NEW movie (not in training set).

    Useful for cold-start scenario or predicting for new content.

    Args:
        model: Trained model
        user_id: User ID
        movie_genres: Genres (string or list)
        movie_synopsis: Synopsis text (optional)
        k: Number of recommendations (for this specific movie)
        genre_names: List of all genre names
        sbert_model: Sentence-BERT model (for encoding synopsis)
        item_genre_features: Genre features (for reference)
        item_synopsis_embeddings: Synopsis embeddings (for reference)
        device: Device to run on

    Returns:
        Dict with predicted score and info

    Examples:
        >>> result = recommend_for_new_movie(
        ...     model, user_id=123,
        ...     movie_genres="Action,Sci-Fi",
        ...     movie_synopsis="A group of astronauts...",
        ...     genre_names=genre_names
        ... )
        >>> print(f"Predicted score: {result['score']:.4f}")
    """
    # Encode features
    genre_vector = encode_genre(movie_genres, genre_names, num_genres=num_genres)

    # Handle synopsis embedding
    synopsis_embedding = None
    if movie_synopsis is not None:
        synopsis_embedding = encode_synopsis(movie_synopsis, sbert_model)
    elif item_synopsis_embeddings is not None:
        # Use average synopsis embedding for reference
        synopsis_embedding = item_synopsis_embeddings.mean(axis=0)
    elif hasattr(model, 'synopsis_embed_dim'):
        # Model requires synopsis - use zero vector if not provided
        synopsis_embedding = np.zeros(model.synopsis_embed_dim, dtype=np.float32)
    else:
        # Fallback: assume 384-dim synopsis (SBERT default)
        synopsis_embedding = np.zeros(384, dtype=np.float32)

    # For new movie, we need a placeholder item_id
    # Use the model's num_items as a temporary new item
    temp_item_id = model.num_items - 1  # Use last valid item as reference

    score = predict_score(
        model, user_id, temp_item_id,
        genre_vector=genre_vector,
        synopsis_embedding=synopsis_embedding,
        device=device
    )

    return {
        'user_id': user_id,
        'genres': movie_genres if isinstance(movie_genres, str) else ','.join(movie_genres),
        'synopsis_length': len(movie_synopsis) if movie_synopsis else 0,
        'predicted_score': score,
    }


# ============================================================================
# CLI / Interactive Functions
# ============================================================================

def interactive_demo():
    """
    Interactive demo for model inference.
    """
    print("="*70)
    print("NCF/NeuMF+ INFERENCE DEMO")
    print("="*70)

    # 1. Load model
    print("\n[1] Loading model...")
    checkpoint_path = input("Enter checkpoint path (default: experiments/trained_models/NeuMFPlus_genre_synopsis_best.pt): ").strip()
    if not checkpoint_path:
        checkpoint_path = 'experiments/trained_models/NeuMFPlus_genre_synopsis_best.pt'

    try:
        model, checkpoint = load_model(checkpoint_path, device='cpu')
        config = checkpoint['model_config']
        use_genre = config.get('use_genre', False)
        use_synopsis = config.get('use_synopsis', False)
        num_genres = config.get('num_genres', 0)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load mappings
    print("\n[2] Loading mappings...")
    data = load_mappings()
    genre_names = data.get('genre_names', [])
    num_items = data['num_items']
    print(f"   Users: {data['num_users']:,}")
    print(f"   Items: {num_items:,}")
    print(f"   Genres: {len(genre_names)} (model expects {num_genres})")

    # 3. Load features
    print("\n[3] Loading features...")
    features = load_features()

    # 4. Interactive prediction
    print("\n" + "="*70)
    print("READY FOR PREDICTION!")
    print("="*70)
    print(f"\nModel config: use_genre={use_genre}, use_synopsis={use_synopsis}")
    print(f"\nOptions:")
    print("  1. Predict score for user-item pair")
    print("  2. Recommend top-K items for user")
    print("  3. Predict for new movie (cold-start)")
    print("  4. Batch predict multiple users")
    print("  5. Exit")

    while True:
        choice = input("\nSelect option (1-5): ").strip()

        if choice == '1':
            # Single prediction
            print("\n--- Predict Score ---")
            user_id = int(input("User ID: ") or 0)
            item_id = int(input("Item ID: ") or 0)

            genre = None
            synopsis = None

            if use_genre:
                if features.get('genre_features') is not None:
                    genre = features['genre_features'][item_id]
                else:
                    genres_input = input("Genres (comma-separated, or press Enter for zero vector): ")
                    if genres_input:
                        genre = encode_genre(genres_input, genre_names, num_genres=num_genres)
                    else:
                        # Use zero vector when model requires genre but not provided
                        genre = np.zeros(num_genres, dtype=np.float32)
                        print(f"  (Using zero vector for genre)")

            if use_synopsis:
                if features.get('synopsis_embeddings') is not None:
                    synopsis = features['synopsis_embeddings'][item_id]
                else:
                    synopsis_input = input("Synopsis (or press Enter for zero vector): ")
                    if synopsis_input:
                        synopsis = encode_synopsis(synopsis_input)
                    else:
                        # Use zero vector when model requires synopsis but not provided
                        synopsis = np.zeros(384, dtype=np.float32)
                        print(f"  (Using zero vector for synopsis)")

            score = predict_score(model, user_id, item_id, genre, synopsis)
            print(f"\n✅ Predicted score: {score:.4f}")

        elif choice == '2':
            # Recommend top-K
            print("\n--- Top-K Recommendations ---")
            user_id = int(input("User ID: ") or 0)
            k = int(input("K (number of recommendations): ") or 10)

            seen_items = None
            use_seen = input("Exclude seen items? (y/n, default=n): ").strip().lower()
            if use_seen == 'y':
                seen_input = input("Enter seen item IDs (comma-separated): ")
                seen_items = [int(x.strip()) for x in seen_input.split(',') if x.strip()]

            recommendations = recommend(
                model, user_id, k,
                item_genre_features=features.get('genre_features'),
                item_synopsis_embeddings=features.get('synopsis_embeddings'),
                seen_items=seen_items
            )

            print(f"\n✅ Top {k} recommendations for user {user_id}:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. Item {rec['item_id']}: {rec['score']:.4f}")

        elif choice == '3':
            # New movie prediction
            print("\n--- Predict for New Movie ---")
            user_id = int(input("User ID: ") or 0)
            genres_input = input("Genres (comma-separated): ")
            synopsis_input = input("Synopsis (optional): ")

            score = recommend_for_new_movie(
                model, user_id,
                movie_genres=genres_input,
                movie_synopsis=synopsis_input if synopsis_input else None,
                genre_names=genre_names,
                num_genres=num_genres
            )

            print(f"\n✅ Predicted score: {score['predicted_score']:.4f}")

        elif choice == '4':
            # Batch prediction
            print("\n--- Batch Prediction ---")
            print("Feature coming soon!")

        elif choice == '5':
            print("\n👋 Goodbye!")
            break

        else:
            print("Invalid option. Please select 1-5.")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        interactive_demo()
    else:
        print("="*70)
        print("NCF/NeuMF+ INFERENCE MODULE")
        print("="*70)
        print("\nUsage:")
        print("  python inference.py")
        print("\nOr import in your code:")
        print("  from inference import load_model, predict_score, recommend")
        print("\nFor interactive demo:")
        print("  python inference.py --demo")
        print("="*70)
