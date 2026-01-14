#!/usr/bin/env python3
"""
Generate item_genre_features.npy from movies_metadata.csv

This script extracts genre information from TMDB metadata and creates
a multi-hot encoded array that can be used by the recommendation models.

Run this in Google Colab or locally to generate the missing genre features file.
"""

import os
import pickle
import json
import ast
import numpy as np
import pandas as pd
from typing import Dict, List, Set


def parse_genres_from_json(genres_str: str) -> List[str]:
    """Parse genres from JSON or Python list string."""
    if pd.isna(genres_str) or genres_str == "":
        return []

    # Try JSON first (standard format with double quotes)
    try:
        genres = json.loads(genres_str)
        return [g["name"] for g in genres]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try Python literal evaluation (for single-quoted format)
    try:
        genres = ast.literal_eval(genres_str)
        return [g["name"] for g in genres]
    except (ValueError, SyntaxError, KeyError, TypeError):
        return []


def extract_all_genres(movies_df: pd.DataFrame) -> List[str]:
    """Extract all unique genres from the metadata DataFrame."""
    all_genres = set()
    for genres_str in movies_df["genres"].dropna():
        genres = parse_genres_from_json(genres_str)
        all_genres.update(genres)
    return sorted(list(all_genres))


def generate_genre_features(
    movies_metadata_path: str,
    mappings_path: str,
    output_path: str
) -> None:
    """
    Generate item_genre_features.npy from movies metadata.

    Args:
        movies_metadata_path: Path to movies_metadata.csv
        mappings_path: Path to mappings.pkl (contains item_id_map)
        output_path: Path to save the generated genre features
    """
    print("="*70)
    print("GENERATING GENRE FEATURES FROM MOVIES METADATA")
    print("="*70)

    # Load mappings
    print(f"\n[1] Loading mappings from: {mappings_path}")
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    num_items = mappings['num_items']
    num_genres_from_mapping = mappings.get('num_genres', 0)
    genre_names_from_mapping = mappings.get('genre_names', [])
    item_id_map = mappings.get('item_id_map', {})

    print(f"   Items: {num_items:,}")
    print(f"   Genres (from mapping): {num_genres_from_mapping}")
    if genre_names_from_mapping:
        print(f"   Genre names: {genre_names_from_mapping}")

    # Load movies metadata
    print(f"\n[2] Loading movies metadata from: {movies_metadata_path}")
    movies_df = pd.read_csv(movies_metadata_path, low_memory=False)

    # Clean TMDB ID column
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
    movies_df = movies_df[movies_df['id'].notna()]
    movies_df['id'] = movies_df['id'].astype(int)

    print(f"   Movies loaded: {len(movies_df):,}")

    # Extract all unique genres
    print(f"\n[3] Extracting unique genres...")
    all_genres = extract_all_genres(movies_df)
    num_genres = len(all_genres)

    print(f"   Unique genres found: {num_genres}")
    print(f"   Genres: {all_genres}")

    # Use genre names from mapping if available, otherwise use extracted
    if genre_names_from_mapping and len(genre_names_from_mapping) == num_genres:
        genre_list = genre_names_from_mapping
        print(f"\n[4] Using genre names from mapping")
    else:
        genre_list = all_genres
        print(f"\n[4] Using extracted genre names")

    # Create genre features array
    print(f"\n[5] Creating genre features array...")
    print(f"   Shape: ({num_items}, {len(genre_list)})")

    genre_features = np.zeros((num_items, len(genre_list)), dtype=np.float32)

    # Track statistics
    movies_with_genres = 0
    items_matched = 0
    genre_counts = {g: 0 for g in genre_list}

    # Fill genre features for each movie
    print(f"\n[6] Mapping movies to items and encoding genres...")

    for _, row in movies_df.iterrows():
        tmdb_id = row['id']

        # Check if this TMDB ID maps to an internal item ID
        if tmdb_id in item_id_map:
            item_idx = item_id_map[tmdb_id]

            # Parse genres for this movie
            genres = parse_genres_from_json(row['genres'])

            if genres:
                movies_with_genres += 1
                items_matched += 1

                # Set multi-hot encoding for each genre
                for genre in genres:
                    if genre in genre_list:
                        genre_idx = genre_list.index(genre)
                        genre_features[item_idx, genre_idx] = 1.0
                        genre_counts[genre] += 1

    # Print statistics
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nStatistics:")
    print(f"  Movies with genre info: {movies_with_genres:,} / {len(movies_df):,}")
    print(f"  Items matched: {items_matched:,} / {num_items:,}")
    print(f"  Coverage: {items_matched/num_items*100:.1f}%")

    print(f"\nGenre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        pct = count / items_matched * 100 if items_matched > 0 else 0
        print(f"  {genre:20s}: {count:6,} items ({pct:5.1f}%)")

    # Verify the array
    print(f"\nArray shape: {genre_features.shape}")
    print(f"Array dtype: {genre_features.dtype}")
    print(f"Non-zero elements: {np.count_nonzero(genre_features):,}")
    print(f"Sparsity: {(1 - np.count_nonzero(genre_features) / genre_features.size) * 100:.1f}%")

    # Save the file
    print(f"\n[7] Saving genre features to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, genre_features)

    print(f"\n✅ Genre features saved successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

    # Also save a metadata file for reference
    metadata_path = output_path.replace('.npy', '_metadata.pkl')
    metadata = {
        'num_items': num_items,
        'num_genres': len(genre_list),
        'genre_names': genre_list,
        'items_with_genres': items_matched,
        'genre_counts': genre_counts,
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   Metadata: {metadata_path}")


if __name__ == "__main__":
    # For Google Colab usage
    import sys

    # Check if running in Colab
    if 'google.colab' in sys.modules:
        # Colab paths
        BASE_DIR = "/content/drive/MyDrive/NCF-Movie-Recommender"
        movies_metadata_path = os.path.join(BASE_DIR, "datasets", "movies_metadata.csv")
        mappings_path = os.path.join(BASE_DIR, "data", "mappings.pkl")
        output_path = os.path.join(BASE_DIR, "data", "item_genre_features.npy")
    else:
        # Local paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        movies_metadata_path = os.path.join(BASE_DIR, "datasets", "movies_metadata.csv")
        mappings_path = os.path.join(BASE_DIR, "data", "mappings.pkl")
        output_path = os.path.join(BASE_DIR, "data", "item_genre_features.npy")

    # Verify paths exist
    if not os.path.exists(movies_metadata_path):
        print(f"❌ Movies metadata not found: {movies_metadata_path}")
        sys.exit(1)

    if not os.path.exists(mappings_path):
        print(f"❌ Mappings file not found: {mappings_path}")
        sys.exit(1)

    # Generate genre features
    generate_genre_features(
        movies_metadata_path=movies_metadata_path,
        mappings_path=mappings_path,
        output_path=output_path
    )
