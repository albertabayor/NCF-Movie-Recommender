#!/usr/bin/env python3
"""
Map synopsis embeddings to MovieLens movieIds.

This script:
1. Loads synopsis embeddings (indexed by TMDB ID)
2. Loads TMDB to movieId mapping from links.csv
3. Creates item-to-synopsis mapping indexed by movieId
4. Saves mapping for use in NeuMF+ training

Usage:
    python map_synopsis_to_movieid.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("=" * 70)
    print("MAPPING SYNOPSIS EMBEDDINGS TO MOVIELENS movieIds")
    print("=" * 70)

    # Load synopsis embeddings
    print("\n[1/5] Loading synopsis embeddings...")
    synopsis_path = Path("data/synopsis_embeddings.npy")
    metadata_path = Path("data/synopsis_metadata.pkl")
    mapping_path = Path("data/tmdb_to_movieid.pkl")

    if not synopsis_path.exists():
        print(f"❌ Synopsis embeddings not found at: {synopsis_path}")
        print("Please run extract_synopsis_embeddings.py first!")
        return

    # Load embeddings
    synopsis_embeddings = np.load(synopsis_path)
    print(f"  Loaded shape: {synopsis_embeddings.shape}")

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    tmdb_ids = metadata['tmdb_ids']
    num_synopsis = len(tmdb_ids)
    embedding_dim = metadata['embedding_dim']

    print(f"  TMDB IDs: {num_synopsis:,}")
    print(f"  Embedding dim: {embedding_dim}")

    # Load TMDB to movieId mapping
    print("\n[2/5] Loading TMDB to movieId mapping...")
    links_path = Path("datasets/links.csv")

    if not links_path.exists():
        print(f"❌ links.csv not found at: {links_path}")
        return

    links = pd.read_csv(links_path)
    print(f"  Loaded {len(links):,} links")

    # Create TMDB ID to movieId mapping
    tmdb_to_movieid = dict(zip(links['tmdbId'].dropna(), links['movieId']))
    print(f"  TMDB → movieId mappings: {len(tmdb_to_movieid):,}")

    # Load processed data to get num_items
    print("\n[3/5] Loading processed data...")
    mappings_path = Path("data/mappings.pkl")

    if not mappings_path.exists():
        print(f"❌ mappings.pkl not found. Run preprocessing first!")
        return

    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    num_items = mappings['num_items']
    print(f"  num_items: {num_items:,}")

    # Create item-to-synopsis mapping
    print("\n[4/5] Creating item-to-synopsis mapping...")
    # Initialize with zeros (for items without synopsis)
    item_synopsis_embeddings = np.zeros((num_items, embedding_dim), dtype=np.float32)

    mapped_count = 0
    for i, tmdb_id in enumerate(tmdb_ids):
        if tmdb_id in tmdb_to_movieid:
            movie_id = tmdb_to_movieid[tmdb_id]
            if movie_id < num_items:  # Valid movieId
                item_synopsis_embeddings[movie_id] = synopsis_embeddings[i]
                mapped_count += 1

    print(f"  Mapped {mapped_count:,} synopsis embeddings to movieIds")
    print(f"  Coverage: {mapped_count / num_items * 100:.1f}% of {num_items:,} items")

    # Save mapping
    print("\n[5/5] Saving item-to-synopsis mapping...")
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings indexed by movieId
    item_synopsis_path = output_dir / "item_synopsis_embeddings.npy"
    np.save(item_synopsis_path, item_synopsis_embeddings)
    print(f"  ✓ Saved to: {item_synopsis_path}")

    # Save metadata
    synopsis_metadata = {
        'num_items': num_items,
        'num_items_with_synopsis': mapped_count,
        'embedding_dim': embedding_dim,
        'coverage': mapped_count / num_items,
    }

    item_metadata_path = output_dir / "item_synopsis_metadata.pkl"
    with open(item_metadata_path, 'wb') as f:
        pickle.dump(synopsis_metadata, f)
    print(f"  ✓ Saved metadata to: {item_metadata_path}")

    print("\n" + "=" * 70)
    print("MAPPING COMPLETE!")
    print("=" * 70)
    print(f"\nCreated files:")
    print(f"  - data/item_synopsis_embeddings.npy ({num_items} x {embedding_dim})")
    print(f"  - data/item_synopsis_metadata.pkl")
    print(f"\nNext steps:")
    print(f"  1. Update preprocessing to load synopsis embeddings")
    print(f"  2. Train NeuMF+ with use_synopsis=True")
    print("=" * 70)


if __name__ == "__main__":
    main()
