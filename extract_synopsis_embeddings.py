#!/usr/bin/env python3
"""
Extract synopsis embeddings using Sentence-BERT.

This script:
1. Loads movie overview from metadata
2. Converts text to 384-dim embeddings using Sentence-BERT
3. Saves embeddings to disk for NeuMF+ training

Usage:
    python extract_synopsis_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_model():
    """Load Sentence-BERT model for embedding extraction."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer

    print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    return model, device


def extract_embeddings(model, texts, device, batch_size=128):
    """
    Extract embeddings from text using Sentence-BERT.

    Args:
        model: Sentence-BERT model
        texts: List of text strings (movie overviews)
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (num_texts, 384)
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i:i+batch_size]

        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )

        embeddings.append(batch_embeddings)

    return np.vstack(embeddings).astype(np.float32)


def main():
    """Extract synopsis embeddings for all movies."""
    print("=" * 70)
    print("SYNOPSIS EMBEDDING EXTRACTION")
    print("=" * 70)

    # Load Sentence-BERT model
    model, device = load_model()

    # Load movies metadata
    print("\n[1/4] Loading movies metadata...")
    metadata_path = Path("datasets/movies_metadata.csv")

    if not metadata_path.exists():
        print(f"❌ Metadata not found at: {metadata_path}")
        print("Please ensure datasets/ folder exists with movies_metadata.csv")
        return

    # Read metadata with low_memory=False for mixed types
    print(f"Reading {metadata_path}...")
    df = pd.read_csv(metadata_path, low_memory=False)

    # Extract relevant columns
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype(int)

    print(f"  Loaded {len(df):,} movies")

    # Load links to get movieId mapping
    print("\n[2/4] Loading links for movieId mapping...")
    links_path = Path("datasets/links.csv")

    if links_path.exists():
        links = pd.read_csv(links_path)
        print(f"  Loaded {len(links):,} links")

        # Create TMDB ID to movieId mapping
        tmdb_to_movie = dict(zip(links['tmdbId'].dropna(), links['movieId']))
        print(f"  Created mapping: {len(tmdb_to_movie):,} TMDB → movieId")
    else:
        print("  ⚠️  links.csv not found, using TMDB IDs as movie IDs")
        tmdb_to_movie = None

    # Prepare overviews
    print("\n[3/4] Preparing movie overviews...")
    df['overview'] = df['overview'].fillna('')

    # Filter movies with non-empty overviews
    df_has_overview = df[df['overview'].str.len() > 0]
    print(f"  Movies with overview: {len(df_has_overview):,} / {len(df):,}")

    # Create overview list (sorted by TMDB ID for consistency)
    df_sorted = df_has_overview.sort_values('id')
    overviews = df_sorted['overview'].tolist()
    tmdb_ids = df_sorted['id'].tolist()

    print(f"  Processing {len(overviews):,} overviews")
    print(f"  Overview length (avg): {df_sorted['overview'].str.len().mean():.0f} chars")

    # Extract embeddings
    print("\n[4/4] Extracting embeddings (this takes ~30-60 minutes)...")
    print(f"  Batch size: 128")
    print(f"  Device: {device}")

    embeddings = extract_embeddings(model, overviews, device, batch_size=128)

    print(f"\n  Embeddings shape: {embeddings.shape}")
    print(f"  Embeddings dtype: {embeddings.dtype}")
    print(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Save embeddings with mapping
    print("\n" + "=" * 70)
    print("SAVING EMBEDDINGS")
    print("=" * 70)

    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    output_path = output_dir / "synopsis_embeddings.npy"
    np.save(output_path, embeddings)
    print(f"✓ Saved embeddings to: {output_path}")

    # Save metadata (TMDB IDs mapping)
    import pickle
    metadata = {
        'tmdb_ids': np.array(tmdb_ids, dtype=np.int32),
        'num_movies': len(tmdb_ids),
        'embedding_dim': embeddings.shape[1],
    }

    metadata_path = output_dir / "synopsis_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to: {metadata_path}")

    # Save TMDB to movieId mapping if available
    if tmdb_to_movie is not None:
        mapping_path = output_dir / "tmdb_to_movieid.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(tmdb_to_movie, f)
        print(f"✓ Saved TMDB→movieId mapping to: {mapping_path}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nEmbeddings saved: {embeddings.shape}")
    print(f"\nNext steps:")
    print(f"  1. Run preprocessing to link embeddings to movieIds")
    print(f"  2. Train NeuMF+ with use_synopsis=True")
    print(f"  3. Enjoy improved recommendations!")
    print("=" * 70)


if __name__ == "__main__":
    main()
