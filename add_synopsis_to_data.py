#!/usr/bin/env python3
"""
Add synopsis embeddings to processed train/val/test data.

This script:
1. Loads existing processed data (train.pkl, val.pkl, test.pkl)
2. Loads item-to-synopsis mapping
3. Adds synopsis_features column to each dataframe
4. Saves updated data

Usage:
    python add_synopsis_to_data.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def add_synopsis_to_df(df, item_synopsis_embeddings, num_items):
    """Add synopsis features to dataframe."""
    synopsis_features_list = []

    for movie_id in tqdm(df['movieId'].values, desc="Adding synopsis features"):
        if 0 <= movie_id < num_items:
            synopsis_features_list.append(item_synopsis_embeddings[movie_id])
        else:
            # Invalid movieId, use zeros
            synopsis_features_list.append(np.zeros(item_synopsis_embeddings.shape[1], dtype=np.float32))

    df['synopsis_features'] = synopsis_features_list
    return df


def main():
    print("=" * 70)
    print("ADDING SYNOPSIS EMBEDDINGS TO PROCESSED DATA")
    print("=" * 70)

    # Load item synopsis embeddings
    print("\n[1/5] Loading item synopsis embeddings...")
    synopsis_path = Path("data/item_synopsis_embeddings.npy")
    metadata_path = Path("data/item_synopsis_metadata.pkl")

    if not synopsis_path.exists():
        print(f"❌ Item synopsis embeddings not found!")
        print("Please run map_synopsis_to_movieid.py first.")
        return

    item_synopsis_embeddings = np.load(synopsis_path)
    print(f"  Shape: {item_synopsis_embeddings.shape}")

    with open(metadata_path, 'rb') as f:
        synopsis_metadata = pickle.load(f)

    num_items = synopsis_metadata['num_items']
    embedding_dim = synopsis_metadata['embedding_dim']
    print(f"  Items: {num_items:,}")
    print(f"  Embedding dim: {embedding_dim}")

    # Load existing processed data
    print("\n[2/5] Loading processed data...")
    train_df = pd.read_pickle('data/train.pkl')
    val_df = pd.read_pickle('data/val.pkl')
    test_df = pd.read_pickle('data/test.pkl')

    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Val: {len(val_df):,} ratings")
    print(f"  Test: {len(test_df):,} ratings")

    # Check if synopsis_features already exists
    if 'synopsis_features' in train_df.columns:
        print("\n  ⚠️  synopsis_features column already exists!")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("  Aborted.")
            return

    # Add synopsis features to train data
    print("\n[3/5] Adding synopsis features to train data...")
    train_df = add_synopsis_to_df(train_df, item_synopsis_embeddings, num_items)
    print(f"  Done! synopsis_features shape: {train_df['synopsis_features'].iloc[0].shape}")

    # Add synopsis features to validation data
    print("\n[4/5] Adding synopsis features to validation data...")
    val_df = add_synopsis_to_df(val_df, item_synopsis_embeddings, num_items)
    print(f"  Done! synopsis_features shape: {val_df['synopsis_features'].iloc[0].shape}")

    # Add synopsis features to test data
    print("\n[5/5] Adding synopsis features to test data...")
    test_df = add_synopsis_to_df(test_df, item_synopsis_embeddings, num_items)
    print(f"  Done! synopsis_features shape: {test_df['synopsis_features'].iloc[0].shape}")

    # Save updated data
    print("\n" + "=" * 70)
    print("SAVING UPDATED DATA")
    print("=" * 70)

    # Backup original files first
    print("\nBacking up original files...")
    import shutil
    for file in ['train.pkl', 'val.pkl', 'test.pkl']:
        src = Path(f'data/{file}')
        dst = Path(f'data/{file}.backup')
        if not dst.exists():
            shutil.copy(src, dst)
            print(f"  ✓ Backed up {file}")

    # Save updated data
    print("\nSaving updated data with synopsis features...")
    train_df.to_pickle('data/train.pkl')
    val_df.to_pickle('data/val.pkl')
    test_df.to_pickle('data/test.pkl')
    print("  ✓ Saved all files")

    # Update mappings.pkl to include synopsis info
    print("\nUpdating mappings.pkl...")
    with open('data/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)

    mappings['synopsis_embed_dim'] = embedding_dim
    mappings['synopsis_coverage'] = synopsis_metadata['coverage']

    with open('data/mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    print("  ✓ Updated mappings.pkl")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nUpdated files:")
    print(f"  - data/train.pkl (with synopsis_features)")
    print(f"  - data/val.pkl (with synopsis_features)")
    print(f"  - data/test.pkl (with synopsis_features)")
    print(f"  - data/mappings.pkl (updated)")
    print(f"\nOriginal files backed up as: *.pkl.backup")
    print(f"\nNext: Train NeuMF+ with use_synopsis=True")
    print("=" * 70)


if __name__ == "__main__":
    main()
