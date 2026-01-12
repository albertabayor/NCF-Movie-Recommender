"""
Memory-efficient preprocessing pipeline for NCF Movie Recommender.

This module implements a streamlined preprocessing pipeline designed to work
within Colab's RAM limits (~12GB). Key optimizations:
1. Filter sparse users/items BEFORE merging (reduces data size early)
2. Process genres efficiently (skip full merge when possible)
3. Use in-place operations to reduce memory copies
4. Optional mode to skip content features for baseline models

Use run_minimal() for fast preprocessing without content features.
Use run_full() for complete preprocessing with genres.

The pipeline follows the methodology outlined in PROJECT_PLAN.md section 4.1.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from .config import config
from .data_loader import DataLoader, parse_genres, extract_all_genres


class DataPreprocessor:
    """
    Memory-efficient preprocessing pipeline for MovieLens data.

    This class orchestrates the entire preprocessing workflow from raw
    CSV files to train/val/test splits ready for model training.

    Usage:
        # Fast preprocessing (no content features)
        preprocessor = DataPreprocessor()
        preprocessor.run_minimal()

        # Full preprocessing (with genres)
        preprocessor = DataPreprocessor()
        preprocessor.run_full()
    """

    def __init__(self, cfg=None):
        """
        Initialize the preprocessor.

        Args:
            cfg: Configuration object (uses global config if None)
        """
        self.cfg = cfg or config
        self.loader = DataLoader(self.cfg.paths.DATASETS_DIR)

        # Internal state
        self.genre_encoder: Optional[MultiLabelBinarizer] = None
        self.user_map: Optional[Dict[int, int]] = None
        self.item_map: Optional[Dict[int, int]] = None
        self.reverse_user_map: Optional[Dict[int, int]] = None
        self.reverse_item_map: Optional[Dict[int, int]] = None

        # Statistics
        self.stats_before = {}
        self.stats_after = {}

    def run(self) -> None:
        """
        Execute the full preprocessing pipeline (with genres).

        Alias for run_full() for backward compatibility.
        """
        self.run_full()

    def run_full(self) -> None:
        """
        Execute the full preprocessing pipeline with content features.

        This processes genres from metadata but uses memory optimizations.
        """
        print("=" * 60)
        print("NCF MOVIE RECOMMENDER - PREPROCESSING (FULL)")
        print("=" * 60)

        # Step 1: Load and filter ratings FIRST (before merge)
        print("\n[Step 1/8] Loading and filtering ratings...")
        ratings = self.loader.load_ratings()

        # Filter sparse users/items early to reduce data size
        initial_users = ratings["userId"].nunique()
        initial_items = ratings["movieId"].nunique()

        user_counts = ratings["userId"].value_counts()
        item_counts = ratings["movieId"].value_counts()

        valid_users = set(user_counts[user_counts >= self.cfg.data.MIN_USER_RATINGS].index)
        valid_items = set(item_counts[item_counts >= self.cfg.data.MIN_ITEM_RATINGS].index)

        ratings = ratings[ratings["userId"].isin(valid_users)]
        ratings = ratings[ratings["movieId"].isin(valid_items)]

        # Free memory
        del user_counts, item_counts, valid_users, valid_items

        print(f"  Users: {initial_users:,} -> {ratings['userId'].nunique():,}")
        print(f"  Items: {initial_items:,} -> {ratings['movieId'].nunique():,}")
        print(f"  Ratings: {len(ratings):,}")

        # Step 2: Load metadata
        print("\n[Step 2/8] Loading metadata...")
        metadata = self.loader.load_movies_metadata()
        links = self.loader.load_links()

        # Step 3: Create user/item mappings
        print("\n[Step 3/8] Creating mappings...")
        all_users = sorted(ratings["userId"].unique())
        all_items = sorted(ratings["movieId"].unique())

        self.user_map = {old: new for new, old in enumerate(all_users)}
        self.item_map = {old: new for new, old in enumerate(all_items)}
        self.reverse_user_map = {new: old for old, new in self.user_map.items()}
        self.reverse_item_map = {new: old for old, new in self.item_map.items()}

        print(f"  Created mappings: {len(self.user_map):,} users, {len(self.item_map):,} items")

        # Step 4: Process genres efficiently (in-memory, no merge needed)
        print("\n[Step 4/8] Processing genres...")
        self.genre_encoder = MultiLabelBinarizer()

        # Parse genres from metadata (only for movies we have)
        movie_metadata = {}

        # Build mapping from movieId to genres
        # First, get movieId -> tmdbId mapping from links
        movie_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))

        # Then, get tmdbId -> genres from metadata
        tmdb_to_genres = {}
        for _, row in metadata.iterrows():
            tmdb_id = row["id"]
            if pd.notna(tmdb_id):
                genres = parse_genres(row["genres"])
                tmdb_to_genres[int(tmdb_id)] = genres

        # Map movieId -> genres
        num_items = len(self.item_map)
        all_genre_lists = []

        for item_id in range(num_items):
            original_id = self.reverse_item_map[item_id]
            if original_id in movie_to_tmdb:
                tmdb_id = movie_to_tmdb[original_id]
                if pd.notna(tmdb_id) and int(tmdb_id) in tmdb_to_genres:
                    all_genre_lists.append(tmdb_to_genres[int(tmdb_id)])
                else:
                    all_genre_lists.append([])
            else:
                all_genre_lists.append([])

        # Fit genre encoder
        self.genre_encoder.fit(all_genre_lists)

        print(f"  Created genre encoder: {len(self.genre_encoder.classes_)} genres")
        print(f"    Genres: {', '.join(self.genre_encoder.classes_)}")

        # Step 5: Apply mappings and clean data
        print("\n[Step 5/8] Applying mappings and cleaning...")
        ratings["userId"] = ratings["userId"].map(self.user_map)
        ratings["movieId"] = ratings["movieId"].map(self.item_map)

        # Remove duplicates
        initial_count = len(ratings)
        dup_count = ratings.duplicated(subset=["userId", "movieId"]).sum()
        if dup_count > 0:
            ratings = ratings.drop_duplicates(subset=["userId", "movieId"])
            print(f"  Removed {dup_count:,} duplicates")

        # Sort by timestamp
        ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
        ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

        print(f"  After cleaning: {len(ratings):,} ratings")

        # Step 6: Time-based split
        print("\n[Step 6/8] Creating train/val/test split...")
        train_rows = []
        val_rows = []
        test_rows = []

        for user_id, user_df in tqdm(ratings.groupby("userId"), desc="  Splitting users"):
            n = len(user_df)
            train_end = int(n * self.cfg.data.TRAIN_RATIO)
            val_end = train_end + int(n * self.cfg.data.VAL_RATIO)

            train_rows.append(user_df.iloc[:train_end])
            val_rows.append(user_df.iloc[train_end:val_end])
            test_rows.append(user_df.iloc[val_end:])

        train_df = pd.concat(train_rows).reset_index(drop=True)
        val_df = pd.concat(val_rows).reset_index(drop=True)
        test_df = pd.concat(test_rows).reset_index(drop=True)

        print(f"  Train: {len(train_df):,} ratings ({len(train_df)/len(ratings)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} ratings ({len(val_df)/len(ratings)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} ratings ({len(test_df)/len(ratings)*100:.1f}%)")

        # Free memory
        del ratings, train_rows, val_rows, test_rows

        # Step 7: Encode genres
        print("\n[Step 7/8] Encoding genres...")
        train_df["genre_features"] = list(self.genre_encoder.transform(
            [all_genre_lists[item_id] for item_id in train_df["movieId"]]
        ))
        val_df["genre_features"] = list(self.genre_encoder.transform(
            [all_genre_lists[item_id] for item_id in val_df["movieId"]]
        ))
        test_df["genre_features"] = list(self.genre_encoder.transform(
            [all_genre_lists[item_id] for item_id in test_df["movieId"]]
        ))

        # Step 8: Save
        print("\n[Step 8/8] Saving processed data...")
        self._save_data(train_df, val_df, test_df)

        # Create cold-start set
        print("\n[Bonus] Creating cold-start evaluation set...")
        cold_start_df = self._create_cold_start_set(train_df, test_df)
        self._save_cold_start(cold_start_df)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        self._print_summary(train_df, val_df, test_df)

    def run_minimal(self) -> None:
        """
        Execute minimal preprocessing without content features.

        This is the fastest option and uses minimal RAM:
        - No metadata merge
        - No genre parsing
        - Only user-item interactions

        Use this for baseline CF models (GMF, MLP, NeuMF).
        """
        print("=" * 60)
        print("NCF MOVIE RECOMMENDER - PREPROCESSING (MINIMAL)")
        print("=" * 60)

        # Step 1: Load and filter ratings
        print("\n[Step 1/5] Loading and filtering ratings...")
        ratings = self.loader.load_ratings()

        # Filter sparse users/items early
        initial_users = ratings["userId"].nunique()
        initial_items = ratings["movieId"].nunique()

        user_counts = ratings["userId"].value_counts()
        item_counts = ratings["movieId"].value_counts()

        valid_users = set(user_counts[user_counts >= self.cfg.data.MIN_USER_RATINGS].index)
        valid_items = set(item_counts[item_counts >= self.cfg.data.MIN_ITEM_RATINGS].index)

        ratings = ratings[ratings["userId"].isin(valid_users)]
        ratings = ratings[ratings["movieId"].isin(valid_items)]

        print(f"  Users: {initial_users:,} -> {ratings['userId'].nunique():,}")
        print(f"  Items: {initial_items:,} -> {ratings['movieId'].nunique():,}")
        print(f"  Ratings: {len(ratings):,}")

        # Step 2: Create mappings
        print("\n[Step 2/5] Creating mappings...")
        all_users = sorted(ratings["userId"].unique())
        all_items = sorted(ratings["movieId"].unique())

        self.user_map = {old: new for new, old in enumerate(all_users)}
        self.item_map = {old: new for new, old in enumerate(all_items)}
        self.reverse_user_map = {new: old for old, new in self.user_map.items()}
        self.reverse_item_map = {new: old for old, new in self.item_map.items()}

        print(f"  Created mappings: {len(self.user_map):,} users, {len(self.item_map):,} items")

        # Step 3: Clean and split
        print("\n[Step 3/5] Cleaning and splitting...")
        ratings["userId"] = ratings["userId"].map(self.user_map)
        ratings["movieId"] = ratings["movieId"].map(self.item_map)

        # Remove duplicates
        dup_count = ratings.duplicated(subset=["userId", "movieId"]).sum()
        if dup_count > 0:
            ratings = ratings.drop_duplicates(subset=["userId", "movieId"])
            print(f"  Removed {dup_count:,} duplicates")

        # Sort and split
        ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
        ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

        train_rows = []
        val_rows = []
        test_rows = []

        for user_id, user_df in tqdm(ratings.groupby("userId"), desc="  Splitting"):
            n = len(user_df)
            train_end = int(n * 0.70)
            val_end = train_end + int(n * 0.15)

            train_rows.append(user_df.iloc[:train_end][["userId", "movieId", "rating"]])
            val_rows.append(user_df.iloc[train_end:val_end][["userId", "movieId", "rating"]])
            test_rows.append(user_df.iloc[val_end:][["userId", "movieId", "rating"]])

        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)

        print(f"  Train: {len(train_df):,}")
        print(f"  Val:   {len(val_df):,}")
        print(f"  Test:  {len(test_df):,}")

        del ratings, train_rows, val_rows, test_rows

        # Step 4: Add dummy genre features
        print("\n[Step 4/5] Adding placeholder genre features...")
        num_genres = 19
        dummy_genre = np.zeros(num_genres, dtype=np.float32)

        train_df["genre_features"] = [dummy_genre.copy()] * len(train_df)
        val_df["genre_features"] = [dummy_genre.copy()] * len(val_df)
        test_df["genre_features"] = [dummy_genre.copy()] * len(test_df)

        # Step 5: Save
        print("\n[Step 5/5] Saving data...")
        self._save_data(train_df, val_df, test_df)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        self._print_summary(train_df, val_df, test_df)

        print("\nNote: Genre features are placeholders (all zeros).")
        print("      Use run_full() to process actual genres for content-aware models.")

    def _save_data(self, train_df, val_df, test_df):
        """Save processed data splits."""
        # Create directories
        self.cfg.paths.ensure_dirs()

        # Save data
        train_df[["userId", "movieId", "rating", "genre_features"]].to_pickle(
            self.cfg.paths.train_path
        )
        val_df[["userId", "movieId", "rating", "genre_features"]].to_pickle(
            self.cfg.paths.val_path
        )
        test_df[["userId", "movieId", "rating", "genre_features"]].to_pickle(
            self.cfg.paths.test_path
        )

        # Save mappings
        if self.genre_encoder is None:
            num_genres = 19
            genre_classes = ["Action", "Adventure", "Animation", "Children", "Comedy",
                           "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                           "Horror", "IMAX", "Musical", "Mystery", "Romance",
                           "Sci-Fi", "Thriller", "War", "Western"]
        else:
            num_genres = len(self.genre_encoder.classes_)
            genre_classes = list(self.genre_encoder.classes_)

        mappings = {
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "num_users": len(self.user_map),
            "num_items": len(self.item_map),
            "num_genres": num_genres,
            "genre_classes": genre_classes,
        }

        with open(self.cfg.paths.mappings_path, "wb") as f:
            pickle.dump(mappings, f)

        print(f"  Saved to {self.cfg.paths.DATA_DIR}")

    def _save_cold_start(self, cold_start_df):
        """Save cold-start evaluation set."""
        cold_start_df.to_pickle(self.cfg.paths.cold_start_test_path)
        print(f"  Saved cold-start test to {self.cfg.paths.cold_start_test_path}")

    def _create_cold_start_set(self, train_df, test_df):
        """Create cold-start evaluation set."""
        # Find cold-start users (<= 10 ratings in train)
        user_train_counts = train_df["userId"].value_counts()
        cold_start_users = user_train_counts[
            user_train_counts <= self.cfg.eval.COLD_START_USER_THRESHOLD
        ].index

        # Find cold-start items (<= 10 ratings in train)
        item_train_counts = train_df["movieId"].value_counts()
        cold_start_items = item_train_counts[
            item_train_counts <= self.cfg.eval.COLD_START_ITEM_THRESHOLD
        ].index

        # Filter test set
        cold_start_df = test_df[test_df["userId"].isin(cold_start_users)].copy()
        cold_start_df["is_cold_item"] = cold_start_df["movieId"].isin(cold_start_items)

        print(f"  Cold-start users: {len(cold_start_users):,}")
        print(f"  Cold-start items: {len(cold_start_items):,}")
        print(f"  Cold-start test samples: {len(cold_start_df):,}")

        return cold_start_df

    def _print_summary(self, train_df, val_df, test_df):
        """Print summary of processed data."""
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Users: {len(self.user_map):,}")
        print(f"Items: {len(self.item_map):,}")
        print(f"Genres: {len(pickle.load(open(self.cfg.paths.mappings_path, 'rb'))['genre_classes'])}")
        print(f"\nTrain samples: {len(train_df):,}")
        print(f"Val samples:   {len(val_df):,}")
        print(f"Test samples:  {len(test_df):,}")
        print(f"{'='*60}")


# Legacy methods for backward compatibility
def _merge_data(self, data):
    """Legacy merge method (not recommended due to memory usage)."""
    ratings = data["ratings"]
    metadata = data["metadata"]
    links = data["links"]

    print(f"  Initial ratings: {len(ratings):,}")

    # Merge ratings with links
    merged = ratings.merge(
        links[["movieId", "tmdbId"]],
        on="movieId",
        how="left"
    )
    print(f"  After merging with links: {len(merged):,}")

    # Merge with metadata via tmdbId
    merged["tmdbId"] = pd.to_numeric(merged["tmdbId"], errors="coerce")
    metadata["id"] = pd.to_numeric(metadata["id"], errors="coerce")

    merged = merged.merge(
        metadata[["id", "title", "overview", "genres"]],
        left_on="tmdbId",
        right_on="id",
        how="left"
    )
    print(f"  After merging with metadata: {len(merged):,}")

    # Parse genres
    merged["genres_list"] = merged["genres"].apply(parse_genres)
    print("  Parsed genres")

    return merged.drop(columns=["id", "genres", "title", "overview"], errors="ignore")


def main():
    """Run preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MovieLens data")
    parser.add_argument("--mode", choices=["full", "minimal"], default="minimal",
                       help="Processing mode: full (with genres) or minimal (without)")
    args = parser.parse_args()

    preprocessor = DataPreprocessor()

    if args.mode == "full":
        print("Running FULL preprocessing (with content features)...")
        preprocessor.run_full()
    else:
        print("Running MINIMAL preprocessing (baseline CF only)...")
        preprocessor.run_minimal()


if __name__ == "__main__":
    main()
