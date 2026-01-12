"""
Preprocessing pipeline for NCF Movie Recommender.

This module implements the full data preprocessing pipeline:
1. Load and merge all datasets
2. Data cleaning and quality checks
3. Filter sparse users/movies
4. Time-based train/val/test split (per-user)
5. Create cold-start evaluation sets
6. Save processed data and mappings

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
    Complete preprocessing pipeline for MovieLens data.

    This class orchestrates the entire preprocessing workflow from raw
    CSV files to train/val/test splits ready for model training.
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
        self.user_map: Optional[Dict[int, int]] = None  # Original -> Remapped
        self.item_map: Optional[Dict[int, int]] = None  # Original -> Remapped
        self.reverse_user_map: Optional[Dict[int, int]] = None  # Remapped -> Original
        self.reverse_item_map: Optional[Dict[int, int]] = None  # Remapped -> Original

        # Statistics
        self.stats_before = {}
        self.stats_after = {}

    def run(self) -> None:
        """
        Execute the full preprocessing pipeline.

        This runs all steps in sequence and saves the results.
        """
        print("=" * 60)
        print("NCF MOVIE RECOMMENDER - PREPROCESSING PIPELINE")
        print("=" * 60)

        # Step 1: Load data
        print("\n[Step 1/7] Loading data...")
        data = self.loader.load_all()

        # Step 2: Merge datasets
        print("\n[Step 2/7] Merging datasets...")
        merged_df = self._merge_data(data)

        # Step 3: Clean and validate
        print("\n[Step 3/7] Cleaning and validating data...")
        merged_df = self._clean_data(merged_df)
        self.stats_before = self._compute_statistics(merged_df, "before filtering")

        # Step 4: Filter sparse users/movies
        print("\n[Step 4/7] Filtering sparse users and movies...")
        filtered_df = self._filter_sparse(merged_df)
        self.stats_after = self._compute_statistics(filtered_df, "after filtering")

        # Step 5: Time-based split
        print("\n[Step 5/7] Creating time-based train/val/test split...")
        train_df, val_df, test_df = self._time_based_split(filtered_df)

        # Step 6: Create cold-start evaluation set
        print("\n[Step 6/7] Creating cold-start evaluation set...")
        cold_start_df = self._create_cold_start_set(train_df, test_df)

        # Step 7: Save processed data
        print("\n[Step 7/7] Saving processed data...")
        self._save_processed_data(
            train_df, val_df, test_df, cold_start_df, merged_df
        )

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)

    # ========================================================================
    # STEP 2: DATA MERGING
    # ========================================================================

    def _merge_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge ratings with metadata via links table.

        Join path: ratings -> links (via movieId) -> metadata (via tmdbId)

        Args:
            data: Dictionary with ratings, metadata, links DataFrames

        Returns:
            Merged DataFrame with all information
        """
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
        # Convert tmdbId to int for merging (metadata uses int)
        merged["tmdbId"] = pd.to_numeric(merged["tmdbId"], errors="coerce")
        metadata["id"] = pd.to_numeric(metadata["id"], errors="coerce")

        merged = merged.merge(
            metadata[["id", "title", "overview", "genres"]],
            left_on="tmdbId",
            right_on="id",
            how="left"
        )

        print(f"  After merging with metadata: {len(merged):,}")

        # Parse genres from JSON
        print("  Parsing genres...")
        merged["genres_list"] = merged["genres"].apply(parse_genres)

        # Clean up
        merged = merged.drop(columns=["id", "genres"])

        return merged

    # ========================================================================
    # STEP 3: CLEANING & VALIDATION
    # ========================================================================

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data and perform quality checks.

        Checks:
        - Rating range: 0.5 - 5.0
        - Remove rows with empty overview
        - Remove duplicates (userId, movieId)
        - Remove movies without genres
        - Verify timestamp ordering

        Args:
            df: Merged DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)

        # 1. Validate rating range
        df = df[
            (df["rating"] >= self.cfg.data.MIN_RATING) &
            (df["rating"] <= self.cfg.data.MAX_RATING)
        ]
        print(f"  After rating validation: {len(df):,} ({len(df) - initial_count:,} removed)")

        # 2. Remove rows with empty overview
        df = df[df["overview"].notna() & (df["overview"] != "")]
        print(f"  After removing empty overview: {len(df):,}")

        # 3. Remove duplicates (userId, movieId)
        dup_count = df.duplicated(subset=["userId", "movieId"]).sum()
        if dup_count > 0:
            print(f"  Removing {dup_count:,} duplicate (userId, movieId) pairs")
            df = df.drop_duplicates(subset=["userId", "movieId"])

        # 4. Remove movies without genres
        df = df[df["genres_list"].apply(len) > 0]
        print(f"  After removing movies without genres: {len(df):,}")

        # 5. Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

        # 6. Verify timestamp ordering per user
        # Check if timestamps are monotonically increasing for each user
        def check_monotonic(group):
            return group["datetime"].is_monotonic_increasing

        monotonic_check = df.groupby("userId").apply(check_monotonic)
        non_monotonic_users = monotonic_check[~monotonic_check].index.tolist()

        if non_monotonic_users:
            print(f"  WARNING: {len(non_monotonic_users)} users have non-monotonic timestamps")
            # Sort per user to ensure monotonic ordering
            df = df.sort_values(["userId", "timestamp"])
            print(f"  Fixed by sorting per-user")

        # 7. Check: all movies in ratings exist in metadata
        movies_with_metadata = df["tmdbId"].notna().sum()
        total = len(df)
        coverage = (movies_with_metadata / total * 100) if total > 0 else 0.0
        print(f"  Movies with metadata: {movies_with_metadata:,}/{total:,} ({coverage:.1f}%)")

        return df

    # ========================================================================
    # STEP 4: FILTERING
    # ========================================================================

    def _filter_sparse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out sparse users and movies.

        Users with < MIN_USER_RATINGS are removed.
        Movies with < MIN_ITEM_RATINGS are removed.

        Args:
            df: Cleaned DataFrame

        Returns:
            Filtered DataFrame
        """
        initial_users = df["userId"].nunique()
        initial_items = df["movieId"].nunique()
        initial_ratings = len(df)

        # Iteratively filter (users first, then items, repeat until stable)
        print(f"  Initial: {initial_users:,} users, {initial_items:,} items, {initial_ratings:,} ratings")

        prev_count = 0
        iteration = 0

        while True:
            iteration += 1
            print(f"  Iteration {iteration}:")

            # Filter users
            user_counts = df["userId"].value_counts()
            valid_users = user_counts[user_counts >= self.cfg.data.MIN_USER_RATINGS].index
            df = df[df["userId"].isin(valid_users)]
            print(f"    Users: {df['userId'].nunique():,}")

            # Filter items
            item_counts = df["movieId"].value_counts()
            valid_items = item_counts[item_counts >= self.cfg.data.MIN_ITEM_RATINGS].index
            df = df[df["movieId"].isin(valid_items)]
            print(f"    Items: {df['movieId'].nunique():,}")

            # Check convergence
            current_count = len(df)
            if current_count == prev_count:
                break
            prev_count = current_count

        print(f"  Final: {df['userId'].nunique():,} users, {df['movieId'].nunique():,} items, {len(df):,} ratings")
        print(f"  Removed: {initial_ratings - len(df):,} ratings ({(1 - len(df)/initial_ratings)*100:.1f}%)")

        return df

    # ========================================================================
    # STEP 5: TIME-BASED SPLIT
    # ========================================================================

    def _time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/val/test split PER USER.

        For each user:
        - Sort by timestamp
        - Train: first 70% of ratings
        - Val: next 15% of ratings
        - Test: last 15% of ratings

        Args:
            df: Filtered DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"  Creating per-user time-based split...")

        # Sort by userId and timestamp
        df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)

        train_rows = []
        val_rows = []
        test_rows = []

        # Split per user
        for user_id, user_df in tqdm(df.groupby("userId"), desc="  Splitting users"):
            n = len(user_df)

            # Calculate split indices
            train_end = int(n * self.cfg.data.TRAIN_RATIO)
            val_end = train_end + int(n * self.cfg.data.VAL_RATIO)

            train_rows.append(user_df.iloc[:train_end])
            val_rows.append(user_df.iloc[train_end:val_end])
            test_rows.append(user_df.iloc[val_end:])

        train_df = pd.concat(train_rows).reset_index(drop=True)
        val_df = pd.concat(val_rows).reset_index(drop=True)
        test_df = pd.concat(test_rows).reset_index(drop=True)

        print(f"  Train: {len(train_df):,} ratings ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} ratings ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} ratings ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    # ========================================================================
    # STEP 6: COLD-START EVALUATION SET
    # ========================================================================

    def _create_cold_start_set(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create cold-start evaluation set from test data.

        Cold-start users: users with <= COLD_START_USER_THRESHOLD ratings in train
        Cold-start items: items with <= COLD_START_ITEM_THRESHOLD ratings in train

        Args:
            train_df: Training data
            test_df: Test data

        Returns:
            Cold-start test subset
        """
        # Find cold-start users
        user_train_counts = train_df["userId"].value_counts()
        cold_start_users = user_train_counts[
            user_train_counts <= self.cfg.eval.COLD_START_USER_THRESHOLD
        ].index

        # Find cold-start items
        item_train_counts = train_df["movieId"].value_counts()
        cold_start_items = item_train_counts[
            item_train_counts <= self.cfg.eval.COLD_START_ITEM_THRESHOLD
        ].index

        # Filter test set for cold-start users
        cold_start_df = test_df[test_df["userId"].isin(cold_start_users)].copy()

        # Mark cold-start items in this subset
        cold_start_df["is_cold_item"] = cold_start_df["movieId"].isin(cold_start_items)

        print(f"  Cold-start users in train: {len(cold_start_users):,}")
        print(f"  Cold-start items in train: {len(cold_start_items):,}")
        print(f"  Cold-start test samples: {len(cold_start_df):,}")
        print(f"    From cold-start users: {len(cold_start_df):,}")
        print(f"    With cold-start items: {cold_start_df['is_cold_item'].sum():,}")

        return cold_start_df

    # ========================================================================
    # STEP 7: SAVE PROCESSED DATA
    # ========================================================================

    def _save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cold_start_df: pd.DataFrame,
        merged_df: pd.DataFrame,
    ) -> None:
        """
        Save processed data and create mappings.

        This creates:
        - User/item mappings (original ID -> remapped 0..n-1)
        - Genre encoder (for multi-hot encoding)
        - Processed data splits
        - Statistics

        Args:
            train_df, val_df, test_df: Data splits
            cold_start_df: Cold-start evaluation set
            merged_df: Full merged data (for genre encoding)
        """
        # Create user and item mappings
        all_users = merged_df["userId"].unique()
        all_items = merged_df["movieId"].unique()

        self.user_map = {old: new for new, old in enumerate(sorted(all_users))}
        self.item_map = {old: new for new, old in enumerate(sorted(all_items))}
        self.reverse_user_map = {new: old for old, new in self.user_map.items()}
        self.reverse_item_map = {new: old for old, new in self.item_map.items()}

        print(f"  Created mappings: {len(self.user_map):,} users, {len(self.item_map):,} items")

        # Create genre encoder
        self.genre_encoder = MultiLabelBinarizer()
        all_genres_list = merged_df["genres_list"].tolist()
        self.genre_encoder.fit(all_genres_list)

        print(f"  Created genre encoder: {len(self.genre_encoder.classes_)} genres")
        print(f"    Genres: {', '.join(self.genre_encoder.classes_)}")

        # Apply mappings to data splits
        def remap_ids(df):
            df = df.copy()
            df["userId"] = df["userId"].map(self.user_map)
            df["movieId"] = df["movieId"].map(self.item_map)
            return df

        train_df = remap_ids(train_df)
        val_df = remap_ids(val_df)
        test_df = remap_ids(test_df)
        cold_start_df = remap_ids(cold_start_df)

        # Encode genres
        train_df["genre_features"] = list(self.genre_encoder.transform(train_df["genres_list"]))
        val_df["genre_features"] = list(self.genre_encoder.transform(val_df["genres_list"]))
        test_df["genre_features"] = list(self.genre_encoder.transform(test_df["genres_list"]))
        cold_start_df["genre_features"] = list(self.genre_encoder.transform(cold_start_df["genres_list"]))

        # Save data splits
        train_df[["userId", "movieId", "rating", "genre_features"]].to_pickle(
            self.cfg.paths.train_path
        )
        val_df[["userId", "movieId", "rating", "genre_features"]].to_pickle(
            self.cfg.paths.val_path
        )
        test_df[["userId", "movieId", "rating", "genre_features"]].to_pickle(
            self.cfg.paths.test_path
        )
        cold_start_df[["userId", "movieId", "rating", "genre_features", "is_cold_item"]].to_pickle(
            self.cfg.paths.cold_start_test_path
        )

        print(f"  Saved data splits to {self.cfg.paths.DATA_DIR}")

        # Save mappings
        mappings = {
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "genre_encoder": self.genre_encoder,
            "num_users": len(self.user_map),
            "num_items": len(self.item_map),
            "num_genres": len(self.genre_encoder.classes_),
            "genre_classes": list(self.genre_encoder.classes_),
        }

        with open(self.cfg.paths.mappings_path, "wb") as f:
            pickle.dump(mappings, f)

        print(f"  Saved mappings to {self.cfg.paths.mappings_path}")

        # Save statistics
        statistics = {
            "before_filtering": self.stats_before,
            "after_filtering": self.stats_after,
            "split_counts": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
                "cold_start_test": len(cold_start_df),
            },
            "metadata": {
                "num_users": len(self.user_map),
                "num_items": len(self.item_map),
                "num_genres": len(self.genre_encoder.classes_),
                "genres": list(self.genre_encoder.classes_),
                "generated_at": datetime.now().isoformat(),
            },
        }

        with open(self.cfg.paths.statistics_path, "w") as f:
            json.dump(statistics, f, indent=2)

        print(f"  Saved statistics to {self.cfg.paths.statistics_path}")

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _compute_statistics(self, df: pd.DataFrame, label: str) -> Dict:
        """Compute statistics for a DataFrame."""
        num_users = int(df["userId"].nunique())
        num_items = int(df["movieId"].nunique())
        num_ratings = int(len(df))

        # Handle empty dataframes gracefully
        if num_ratings == 0:
            return {
                label: {
                    "num_users": 0,
                    "num_items": 0,
                    "num_ratings": 0,
                    "avg_rating": 0.0,
                    "sparsity": 0.0,
                }
            }

        sparsity = 0.0
        if num_users > 0 and num_items > 0:
            sparsity = float(1 - num_ratings / (num_users * num_items))

        return {
            label: {
                "num_users": num_users,
                "num_items": num_items,
                "num_ratings": num_ratings,
                "avg_rating": float(df["rating"].mean()),
                "sparsity": sparsity,
            }
        }


def main():
    """Run preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
