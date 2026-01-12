"""
Data loader module for NCF Movie Recommender.

This module handles loading raw dataset files:
- ratings.csv: User-item ratings with timestamps
- movies_metadata.csv: Movie metadata including genres and overview
- links.csv: Mapping between MovieLens and TMDB/IMDb IDs
- keywords.csv: Optional movie keywords

References:
- MovieLens 25M Dataset: https://grouplens.org/datasets/movielens/25m/
- TMDB Metadata: https://www.themoviedb.org/
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class DataLoader:
    """
    Load and validate raw dataset files.

    This class provides methods to load each dataset file with
    proper type inference and basic validation.
    """

    def __init__(self, datasets_dir: str = "datasets"):
        """
        Initialize the data loader.

        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = Path(datasets_dir)
        self._verify_files_exist()

    def _verify_files_exist(self) -> None:
        """Verify that all required data files exist."""
        required_files = [
            "ratings.csv",
            "movies_metadata.csv",
            "links.csv",
        ]

        missing = []
        for filename in required_files:
            filepath = self.datasets_dir / filename
            if not filepath.exists():
                missing.append(filename)

        if missing:
            raise FileNotFoundError(
                f"Missing required data files: {', '.join(missing)}\n"
                f"Expected location: {self.datasets_dir}"
            )

    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings.csv file.

        Expected columns:
            userId: int - User ID
            movieId: int - Movie ID (MovieLens)
            rating: float - Rating from 0.5 to 5.0
            timestamp: int - Unix timestamp

        Returns:
            DataFrame with columns [userId, movieId, rating, timestamp]
        """
        filepath = self.datasets_dir / "ratings.csv"

        print(f"Loading ratings from {filepath}...")
        df = pd.read_csv(
            filepath,
            usecols=["userId", "movieId", "rating", "timestamp"],
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
                "timestamp": np.int64,
            },
        )

        print(f"  Loaded {len(df):,} ratings")
        print(f"  Users: {df['userId'].nunique():,}")
        print(f"  Movies: {df['movieId'].nunique():,}")

        return df

    def load_movies_metadata(self) -> pd.DataFrame:
        """
        Load movies_metadata.csv file.

        Expected columns (we use a subset):
            id: str/int - TMDB ID (primary key)
            title: str - Movie title
            overview: str - Movie synopsis
            genres: str - JSON string of genre objects
            release_date: str - Release date (YYYY-MM-DD)

        Returns:
            DataFrame with movie metadata
        """
        filepath = self.datasets_dir / "movies_metadata.csv"

        print(f"Loading movies metadata from {filepath}...")

        # Load with low_memory=False to handle mixed types
        df = pd.read_csv(filepath, low_memory=False)

        # Extract columns we need
        required_cols = ["id", "title", "overview", "genres"]
        available_cols = [col for col in required_cols if col in df.columns]

        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            print(f"  WARNING: Missing columns: {missing}")

        df = df[available_cols].copy()

        # Clean TMDB ID (sometimes stored as string with errors)
        df["id"] = pd.to_numeric(df["id"], errors="coerce")

        print(f"  Loaded {len(df):,} movies")
        print(f"  Movies with overview: {df['overview'].notna().sum():,}")
        print(f"  Movies with genres: {df['genres'].notna().sum():,}")

        return df

    def load_links(self) -> pd.DataFrame:
        """
        Load links.csv file.

        Expected columns:
            movieId: int - MovieLens ID
            imdbId: int - IMDb ID
            tmdbId: int/float - TMDB ID (may have NaN)

        Returns:
            DataFrame with ID mappings
        """
        filepath = self.datasets_dir / "links.csv"

        print(f"Loading links from {filepath}...")
        df = pd.read_csv(
            filepath,
            dtype={
                "movieId": np.int32,
                "imdbId": np.int32,
                "tmdbId": np.float64,  # May contain NaN
            },
        )

        # Count valid TMDB IDs
        valid_tmdb = df["tmdbId"].notna().sum()

        print(f"  Loaded {len(df):,} links")
        print(f"  Valid TMDB IDs: {valid_tmdb:,}")

        return df

    def load_keywords(self) -> Optional[pd.DataFrame]:
        """
        Load keywords.csv file (optional).

        Expected columns:
            id: int - TMDB ID
            keywords: str - JSON string of keyword objects

        Returns:
            DataFrame with keywords, or None if file doesn't exist
        """
        filepath = self.datasets_dir / "keywords.csv"

        if not filepath.exists():
            print(f"  Keywords file not found (optional)")
            return None

        print(f"Loading keywords from {filepath}...")
        df = pd.read_csv(filepath)

        print(f"  Loaded {len(df):,} keywords")

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data files.

        Returns:
            Dictionary with keys: ratings, metadata, links, (optional) keywords
        """
        data = {
            "ratings": self.load_ratings(),
            "metadata": self.load_movies_metadata(),
            "links": self.load_links(),
            "keywords": self.load_keywords(),
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        return data


def parse_genres(genres_json: str) -> List[str]:
    """
    Parse genres from JSON string.

    Args:
        genres_json: JSON string like '[{"id": 1, "name": "Action"}, ...]'

    Returns:
        List of genre names
    """
    if pd.isna(genres_json) or genres_json == "":
        return []

    try:
        genres = json.loads(genres_json)
        return [g["name"] for g in genres]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def extract_all_genres(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique genres from the metadata DataFrame.

    Args:
        df: DataFrame with 'genres' column (JSON strings)

    Returns:
        Sorted list of unique genre names
    """
    all_genres = set()
    for genres_str in df["genres"].dropna():
        genres = parse_genres(genres_str)
        all_genres.update(genres)

    return sorted(list(all_genres))


def get_dataset_statistics(data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Compute basic statistics about the loaded datasets.

    Args:
        data: Dictionary with DataFrames from load_all()

    Returns:
        Dictionary with statistics
    """
    ratings = data["ratings"]
    metadata = data["metadata"]
    links = data["links"]

    stats = {
        "ratings": {
            "total_ratings": int(len(ratings)),
            "unique_users": int(ratings["userId"].nunique()),
            "unique_movies": int(ratings["movieId"].nunique()),
            "rating_mean": float(ratings["rating"].mean()),
            "rating_std": float(ratings["rating"].std()),
            "rating_min": float(ratings["rating"].min()),
            "rating_max": float(ratings["rating"].max()),
            "sparsity": float(
                1 - len(ratings) / (ratings["userId"].nunique() * ratings["movieId"].nunique())
            ),
        },
        "metadata": {
            "total_movies": int(len(metadata)),
            "with_overview": int(metadata["overview"].notna().sum()),
            "with_genres": int(metadata["genres"].notna().sum()),
        },
        "links": {
            "total_links": int(len(links)),
            "with_tmdb_id": int(links["tmdbId"].notna().sum()),
        },
    }

    # Ratings per user statistics
    user_counts = ratings["userId"].value_counts()
    stats["user_interaction"] = {
        "mean_ratings_per_user": float(user_counts.mean()),
        "median_ratings_per_user": float(user_counts.median()),
        "min_ratings_per_user": int(user_counts.min()),
        "max_ratings_per_user": int(user_counts.max()),
    }

    # Ratings per item statistics
    item_counts = ratings["movieId"].value_counts()
    stats["item_interaction"] = {
        "mean_ratings_per_item": float(item_counts.mean()),
        "median_ratings_per_item": float(item_counts.median()),
        "min_ratings_per_item": int(item_counts.min()),
        "max_ratings_per_item": int(item_counts.max()),
    }

    return stats


if __name__ == "__main__":
    # Test loading data
    loader = DataLoader()
    data = loader.load_all()

    # Print statistics
    stats = get_dataset_statistics(data)
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(json.dumps(stats, indent=2))
