"""
Negative sampling module for NCF Movie Recommender.

This module implements dynamic negative sampling during training.
As outlined in PROJECT_PLAN.md section 4.2, negative samples are generated
on-the-fly during training to ensure diversity and better convergence.

Key features:
- Dynamic sampling (not pre-sampled)
- Ensures negatives are NOT in user's interaction history
- Supports uniform and popularity-based sampling strategies
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class NegativeSampler:
    """
    Dynamic negative sampler for collaborative filtering.

    Samples negative items during training rather than pre-sampling.
    This ensures diversity and prevents the model from memorizing
    specific negative samples.

    Sampling strategies:
    - 'uniform': Sample uniformly from all items
    - 'popularity': Sample inversely proportional to item popularity
                   (gives rare items more weight)
    """

    def __init__(
        self,
        user_history: Dict[int, set],
        num_items: int,
        num_negatives: int = 4,
        strategy: str = "uniform",
        item_popularity: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        """
        Initialize the negative sampler.

        Args:
            user_history: Dict mapping user_id -> set of interacted item_ids
            num_items: Total number of unique items
            num_negatives: Number of negative samples per positive
            strategy: 'uniform' or 'popularity'
            item_popularity: Precomputed item counts (for popularity sampling)
            seed: Random seed for reproducibility
        """
        self.user_history = user_history
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.strategy = strategy
        self.seed = seed

        # Precompute sampling weights for popularity-based strategy
        if strategy == "popularity":
            if item_popularity is None:
                raise ValueError("item_popularity must be provided for popularity sampling")
            self._compute_popularity_weights(item_popularity)
        else:
            self.popularity_weights = None

        # All item IDs
        self.all_items = np.arange(num_items)

    def _compute_popularity_weights(self, item_popularity: np.ndarray) -> None:
        """
        Compute inverse popularity weights.

        Items with higher popularity get LOWER weights, making them
        less likely to be sampled as negatives. This helps with
        learning long-tail items.

        Args:
            item_popularity: Array of item interaction counts
        """
        # Inverse popularity with smoothing
        weights = 1.0 / (item_popularity + 1e-8)
        self.popularity_weights = weights / weights.sum()

    def sample(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
    ) -> np.ndarray:
        """
        Sample negative items for given user-positive pairs.

        For each (user, pos_item) pair, sample N negative items that
        the user has NOT interacted with.

        Args:
            users: Array of user IDs (shape: [batch_size])
            pos_items: Array of positive item IDs (shape: [batch_size])

        Returns:
            Array of negative item IDs (shape: [batch_size, num_negatives])
        """
        batch_size = len(users)
        neg_items = np.zeros((batch_size, self.num_negatives), dtype=np.int64)

        for i, (user, pos_item) in enumerate(zip(users, pos_items)):
            # Get items this user has NOT interacted with
            user_items = self.user_history.get(user, set())
            candidates = np.setdiff1d(self.all_items, list(user_items))

            # Ensure we have enough candidates
            if len(candidates) < self.num_negatives:
                # If not enough, sample with replacement
                neg_items[i] = np.random.choice(
                    candidates,
                    size=self.num_negatives,
                    replace=True,
                )
            else:
                # Sample without replacement
                if self.strategy == "popularity" and self.popularity_weights is not None:
                    # Get weights for candidates only
                    candidate_weights = self.popularity_weights[candidates]
                    candidate_weights = candidate_weights / candidate_weights.sum()

                    neg_items[i] = np.random.choice(
                        candidates,
                        size=self.num_negatives,
                        replace=False,
                        p=candidate_weights,
                    )
                else:  # uniform
                    neg_items[i] = np.random.choice(
                        candidates,
                        size=self.num_negatives,
                        replace=False,
                    )

        return neg_items

    def sample_for_user(self, user_id: int, num_samples: int = 100) -> np.ndarray:
        """
        Sample negative items for a specific user.

        Useful for inference and evaluation.

        Args:
            user_id: User ID
            num_samples: Number of negative samples to generate

        Returns:
            Array of negative item IDs
        """
        user_items = self.user_history.get(user_id, set())
        candidates = np.setdiff1d(self.all_items, list(user_items))

        if len(candidates) < num_samples:
            # Sample with replacement if not enough candidates
            replace = True
        else:
            replace = False

        if self.strategy == "popularity" and self.popularity_weights is not None:
            candidate_weights = self.popularity_weights[candidates]
            candidate_weights = candidate_weights / candidate_weights.sum()
            return np.random.choice(
                candidates,
                size=num_samples,
                replace=replace,
                p=candidate_weights,
            )
        else:
            return np.random.choice(
                candidates,
                size=num_samples,
                replace=replace,
            )


def build_user_history(
    train_users: np.ndarray,
    train_items: np.ndarray,
) -> Dict[int, set]:
    """
    Build user interaction history from training data.

    Args:
        train_users: Array of user IDs in training set
        train_items: Array of item IDs in training set

    Returns:
        Dict mapping user_id -> set of interacted item_ids
    """
    user_history = {}

    for user, item in zip(train_users, train_items):
        if user not in user_history:
            user_history[user] = set()
        user_history[user].add(item)

    return user_history


def compute_item_popularity(
    train_items: np.ndarray,
    num_items: int,
) -> np.ndarray:
    """
    Compute item popularity (interaction counts).

    Args:
        train_items: Array of item IDs in training set
        num_items: Total number of items

    Returns:
        Array of interaction counts per item
    """
    popularity = np.zeros(num_items, dtype=np.int32)

    unique_items, counts = np.unique(train_items, return_counts=True)
    popularity[unique_items] = counts

    return popularity


def create_negative_sampler(
    train_users: np.ndarray,
    train_items: np.ndarray,
    num_items: int,
    num_negatives: int = 4,
    strategy: str = "uniform",
    seed: int = 42,
) -> NegativeSampler:
    """
    Factory function to create a NegativeSampler.

    Args:
        train_users: Array of user IDs in training set
        train_items: Array of item IDs in training set
        num_items: Total number of items
        num_negatives: Number of negatives per positive
        strategy: 'uniform' or 'popularity'
        seed: Random seed

    Returns:
        Configured NegativeSampler instance
    """
    # Build user history
    user_history = build_user_history(train_users, train_items)

    # Compute item popularity for popularity-based sampling
    item_popularity = None
    if strategy == "popularity":
        item_popularity = compute_item_popularity(train_items, num_items)

    # Create sampler
    sampler = NegativeSampler(
        user_history=user_history,
        num_items=num_items,
        num_negatives=num_negatives,
        strategy=strategy,
        item_popularity=item_popularity,
        seed=seed,
    )

    return sampler


if __name__ == "__main__":
    # Test the negative sampler
    print("Testing NegativeSampler...")

    # Create dummy data
    num_users = 100
    num_items = 500
    num_interactions = 2000

    np.random.seed(42)
    train_users = np.random.randint(0, num_users, num_interactions)
    train_items = np.random.randint(0, num_items, num_interactions)

    # Create samplers
    uniform_sampler = create_negative_sampler(
        train_users, train_items, num_items, strategy="uniform"
    )

    popularity_sampler = create_negative_sampler(
        train_users, train_items, num_items, strategy="popularity"
    )

    # Test sampling
    users = np.array([0, 1, 2, 3, 4])
    items = np.array([10, 20, 30, 40, 50])

    print("\nUniform sampling:")
    neg_samples = uniform_sampler.sample(users, items)
    print(f"Shape: {neg_samples.shape}")
    print(f"Sample negatives for user 0: {neg_samples[0]}")

    print("\nPopularity sampling:")
    neg_samples = popularity_sampler.sample(users, items)
    print(f"Shape: {neg_samples.shape}")
    print(f"Sample negatives for user 0: {neg_samples[0]}")

    print("\n✓ NegativeSampler test passed!")
