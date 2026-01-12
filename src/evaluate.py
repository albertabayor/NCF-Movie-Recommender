"""
Evaluation module for NCF models.

This module implements evaluation metrics for recommender systems:
- Hit Rate @K (HR@K): Fraction of users with at least one hit in top-K
- NDCG @K: Normalized Discounted Cumulative Gain
- AUC: Area Under the ROC Curve
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def hit_rate_at_k(
    predictions: np.ndarray,
    ground_truth: set,
    k: int,
) -> float:
    """
    Compute Hit Rate @K.

    HR@K = 1 if at least one item in top-K is in ground truth, else 0

    Args:
        predictions: Array of predicted item IDs (ranked)
        ground_truth: Set of relevant item IDs
        k: Number of top items to consider

    Returns:
        HR@K score (0 or 1)
    """
    top_k = predictions[:k]
    hit = len(set(top_k) & ground_truth) > 0
    return float(hit)


def ndcg_at_k(
    predictions: np.ndarray,
    ground_truth: set,
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @K.

    NDCG@K = DCG@K / IDCG@K

    where:
    DCG@K = sum(relevance_i / log2(i+1)) for i in 1..K
    IDCG@K = DCG@K with perfect ranking

    Args:
        predictions: Array of predicted item IDs (ranked)
        ground_truth: Set of relevant item IDs (all have relevance=1)
        k: Number of top items to consider

    Returns:
        NDCG@K score (0 to 1)
    """
    top_k = predictions[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 1)

    # IDCG (perfect ranking)
    idcg = 0.0
    num_relevant = min(len(ground_truth), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def compute_auc(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> float:
    """
    Compute AUC (Area Under ROC Curve).

    AUC = P(pos_score > neg_score) + 0.5 * P(pos_score == neg_score)

    Args:
        pos_scores: Scores for positive items
        neg_scores: Scores for negative items

    Returns:
        AUC score (0 to 1)
    """
    # Count how often pos > neg
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    auc = 0.0
    for pos_score in pos_scores:
        auc += (pos_score > neg_scores).sum()
        auc += (pos_score == neg_scores).sum() * 0.5

    auc /= len(pos_scores) * len(neg_scores)

    return float(auc)


def evaluate_user(
    model: nn.Module,
    user_id: int,
    ground_truth_items: set,
    all_items: np.ndarray,
    k_values: List[int],
    device: torch.device,
    num_negatives_test: int = 99,
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluate a single user.

    Generates rankings by scoring the user's positive items against
    randomly sampled negative items.

    Args:
        model: NCF model
        user_id: User ID to evaluate
        ground_truth_items: Set of relevant items for this user
        all_items: Array of all item IDs
        k_values: List of K values for HR@K and NDCG@K
        device: Device to run on
        num_negatives_test: Number of negative items to sample
        **kwargs: Additional model args (genre_features, synopsis_embeddings)

    Returns:
        Dict with metrics for this user
    """
    model.eval()

    with torch.no_grad():
        # Sample negative items (not in ground truth)
        neg_items = np.random.choice(
            list(set(all_items) - ground_truth_items),
            size=min(num_negatives_test, len(all_items) - len(ground_truth_items)),
            replace=False,
        )

        # Combine with positive items
        test_items = np.concatenate([list(ground_truth_items), neg_items])

        # Score all items
        user_tensor = torch.LongTensor([user_id] * len(test_items)).to(device)
        item_tensor = torch.LongTensor(test_items).to(device)

        # Handle additional kwargs (genre_features, synopsis_embeddings)
        model_kwargs = {}
        if "genre_features" in kwargs and kwargs["genre_features"] is not None:
            # Get genre features for test items
            genre_features = kwargs["genre_features"][test_items]
            model_kwargs["genre_features"] = torch.FloatTensor(genre_features).to(device)

        if "synopsis_embeddings" in kwargs and kwargs["synopsis_embeddings"] is not None:
            # Get synopsis embeddings for test items
            synopsis_embeddings = kwargs["synopsis_embeddings"][test_items]
            model_kwargs["synopsis_embeddings"] = torch.FloatTensor(synopsis_embeddings).to(device)

        scores = model(user_tensor, item_tensor, **model_kwargs).squeeze(-1)

        # Sort by score (descending)
        ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
        ranked_items = test_items[ranked_indices]

    # Compute metrics
    metrics = {}
    for k in k_values:
        metrics[f"hr@{k}"] = hit_rate_at_k(ranked_items, ground_truth_items, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_items, ground_truth_items, k)

    # AUC (using first positive vs negatives)
    pos_idx = list(ground_truth_items)[0]
    pos_score = scores[test_items == pos_idx].cpu().item()
    neg_scores = scores[test_items != pos_idx].cpu().numpy()
    metrics["auc"] = compute_auc([pos_score], neg_scores)

    return metrics


def evaluate_model(
    model: nn.Module,
    users: np.ndarray,
    items: np.ndarray,
    k_values: List[int] = None,
    device: str = "cuda",
    num_items: int = None,
    user_history: Dict[int, set] = None,
    num_negatives_test: int = 99,
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: NCF model
        users: Test user IDs
        items: Test item IDs (positive interactions)
        k_values: List of K values for HR@K and NDCG@K
        device: Device to run on
        num_items: Total number of items (for negative sampling)
        user_history: Dict mapping user -> interacted items (to exclude from negatives)
        num_negatives_test: Number of negatives to sample per user
        **kwargs: Additional model args (genre_features, synopsis_embeddings)

    Returns:
        Dict with average metrics across all users
    """
    if k_values is None:
        k_values = [5, 10, 20]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Group items by user
    user_items = {}
    for user, item in zip(users, items):
        if user not in user_items:
            user_items[user] = set()
        user_items[user].add(item)

    # Evaluate each user
    all_metrics = []
    all_items_arr = np.arange(num_items)

    for user_id, ground_truth in tqdm(user_items.items(), desc="Evaluating"):
        user_metrics = evaluate_user(
            model=model,
            user_id=user_id,
            ground_truth_items=ground_truth,
            all_items=all_items_arr,
            k_values=k_values,
            device=device,
            num_negatives_test=num_negatives_test,
            **kwargs,
        )
        all_metrics.append(user_metrics)

    # Average metrics
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        avg_metrics[metric_name] = np.mean(values)

    return avg_metrics


def evaluate_cold_start(
    model: nn.Module,
    test_users: np.ndarray,
    test_items: np.ndarray,
    cold_start_mask: np.ndarray,
    k_values: List[int] = None,
    device: str = "cuda",
    num_items: int = None,
    user_history: Dict[int, set] = None,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model separately on cold-start and non-cold-start items.

    Args:
        model: NCF model
        test_users: Test user IDs
        test_items: Test item IDs
        cold_start_mask: Boolean array indicating cold-start items
        k_values: List of K values for HR@K and NDCG@K
        device: Device to run on
        num_items: Total number of items
        user_history: Dict mapping user -> interacted items
        **kwargs: Additional model args

    Returns:
        Dict with 'cold_start' and 'warm' metrics
    """
    cold_start_indices = np.where(cold_start_mask)[0]
    warm_indices = np.where(~cold_start_mask)[0]

    results = {}

    if len(cold_start_indices) > 0:
        results["cold_start"] = evaluate_model(
            model=model,
            users=test_users[cold_start_indices],
            items=test_items[cold_start_indices],
            k_values=k_values,
            device=device,
            num_items=num_items,
            user_history=user_history,
            **kwargs,
        )

    if len(warm_indices) > 0:
        results["warm"] = evaluate_model(
            model=model,
            users=test_users[warm_indices],
            items=test_items[warm_indices],
            k_values=k_values,
            device=device,
            num_items=num_items,
            user_history=user_history,
            **kwargs,
        )

    return results


if __name__ == "__main__":
    print("Evaluation module for NCF models")
    print("Import this module and use evaluate_model() to evaluate your models")
