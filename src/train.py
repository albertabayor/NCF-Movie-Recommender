"""
Training module for NCF models.

This module implements the training loop with:
- BPR (Bayesian Personalized Ranking) loss
- Dynamic negative sampling during training
- Learning rate scheduling
- Early stopping
- Gradient clipping
- TensorBoard logging
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import config
from .negative_sampling import NegativeSampler
from .evaluate import evaluate_model


class NCFDataset(Dataset):
    """
    Dataset for NCF training with dynamic negative sampling.

    Args:
        users: Array of user IDs
        items: Array of item IDs (positive interactions)
        num_negatives: Number of negative samples per positive
        num_items: Total number of items
        user_history: Dict mapping user -> set of interacted items
        sampling_strategy: 'uniform' or 'popularity'
    """

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        num_negatives: int,
        num_items: int,
        user_history: Dict[int, set],
        sampling_strategy: str = "uniform",
    ):
        self.users = users
        self.items = items
        self.num_negatives = num_negatives
        self.num_items = num_items
        self.user_history = user_history
        self.sampling_strategy = sampling_strategy

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]

        # Sample negative items
        neg_items = self._sample_negatives(user, pos_item)

        return {
            "user": user,
            "pos_item": pos_item,
            "neg_items": neg_items,
        }

    def _sample_negatives(self, user: int, pos_item: int) -> np.ndarray:
        """Sample negative items for a user."""
        user_items = self.user_history.get(user, set())
        candidates = np.setdiff1d(np.arange(self.num_items), list(user_items))

        if len(candidates) < self.num_negatives:
            return np.random.choice(candidates, size=self.num_negatives, replace=True)

        if self.sampling_strategy == "popularity":
            # For simplicity, use uniform sampling here
            # Popularity sampling should be done with pre-computed weights
            return np.random.choice(candidates, size=self.num_negatives, replace=False)
        else:
            return np.random.choice(candidates, size=self.num_negatives, replace=False)


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss.

    BPR loss: -log(sigmoid(pos_score - neg_score))

    This optimizes for the relative ordering of items rather than
    absolute ratings, which is more suitable for recommendation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        """
        Compute BPR loss.

        Args:
            pos_scores: Scores for positive items, shape (batch, 1) or (batch,)
            neg_scores: Scores for negative items, shape (batch, num_neg) or (batch, num_neg)

        Returns:
            Scalar loss value
        """
        # Ensure shapes are correct
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(-1)
        if neg_scores.dim() == 1:
            neg_scores = neg_scores.unsqueeze(-1)

        # BPR: -log(sigma(pos - neg))
        diff = pos_scores - neg_scores  # (batch, num_neg)
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 5.0,
    epoch: int = 0,
) -> float:
    """
    Train for one epoch.

    Args:
        model: NCF model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        max_grad_norm: Max gradient norm for clipping
        epoch: Current epoch number (for progress bar)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        users = batch["user"].to(device)
        pos_items = batch["pos_item"].to(device)
        neg_items = batch["neg_items"].to(device)  # (batch, num_neg)

        optimizer.zero_grad()

        # Forward pass for positives
        pos_output = model(users, pos_items)

        # Forward pass for negatives (need to reshape)
        batch_size = users.size(0)
        num_neg = neg_items.size(1)

        # Expand users for each negative
        users_expanded = users.unsqueeze(1).expand(-1, num_neg).reshape(batch_size * num_neg)
        neg_items_flat = neg_items.reshape(batch_size * num_neg)

        neg_output = model(users_expanded, neg_items_flat)
        neg_output = neg_output.squeeze().reshape(batch_size, num_neg)  # (batch, num_neg)

        # Compute loss
        loss = criterion(pos_output, neg_output)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_model(
    model: nn.Module,
    train_users: np.ndarray,
    train_items: np.ndarray,
    val_data: Optional[Dict] = None,
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    num_negatives: int = 4,
    num_items: int = None,
    device: str = "cuda",
    num_workers: int = 0,
    save_dir: str = None,
    early_stopping_patience: int = 5,
    early_stopping_metric: str = "hr@10",
    lr_scheduler_patience: int = 3,
    lr_scheduler_factor: float = 0.5,
    gradient_clip_max_norm: float = 5.0,
    log_dir: str = None,
) -> Dict:
    """
    Train an NCF model with early stopping and learning rate scheduling.

    Args:
        model: NCF model to train
        train_users: Training user IDs
        train_items: Training item IDs
        val_data: Validation data dict with 'users', 'items', (optional) 'genre_features'
        num_epochs: Maximum number of epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        num_negatives: Number of negative samples per positive
        num_items: Total number of items (for negative sampling)
        device: Device to train on
        save_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        early_stopping_metric: Metric to monitor for early stopping
        lr_scheduler_patience: Patience for LR scheduler
        lr_scheduler_factor: Factor to reduce LR
        gradient_clip_max_norm: Max gradient norm for clipping
        log_dir: TensorBoard log directory

    Returns:
        Dict with training history
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if save_dir is None:
        save_dir = config.paths.TRAINED_MODELS_DIR
    if log_dir is None:
        log_dir = config.paths.TENSORBOARD_LOG_DIR

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create user history for negative sampling
    from .negative_sampling import build_user_history
    user_history = build_user_history(train_users, train_items)

    # Create dataset and dataloader
    dataset = NCFDataset(
        users=train_users,
        items=train_items,
        num_negatives=num_negatives,
        num_items=num_items,
        user_history=user_history,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda" if isinstance(device, torch.device) else device == "cuda"),
    )

    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = BPRLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=1e-6,
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Training loop
    history = {
        "train_loss": [],
        "val_metrics": [],
        "learning_rate": [],
    }

    best_metric = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=gradient_clip_max_norm,
            epoch=epoch,
        )

        history["train_loss"].append(train_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Log to TensorBoard
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Validate
        if val_data is not None:
            val_metrics = evaluate_model(
                model=model,
                users=val_data["users"],
                items=val_data["items"],
                k_values=[10],
                device=device,
                num_items=num_items,
                user_history=user_history,
            )
            history["val_metrics"].append(val_metrics)

            # Log metrics
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)

            current_metric = val_metrics[early_stopping_metric]

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val HR@10: {current_metric:.4f}")

            # Early stopping
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0

                # Save best model
                save_path = Path(save_dir) / f"{model.__class__.__name__}_best.pt"
                model.save(str(save_path), epoch=epoch, metrics=val_metrics)
                print(f"  ✓ New best model saved! (HR@10: {best_metric:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epoch(s)")

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            # Update learning rate
            scheduler.step(current_metric)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best {early_stopping_metric}: {best_metric:.4f}")

    writer.close()

    return history


if __name__ == "__main__":
    print("Training module for NCF models")
    print("Import this module and use train_model() to train your models")
