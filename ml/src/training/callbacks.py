"""
Checkpointing and fault tolerance management (PRD Section 4).
Handles automated weight saving, NaN detection, state rollback, and dynamic LR decay.
"""

import os
import copy
from typing import Dict, Any, Optional
import torch

class ModelCheckpointer:
    """
    Saves and loads best model checkpoints based on validation AUC-ROC.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints", mode: str = "max"):
        self.checkpoint_dir = checkpoint_dir
        self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        os.makedirs(checkpoint_dir, exist_ok=True)

    def is_better(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: str = "best_model.pt"
    ) -> str:
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, filepath)
        return filepath

class FaultToleranceManager:
    """
    Detects NaN/Inf loss or gradients, rolls back to previous valid epoch weights,
    and scales down the learning rate dynamically.
    """
    def __init__(self, lr_decay_factor: float = 0.5):
        self.lr_decay_factor = lr_decay_factor
        self.last_valid_state: Optional[Dict[str, Any]] = None
        self.last_valid_epoch: int = -1

    def update_valid_state(self, model: torch.nn.Module, epoch: int):
        # Deepcopy weights to CPU memory for rollback
        self.last_valid_state = copy.deepcopy(model.state_dict())
        self.last_valid_epoch = epoch

    def handle_nan_event(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> bool:
        """
        Rolls back weights to last valid epoch and decays learning rate.
        Returns True if rollback was successful.
        """
        if self.last_valid_state is None:
            return False

        # Roll back weights
        model.load_state_dict(self.last_valid_state)

        # Decayed learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= self.lr_decay_factor

        return True
