"""
Hybrid Q-MedCAM training loop with differentiated learning rates (REQ-4),
TensorBoard logging, and fault tolerance rollback.
"""

import os
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import HybridQMedCAMLoss
from .metrics import compute_classification_metrics
from .callbacks import ModelCheckpointer, FaultToleranceManager

class HybridTrainer:
    """
    Manages end-to-end training and evaluation for QMedCAMHybridModel.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        backbone_lr: float = 1e-5,
        classical_lr: float = 1e-3,
        quantum_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs/q_medcam"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # REQ-4: Three-tier differentiated parameter groups:
        # 1. Backbone CNN: lower LR (1e-5) to preserve pretrained ImageNet visual representations
        # 2. Classical Bottleneck: standard LR (1e-3) for rapid 512->4 latent convergence
        # 3. Quantum Circuit: conservative LR (1e-4) for quantum Hilbert state stability
        backbone_params = []
        bottleneck_params = []
        quantum_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "quantum" in name:
                quantum_params.append(param)
            elif "bottleneck" in name or "channel_attention" in name:
                bottleneck_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone"
            })
        if bottleneck_params:
            param_groups.append({
                "params": bottleneck_params,
                "lr": classical_lr,
                "weight_decay": weight_decay,
                "name": "bottleneck"
            })
        if quantum_params:
            param_groups.append({
                "params": quantum_params,
                "lr": quantum_lr,
                "weight_decay": 0.0,
                "name": "quantum"
            })

        self.optimizer = torch.optim.AdamW(param_groups)

        self.loss_fn = HybridQMedCAMLoss(l2_reg=weight_decay)
        self.checkpointer = ModelCheckpointer(checkpoint_dir=checkpoint_dir)
        self.fault_manager = FaultToleranceManager()
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels, bottleneck_layer=self.model.backbone.bottleneck)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss encountered at epoch {epoch}. Executing rollback...")
                success = self.fault_manager.handle_nan_event(self.model, self.optimizer)
                if success:
                    print("Rollback successful. Learning rate decayed.")
                    return float("nan")
                else:
                    raise RuntimeError("Fatal: NaN loss encountered with no previous valid state to rollback to.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar("Train/Loss", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, epoch: int, data_loader: Optional[DataLoader] = None, split_name: str = "Val") -> Dict[str, float]:
        self.model.eval()
        loader = data_loader or self.val_loader
        all_probs = []
        all_targets = []
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(loader), 1)
        metrics = compute_classification_metrics(all_targets, all_probs)
        metrics["loss"] = avg_loss

        # Log metrics to TensorBoard (PRD Section 4)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{split_name}/{k}", v, epoch)

        return metrics

    def train(self, num_epochs: int = 25) -> Dict[str, Any]:
        best_val_auc = -1.0
        best_metrics: Dict[str, float] = {}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(num_epochs, 1), eta_min=1e-6)

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            if train_loss != train_loss:  # NaN check
                continue

            self.fault_manager.update_valid_state(self.model, epoch)
            val_metrics = self.evaluate(epoch, split_name="Val")

            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val AUC: {val_metrics['auc_roc']:.4f} - Val F1: {val_metrics['macro_f1']:.4f}")

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_metrics = val_metrics
                self.checkpointer.save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics, filename="best_qmedcam.pt"
                )

            scheduler.step()

        self.writer.close()
        return best_metrics
