"""
Loss function implementation combining BCEWithLogitsLoss and L2 bottleneck regularization (REQ-5).
"""

from typing import Optional
import torch
import torch.nn as nn

class HybridQMedCAMLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and L2 regularization targeting the classical bottleneck.
    """
    def __init__(self, l2_reg: float = 1e-4, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.l2_reg = l2_reg

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        bottleneck_layer: Optional[nn.Module] = None
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        if bottleneck_layer is not None and self.l2_reg > 0.0:
            l2_loss = 0.0
            for param in bottleneck_layer.parameters():
                l2_loss += torch.norm(param, p=2) ** 2
            return bce_loss + (0.5 * self.l2_reg * l2_loss)

        return bce_loss
