"""
Unit test for end-to-end gradient flow across the quantum-classical boundary.
Verifies that gradients from quantum measurements propagate back into ResNet18 conv2.
"""

import torch
import pytest
from src.models.hybrid_model import QMedCAMHybridModel

def test_hybrid_gradient_flow():
    # Instantiate hybrid model with randomly initialized weights for fast unit testing
    model = QMedCAMHybridModel(
        pretrained=False,
        bottleneck_dim=4,
        ansatz_reps=2,
        feature_map_reps=1
    )
    model.train()

    # Create dummy input batch (2 samples, 3 channels, 224x224)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)

    # Forward pass
    logits = model(x)
    assert logits.shape == (2,)

    # Loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)

    # Backward pass
    loss.backward()

    # 1. Verify gradient reached the classical bottleneck
    bottleneck_linear = model.backbone.bottleneck[0]
    assert bottleneck_linear.weight.grad is not None
    assert not torch.isnan(bottleneck_linear.weight.grad).any()
    assert torch.abs(bottleneck_linear.weight.grad).sum() > 0.0

    # 2. Verify gradient reached the target Grad-CAM convolutional layer: layer4[-1].conv2
    target_layer = model.get_gradcam_target_layer()
    assert target_layer.weight.grad is not None
    assert not torch.isnan(target_layer.weight.grad).any()
    assert torch.abs(target_layer.weight.grad).sum() > 0.0

    # 3. Verify quantum parameters received gradients
    quantum_weights = model.quantum_layer.weight
    assert quantum_weights.grad is not None
    assert not torch.isnan(quantum_weights.grad).any()
