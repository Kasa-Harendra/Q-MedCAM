"""
Unit tests for three-tier optimizer parameter grouping (REQ-4).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.hybrid_model import QMedCAMHybridModel
from src.training.trainer import HybridTrainer

def create_dummy_loader():
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=2)

def test_frozen_backbone_optimizer_groups():
    model = QMedCAMHybridModel(pretrained=False, bottleneck_dim=4, ansatz_reps=1, feature_map_reps=1)
    model.freeze_backbone()

    loader = create_dummy_loader()
    trainer = HybridTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        backbone_lr=1e-5,
        classical_lr=1e-3,
        quantum_lr=1e-4
    )

    # When backbone is frozen, backbone_params are not in optimizer
    group_names = [g["name"] for g in trainer.optimizer.param_groups]
    assert "backbone" not in group_names
    assert "bottleneck" in group_names
    assert "quantum" in group_names

    for group in trainer.optimizer.param_groups:
        if group["name"] == "bottleneck":
            assert group["lr"] == 1e-3
        elif group["name"] == "quantum":
            assert group["lr"] == 1e-4

def test_unfrozen_backbone_three_tier_optimizer_groups():
    model = QMedCAMHybridModel(pretrained=False, bottleneck_dim=4, ansatz_reps=1, feature_map_reps=1)
    # Ensure backbone is unfrozen
    model.unfreeze_backbone()

    loader = create_dummy_loader()
    trainer = HybridTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        backbone_lr=1e-5,
        classical_lr=1e-3,
        quantum_lr=1e-4
    )

    # All three groups must exist
    group_names = [g["name"] for g in trainer.optimizer.param_groups]
    assert "backbone" in group_names
    assert "bottleneck" in group_names
    assert "quantum" in group_names

    for group in trainer.optimizer.param_groups:
        if group["name"] == "backbone":
            assert group["lr"] == 1e-5
        elif group["name"] == "bottleneck":
            assert group["lr"] == 1e-3
        elif group["name"] == "quantum":
            assert group["lr"] == 1e-4
