"""
Unit tests for HybridGradCAM and 300 DPI composite visualization.
"""

import os
import tempfile
import torch
import numpy as np
import pytest

from src.models.hybrid_model import QMedCAMHybridModel
from src.explainability.gradcam import HybridGradCAM
from src.explainability.visualizer import generate_overlay, save_composite_xai_artifact

def test_hybrid_gradcam_generation():
    model = QMedCAMHybridModel(
        pretrained=False,
        bottleneck_dim=4,
        ansatz_reps=2,
        feature_map_reps=1
    )
    cam_generator = HybridGradCAM(model)

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = cam_generator.generate_heatmap(x, target_class=1, retain_graph=True)

    # Validate output dimensions and normalization
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0 - 1e-6
    assert np.max(heatmap) <= 1.0 + 1e-6
    assert not np.isnan(heatmap).any()

    cam_generator.remove_hooks()

def test_composite_visualizer_creation():
    raw_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    heatmap = np.random.uniform(0.0, 1.0, (224, 224))

    # Test overlay generation
    overlay = generate_overlay(raw_image, heatmap, alpha=0.5)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_composite.png")
        saved_path = save_composite_xai_artifact(
            raw_image_rgb=raw_image,
            heatmap=heatmap,
            output_path=output_path,
            image_id="TEST_ISIC_0001",
            p_malignant=0.88,
            true_label="mel",
            dpi=150
        )
        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0
