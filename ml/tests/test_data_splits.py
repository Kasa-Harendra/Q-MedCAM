"""
Unit tests for HAM10000 data ingestion, lesion grouping, and zero data leakage.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.splits import (
    map_diagnosis_to_binary,
    create_lesion_grouped_splits,
    assert_zero_leakage,
    BENIGN_CLASSES,
    MALIGNANT_CLASSES,
)
from src.data.sampler import create_weighted_sampler

def test_map_diagnosis_to_binary():
    # Verify benign classes map to 0
    for b in BENIGN_CLASSES:
        assert map_diagnosis_to_binary(b) == 0
        assert map_diagnosis_to_binary(b.upper()) == 0

    # Verify malignant classes map to 1
    for m in MALIGNANT_CLASSES:
        assert map_diagnosis_to_binary(m) == 1
        assert map_diagnosis_to_binary(m.upper()) == 1

    with pytest.raises(ValueError):
        map_diagnosis_to_binary("unknown_disease")

def test_zero_data_leakage():
    # Construct mock dataset where multiple images share the same lesion_id
    data = []
    # 50 unique lesions, each having 1 to 3 images
    np.random.seed(42)
    for i in range(50):
        lesion_id = f"HAM_LESION_{i:04d}"
        dx = np.random.choice(list(BENIGN_CLASSES | MALIGNANT_CLASSES))
        num_images = np.random.randint(1, 4)
        for j in range(num_images):
            image_id = f"ISIC_{i:04d}_{j}"
            data.append({"image_id": image_id, "lesion_id": lesion_id, "dx": dx})

    df = pd.DataFrame(data)

    train_df, val_df, test_df = create_lesion_grouped_splits(
        df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    # 1. Zero lesion overlap
    assert assert_zero_leakage(train_df, val_df, test_df) is True

    # 2. Total images match original
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # 3. Label column exists
    assert "binary_label" in train_df.columns
    assert "binary_label" in val_df.columns
    assert "binary_label" in test_df.columns

def test_leakage_assertion_raises_on_overlap():
    df1 = pd.DataFrame({"lesion_id": ["L1", "L2"], "image_id": ["I1", "I2"]})
    df2 = pd.DataFrame({"lesion_id": ["L2", "L3"], "image_id": ["I3", "I4"]})
    df3 = pd.DataFrame({"lesion_id": ["L4"], "image_id": ["I5"]})

    with pytest.raises(AssertionError, match="Data leakage detected"):
        assert_zero_leakage(df1, df2, df3)

def test_weighted_sampler():
    # 90 benign, 10 malignant
    labels = np.array([0] * 90 + [1] * 10)
    sampler = create_weighted_sampler(labels)
    assert len(sampler) == 100
    weights = sampler.weights.numpy()
    # Malignant samples should have higher weight than benign
    assert weights[95] > weights[5]
    # Ratio should be 90 / 10 = 9.0
    assert np.isclose(weights[95] / weights[5], 9.0)

def test_stratified_split_preserves_class_distribution():
    # Create dataset with imbalanced classes: 80% benign (40 lesions), 20% malignant (10 lesions)
    data = []
    for i in range(40):
        data.append({"image_id": f"IMG_B_{i}", "lesion_id": f"LESION_B_{i}", "dx": "nv"})
    for i in range(10):
        data.append({"image_id": f"IMG_M_{i}", "lesion_id": f"LESION_M_{i}", "dx": "mel"})
    df = pd.DataFrame(data)

    train_df, val_df, test_df = create_lesion_grouped_splits(
        df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    # Every split must have both classes represented
    assert (train_df["binary_label"] == 0).sum() > 0
    assert (train_df["binary_label"] == 1).sum() > 0
    assert (val_df["binary_label"] == 0).sum() > 0
    assert (val_df["binary_label"] == 1).sum() > 0
    assert (test_df["binary_label"] == 0).sum() > 0
    assert (test_df["binary_label"] == 1).sum() > 0

    # Malignant ratios in all splits should be approximately ~20% (between 10% and 35%)
    for split_df in [train_df, val_df, test_df]:
        malignant_ratio = (split_df["binary_label"] == 1).mean()
        assert 0.10 <= malignant_ratio <= 0.35, f"Expected ~0.20, got {malignant_ratio}"
