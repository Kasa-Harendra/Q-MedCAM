"""
HAM10000 data splitting and leakage prevention utilities.
Strictly separates train/val/test partitions by lesion_id.
"""

from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# PRD mapping
BENIGN_CLASSES = {"nv", "bkl", "df", "vasc"}
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

def map_diagnosis_to_binary(dx: str) -> int:
    """
    Maps 7 granular HAM10000 diagnosis codes to binary:
    0: Benign ('nv', 'bkl', 'df', 'vasc')
    1: Malignant ('mel', 'bcc', 'akiec')
    """
    dx_lower = str(dx).lower().strip()
    if dx_lower in BENIGN_CLASSES:
        return 0
    elif dx_lower in MALIGNANT_CLASSES:
        return 1
    else:
        raise ValueError(f"Unknown diagnosis '{dx}'. Must be one of {BENIGN_CLASSES | MALIGNANT_CLASSES}")

def create_lesion_grouped_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the HAM10000 dataset into train, validation, and test sets
    strictly grouped by 'lesion_id' to prevent data leakage (REQ-1) while
    performing stratified allocation to preserve benign vs. malignant class balance.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Splits must sum to 1.0"
    
    df = df.copy()
    if "binary_label" not in df.columns:
        df["binary_label"] = df["dx"].apply(map_diagnosis_to_binary)

    rng = np.random.default_rng(random_state)

    # Determine per-lesion consensus label (mode or first)
    lesion_labels = (
        df.groupby("lesion_id")["binary_label"]
        .agg(lambda s: s.mode()[0] if len(s.mode()) > 0 else s.iloc[0])
        .to_dict()
    )

    train_lesions, val_lesions, test_lesions = set(), set(), set()

    # Stratified split: allocate lesions per class independently
    unique_classes = sorted(set(lesion_labels.values()))
    for label in unique_classes:
        lesions_in_class = [lesion for lesion, lab in lesion_labels.items() if lab == label]
        rng.shuffle(lesions_in_class)
        n = len(lesions_in_class)
        n_train = int(round(n * train_size))
        n_val = int(round(n * val_size))

        # Guarantee at least 1 lesion in val and test if class has >= 3 samples
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test

        train_lesions.update(lesions_in_class[:n_train])
        val_lesions.update(lesions_in_class[n_train:n_train + n_val])
        test_lesions.update(lesions_in_class[n_train + n_val:])

    train_df = df[df["lesion_id"].isin(train_lesions)].copy().reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_lesions)].copy().reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_lesions)].copy().reset_index(drop=True)

    # Strict zero-leakage verification
    assert_zero_leakage(train_df, val_df, test_df)

    return train_df, val_df, test_df

def assert_zero_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> bool:
    """
    Asserts zero overlap of lesion_id across train, validation, and test splits (AC-1).
    Raises AssertionError if leakage is detected.
    """
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    train_val_overlap = train_lesions.intersection(val_lesions)
    train_test_overlap = train_lesions.intersection(test_lesions)
    val_test_overlap = val_lesions.intersection(test_lesions)

    if train_val_overlap:
        raise AssertionError(f"Data leakage detected! {len(train_val_overlap)} lesions shared between train and val.")
    if train_test_overlap:
        raise AssertionError(f"Data leakage detected! {len(train_test_overlap)} lesions shared between train and test.")
    if val_test_overlap:
        raise AssertionError(f"Data leakage detected! {len(val_test_overlap)} lesions shared between val and test.")

    return True
