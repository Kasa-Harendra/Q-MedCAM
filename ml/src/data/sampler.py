"""
Class balancing and WeightedRandomSampler implementation for HAM10000.
"""

from typing import List, Union, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(
    labels: Union[List[int], np.ndarray, pd.Series],
    num_samples: Optional[int] = None
) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle heavy class imbalance in HAM10000 (REQ-3).
    
    Args:
        labels: 1D array or Series of binary labels (0 or 1).
        num_samples: Optional number of samples to draw per epoch. If None, defaults to len(labels).
        
    Returns:
        WeightedRandomSampler instance with reciprocal class frequency weights.
    """
    if isinstance(labels, pd.Series):
        labels_arr = labels.to_numpy()
    else:
        labels_arr = np.array(labels)

    class_counts = np.bincount(labels_arr, minlength=2)
    # Guard against zero division
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)

    # Per-sample weights
    sample_weights = class_weights[labels_arr]
    n_samples = int(num_samples) if num_samples is not None else len(sample_weights)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=n_samples,
        replacement=True
    )
    return sampler
