"""
Evaluation metrics for clinical skin lesion screening:
AUC-ROC, Macro F1, Sensitivity (Recall), Specificity, Accuracy.
"""

from typing import Dict, Union
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_probs: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes clinical diagnostic metrics (PRD Section 3.2, 4, 6):
    - AUC-ROC
    - Macro F1
    - Accuracy
    - Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs).astype(float).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    # Safe AUC-ROC calculation (handles edge case where batch contains only 1 class)
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_probs))
    else:
        auc = 0.5

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc_roc": auc,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
