"""
Clinical visualization generator for Q-MedCAM (REQ-8).
Renders 300 DPI side-by-side composites: [Original Lesion] | [Overlayed Jet Grad-CAM].
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_overlay(
    raw_image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on top of a raw RGB lesion image.
    
    Args:
        raw_image_rgb: RGB image array of shape (H, W, 3), range [0, 255] or [0, 1].
        heatmap: 2D array of shape (H, W), normalized in [0, 1].
        alpha: Weight for the original image in blending (1 - alpha for heatmap).
        colormap: OpenCV colormap constant (defaults to COLORMAP_JET).
        
    Returns:
        blended_rgb: Overlayed RGB image array of shape (H, W, 3), range [0, 255].
    """
    if raw_image_rgb.dtype != np.uint8:
        if raw_image_rgb.max() <= 1.0:
            raw_image_rgb = (raw_image_rgb * 255).astype(np.uint8)
        else:
            raw_image_rgb = raw_image_rgb.astype(np.uint8)

    # Ensure heatmap dimensions match raw image
    H, W = raw_image_rgb.shape[:2]
    if heatmap.shape != (H, W):
        heatmap = cv2.resize(heatmap, (W, H))

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(raw_image_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return blended

def save_composite_xai_artifact(
    raw_image_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    image_id: str = "Unknown",
    p_malignant: float = 0.0,
    true_label: Optional[str] = None,
    dpi: int = 300
) -> str:
    """
    Creates and saves a high-resolution side-by-side composite artifact at 300 DPI (REQ-8).
    Format: [Original Lesion Image] | [Overlayed Grad-CAM Heatmap (Jet Colormap)]
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if raw_image_rgb.max() <= 1.0:
        raw_image_rgb = (raw_image_rgb * 255).astype(np.uint8)

    overlay = generate_overlay(raw_image_rgb, heatmap, alpha=0.5)

    risk_str = "HIGH RISK (Malignant)" if p_malignant >= 0.5 else "LOW RISK (Benign)"
    title_str = (
        f"Q-MedCAM Clinical Review | Lesion: {image_id}\n"
        f"Prediction: {risk_str} [P(Malignant) = {p_malignant:.2%}]"
    )
    if true_label is not None:
        title_str += f" | True Diagnosis: {true_label}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)

    axes[0].imshow(raw_image_rgb)
    axes[0].set_title("Original Dermoscopic Lesion", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Interpretability Map (Jet)", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    plt.suptitle(title_str, fontsize=12, y=0.98, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
