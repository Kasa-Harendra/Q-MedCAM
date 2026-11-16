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
    threshold: float = 0.15,
    colormap: int = cv2.COLORMAP_JET,
    transparent_background: bool = True
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on top of a raw RGB lesion image.
    
    Args:
        raw_image_rgb: RGB image array of shape (H, W, 3), range [0, 255] or [0, 1].
        heatmap: 2D array of shape (H, W), normalized in [0, 1].
        alpha: Maximum blending weight for the colored heatmap overlay.
        threshold: Heatmap activation threshold below which overlay is transparent
                   (prevents blue-washing healthy background skin).
        colormap: OpenCV colormap constant (defaults to cv2.COLORMAP_JET).
        transparent_background: When True, healthy skin (heat < threshold) remains
                                completely natural and uncolored, while suspicious
                                focal regions glow with warm colormap tones.
        
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

    heatmap_clipped = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap_clipped)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if transparent_background:
        # Clinical Alpha Mask:
        # Zero tint on healthy skin below threshold; smooth ramp up to alpha on high activations
        alpha_factor = np.clip((heatmap_clipped - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
        alpha_map = alpha_factor * alpha
        alpha_3d = np.repeat(alpha_map[:, :, np.newaxis], 3, axis=2)
        blended = (raw_image_rgb.astype(np.float32) * (1.0 - alpha_3d) +
                   heatmap_colored_rgb.astype(np.float32) * alpha_3d).astype(np.uint8)
    else:
        blended = cv2.addWeighted(raw_image_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)

    return blended

def save_composite_xai_artifact(
    raw_image_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    image_id: str = "Unknown",
    p_malignant: float = 0.0,
    threshold: float = 0.65,
    true_label: Optional[str] = None,
    dpi: int = 300,
    three_panel: bool = True
) -> str:
    """
    Creates and saves a high-resolution clinical diagnostic composite artifact at 300 DPI (REQ-8).
    
    Layout (3-Panel Clinical Gold Standard):
    [Original Dermoscopy] | [Grad-CAM Activation Map] | [Focused Lesion Overlay]
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if raw_image_rgb.max() <= 1.0:
        raw_image_rgb = (raw_image_rgb * 255).astype(np.uint8)

    # Clean focused overlay with transparent healthy skin
    overlay = generate_overlay(
        raw_image_rgb,
        heatmap,
        alpha=0.60,
        threshold=0.18,
        colormap=cv2.COLORMAP_JET,
        transparent_background=True
    )

    risk_str = "HIGH RISK (Malignant)" if p_malignant >= threshold else "LOW RISK (Benign)"
    confidence_str = f"P(Malignant) = {p_malignant:.1%}"
    threshold_str = f"Decision Threshold: T = {threshold:.2f}"

    title_str = (
        f"Q-MedCAM Clinical Diagnostic Review | Lesion: {image_id}\n"
        f"Classification: {risk_str} [{confidence_str} | {threshold_str}]"
    )
    if true_label is not None:
        title_str += f" | Histopathology: {true_label}"

    if three_panel:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=dpi)

        # 1. Original Dermoscopic Lesion
        axes[0].imshow(raw_image_rgb)
        axes[0].set_title("1. Original Dermoscopy", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        # 2. Standalone Grad-CAM Heatmap with Colorbar
        im_heat = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title("2. Attention Heatmap (0.0 to 1.0)", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")
        cbar = plt.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Relative Activation", fontsize=9, fontweight="bold")

        # 3. Clinical Focused Overlay (Transparent Background)
        axes[2].imshow(overlay)
        axes[2].set_title("3. Clinical Focus Overlay", fontsize=11, fontweight="bold", pad=8)
        axes[2].axis("off")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
        axes[0].imshow(raw_image_rgb)
        axes[0].set_title("Original Dermoscopic Lesion", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Interpretability Map", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")

    plt.suptitle(title_str, fontsize=12, y=0.98, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
