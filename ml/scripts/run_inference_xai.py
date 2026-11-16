"""
Clinical inference and explainability script (Sprint 5 Milestone).
Produces 300 DPI composite diagnostic artifacts for dermatological review (REQ-8).
Runs 100% locally with zero external API transmission (HIPAA alignment).
"""

import os
import sys
from typing import Optional, Tuple
import argparse
import yaml
import cv2
import torch
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hybrid_model import QMedCAMHybridModel
from src.explainability.gradcam import HybridGradCAM
from src.explainability.visualizer import save_composite_xai_artifact
from src.data.transforms import get_val_test_transforms
from src.utils.device import get_torch_device

def run_single_inference(
    image_path: str,
    model: QMedCAMHybridModel,
    output_dir: str = "outputs/xai",
    device: Optional[torch.device] = None,
    dpi: int = 300,
    threshold: float = 0.65
) -> Tuple[str, float, str]:
    device = device or torch.device("cpu")
    model.eval()

    image_raw = cv2.imread(image_path)
    if image_raw is None:
        raise ValueError(f"Could not load image at {image_path}")
    image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    transform = get_val_test_transforms(image_size=224)
    augmented = transform(image=image_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # 1. Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        p_malignant = float(torch.sigmoid(logits).item())

    classification = "Malignant" if p_malignant >= threshold else "Benign"

    # 2. Grad-CAM generation
    input_tensor.requires_grad = True
    cam = HybridGradCAM(model)
    heatmap = cam.generate_heatmap(input_tensor, target_class=1, retain_graph=True)
    cam.remove_hooks()

    # 3. Save 300 DPI composite
    image_filename = os.path.basename(image_path)
    image_id = os.path.splitext(image_filename)[0]
    out_filename = f"qmedcam_xai_{image_id}.png"
    out_path = os.path.join(output_dir, out_filename)

    # Resize raw image to match 224x224 crop for aligned side-by-side display
    H, W = heatmap.shape
    image_resized = cv2.resize(image_rgb, (W, H))

    saved_file = save_composite_xai_artifact(
        raw_image_rgb=image_resized,
        heatmap=heatmap,
        output_path=out_path,
        image_id=image_id,
        p_malignant=p_malignant,
        dpi=dpi
    )
    return saved_file, p_malignant, classification

def main():
    parser = argparse.ArgumentParser(description="Run Q-MedCAM Clinical Inference & Grad-CAM XAI")
    parser.add_argument("--image", type=str, required=True, help="Path to input dermoscopic lesion image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_qmedcam.pt", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="outputs/xai", help="Directory to save 300 DPI artifacts")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for composite artifact")
    parser.add_argument("--threshold", type=float, default=0.65, help="Calibrated classification threshold (default: 0.65)")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone: resnet18, efficientnet_b0, efficientnet_b2")
    args = parser.parse_args()

    device = get_torch_device()
    print(f"[XAI Inference] Device: {device}")

    model = QMedCAMHybridModel(backbone_type=args.backbone, pretrained=False, bottleneck_dim=4)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"[XAI Inference] Loaded weights from {args.checkpoint}")
    else:
        print(f"[Warning] Checkpoint not found at {args.checkpoint}. Running with initialized weights for demonstration.")

    model = model.to(device)
    saved_path, p_malignant, classification = run_single_inference(
        image_path=args.image,
        model=model,
        output_dir=args.output_dir,
        device=device,
        dpi=args.dpi,
        threshold=args.threshold
    )
    print(f"[Result] Risk: {p_malignant*100:.2f}% | Calibrated Classification (T={args.threshold:.2f}): {classification}")
    print(f"[Success] Saved 300 DPI Clinical Composite Artifact to: {saved_path}")

if __name__ == "__main__":
    main()
