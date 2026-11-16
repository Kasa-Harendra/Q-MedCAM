"""
Side-by-side comparison of:
1. Classical Baseline ResNet18
2. Pure Bottleneck Hybrid QNN (Without skip connection)
3. Option C Residual Hybrid QNN (With residual skip connection)
"""

import os
import sys
import yaml
import torch
import pandas as pd
from torchvision.models import resnet18

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hybrid_model import QMedCAMHybridModel
from src.data.dataset import HAM10000Dataset
from src.data.splits import create_lesion_grouped_splits
from src.data.transforms import get_val_test_transforms
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_torch_device

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = get_torch_device()
    df = pd.read_csv("data/HAM10000_metadata.csv")
    train_df, val_df, test_df = create_lesion_grouped_splits(
        df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    test_320 = test_df.sample(n=320, random_state=42).reset_index(drop=True)
    images_dir = "data/images"
    test_ds = HAM10000Dataset(test_320, images_dir, transform=get_val_test_transforms())
    loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    results = {}

    # 1. Classical Baseline ResNet18
    print("[1/3] Evaluating Classical Baseline ResNet18...")
    baseline_model = resnet18()
    baseline_model.fc = torch.nn.Linear(baseline_model.fc.in_features, 1)
    baseline_model.load_state_dict(torch.load("checkpoints/baseline_resnet18.pt", map_location=device))
    baseline_model.to(device)
    baseline_model.eval()

    b_probs, b_targets = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = baseline_model(imgs).view(-1)
            b_probs.extend(torch.sigmoid(out).cpu().numpy())
            b_targets.extend(lbls.numpy())
    results["Classical Baseline (ResNet18)"] = compute_classification_metrics(b_targets, b_probs)

    # 2. Pure Bottleneck Hybrid QNN
    print("[2/3] Evaluating Pure Bottleneck Hybrid QNN...")
    pure_qnn = QMedCAMHybridModel(pretrained=False, bottleneck_dim=4, use_residual=False)
    pure_ckpt = torch.load("checkpoints/best_qmedcam_pure_bottleneck.pt", map_location=device)
    pure_state = pure_ckpt["model_state_dict"] if "model_state_dict" in pure_ckpt else pure_ckpt
    pure_qnn.load_state_dict(pure_state, strict=False)
    pure_qnn.to(device)
    pure_qnn.eval()

    p_probs, p_targets = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = pure_qnn(imgs).view(-1)
            p_probs.extend(torch.sigmoid(out).cpu().numpy())
            p_targets.extend(lbls.numpy())
    results["Pure Bottleneck Hybrid QNN"] = compute_classification_metrics(p_targets, p_probs)

    # 3. Option C Residual Hybrid QNN
    print("[3/3] Evaluating Option C Residual Hybrid QNN...")
    residual_qnn = QMedCAMHybridModel(pretrained=False, bottleneck_dim=4, use_residual=True)
    res_ckpt = torch.load("checkpoints/best_qmedcam.pt", map_location=device)
    res_state = res_ckpt["model_state_dict"] if "model_state_dict" in res_ckpt else res_ckpt
    residual_qnn.load_state_dict(res_state)
    residual_qnn.to(device)
    residual_qnn.eval()

    r_probs, r_targets = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = residual_qnn(imgs).view(-1)
            r_probs.extend(torch.sigmoid(out).cpu().numpy())
            r_targets.extend(lbls.numpy())
    results["Option C Residual Hybrid QNN"] = compute_classification_metrics(r_targets, r_probs)

    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Baseline ResNet18':<20} | {'Pure Hybrid QNN':<20} | {'Option C Residual QNN':<20}")
    print("="*80)
    for m in ["auc_roc", "accuracy", "sensitivity", "specificity", "macro_f1", "binary_f1", "tp", "fp", "tn", "fn"]:
        b_val = results["Classical Baseline (ResNet18)"][m]
        p_val = results["Pure Bottleneck Hybrid QNN"][m]
        r_val = results["Option C Residual Hybrid QNN"][m]
        
        if isinstance(b_val, float):
            print(f"{m:<25} | {b_val*100:<19.2f}% | {p_val*100:<19.2f}% | {r_val*100:<19.2f}%")
        else:
            print(f"{m:<25} | {b_val:<20} | {p_val:<20} | {r_val:<20}")
    print("="*80)

if __name__ == "__main__":
    main()
