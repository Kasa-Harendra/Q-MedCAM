"""
Train classical ResNet18 baseline model (Sprint 1 Milestone).
Establishes baseline performance target (Target: >=85% Accuracy).
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import HAM10000Dataset
from src.data.splits import create_lesion_grouped_splits
from src.data.transforms import get_train_transforms, get_val_test_transforms
from src.data.sampler import create_weighted_sampler
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_torch_device

def main():
    parser = argparse.ArgumentParser(description="Train Classical ResNet18 Baseline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_torch_device()
    print(f"[Baseline] Using compute device: {device}")

    # Check data exists
    meta_path = config["data"]["metadata_path"]
    images_dir = config["data"]["images_dir"]

    if not os.path.exists(meta_path):
        print(f"[Error] Metadata file not found at {meta_path}.")
        return

    df = pd.read_csv(meta_path)
    train_df, val_df, test_df = create_lesion_grouped_splits(
        df,
        train_size=config["data"]["train_split"],
        val_size=config["data"]["val_split"],
        test_size=config["data"]["test_split"],
        random_state=config["seed"]
    )
    print(f"[Data] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = HAM10000Dataset(train_df, images_dir, transform=get_train_transforms())
    val_dataset = HAM10000Dataset(val_df, images_dir, transform=get_val_test_transforms())
    test_dataset = HAM10000Dataset(test_df, images_dir, transform=get_val_test_transforms())

    sampler = create_weighted_sampler(train_df["binary_label"]) if config["training"]["use_weighted_sampler"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config["data"]["num_workers"]
    )
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

    # Standard ResNet18 with binary classification head
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Differentiated learning rates for stable fine-tuning:
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": head_params, "lr": 1e-3, "weight_decay": 1e-4}
    ])
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    best_val_auc = 0.0

    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(config["training"]["checkpoint_dir"], "baseline_resnet18.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images).view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images).view(-1)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        metrics = compute_classification_metrics(val_targets, val_probs)
        print(f"Epoch {epoch} - Loss: {total_loss/len(train_loader):.4f} - Val AUC: {metrics['auc_roc']:.4f} - Val Acc: {metrics['accuracy']:.4f}")

        if metrics["auc_roc"] > best_val_auc:
            best_val_auc = metrics["auc_roc"]
            torch.save(model.state_dict(), save_path)

    # Evaluate on held-out test set
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_probs, test_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).view(-1)
            test_probs.extend(torch.sigmoid(logits).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_metrics = compute_classification_metrics(test_targets, test_probs)
    print("\n--- Baseline Test Set Results ---")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
