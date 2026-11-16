"""
Train Hybrid Q-MedCAM model with quantum circuit classification layer (Sprint 4 Milestone).
Executes dual learning rate AdamW optimization and full evaluation.
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hybrid_model import QMedCAMHybridModel
from src.data.dataset import HAM10000Dataset
from src.data.splits import create_lesion_grouped_splits
from src.data.transforms import get_train_transforms, get_val_test_transforms
from src.data.sampler import create_weighted_sampler
from src.training.trainer import HybridTrainer
from src.utils.device import get_torch_device

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Q-MedCAM Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--freeze_backbone", action="store_true", default=None, help="Freeze ResNet backbone and train only head")
    parser.add_argument("--unfreeze_layer4", action="store_true", default=False, help="Unfreeze ONLY layer4 for joint fine-tuning")
    parser.add_argument("--backbone_checkpoint", type=str, default="checkpoints/baseline_resnet18.pt", help="Path to pretrained backbone weights")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max samples per epoch for quantum training (e.g. 400)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max samples for validation during training (e.g. 200)")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max samples for test set evaluation (e.g. 300)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_torch_device()
    print(f"[Q-MedCAM] Device for classical backbone: {device}")

    meta_path = config["data"]["metadata_path"]
    images_dir = config["data"]["images_dir"]

    if not os.path.exists(meta_path):
        print(f"[Error] Metadata file not found at {meta_path}. Please place HAM10000_metadata.csv in {os.path.dirname(meta_path)}.")
        return

    df = pd.read_csv(meta_path)
    train_df, val_df, test_df = create_lesion_grouped_splits(
        df,
        train_size=config["data"]["train_split"],
        val_size=config["data"]["val_split"],
        test_size=config["data"]["test_split"],
        random_state=config["seed"]
    )
    print(f"[Data] Partitioned with zero leakage: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Subsample validation/test for fast quantum verification if requested
    if args.max_val_samples and args.max_val_samples < len(val_df):
        val_eval_df = val_df.sample(n=args.max_val_samples, random_state=config["seed"]).reset_index(drop=True)
    else:
        val_eval_df = val_df

    if args.max_test_samples and args.max_test_samples < len(test_df):
        test_eval_df = test_df.sample(n=args.max_test_samples, random_state=config["seed"]).reset_index(drop=True)
    else:
        test_eval_df = test_df

    train_dataset = HAM10000Dataset(train_df, images_dir, transform=get_train_transforms())
    val_dataset = HAM10000Dataset(val_eval_df, images_dir, transform=get_val_test_transforms())
    test_dataset = HAM10000Dataset(test_eval_df, images_dir, transform=get_val_test_transforms())

    sampler = (
        create_weighted_sampler(train_df["binary_label"], num_samples=args.max_train_samples)
        if config["training"]["use_weighted_sampler"]
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config["data"]["num_workers"]
    )
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

    model = QMedCAMHybridModel(
        pretrained=config["model"]["pretrained"],
        bottleneck_dim=config["model"]["bottleneck_dim"],
        ansatz_reps=config["quantum"]["ansatz_reps"],
        feature_map_reps=config["quantum"]["feature_map_reps"],
        entanglement=config["quantum"]["entanglement"],
        use_residual=config["model"].get("use_residual", True),
        multi_observable=config["model"].get("multi_observable", True),
        use_attention=config["model"].get("use_attention", True)
    )

    # Initialize backbone and classical skip connection from trained baseline checkpoint if available
    if args.backbone_checkpoint and os.path.exists(args.backbone_checkpoint):
        print(f"[Q-MedCAM] Transferring learned dermatological weights from {args.backbone_checkpoint}")
        model.load_baseline_weights(args.backbone_checkpoint)
        print("[Q-MedCAM] Backbone and classical skip connection weights successfully loaded!")

    should_freeze = args.freeze_backbone if args.freeze_backbone is not None else config["training"].get("freeze_backbone", False)
    if args.unfreeze_layer4:
        print("[Q-MedCAM] Fine-tuning layer4 + classical head + quantum head (early layers conv1-layer3 frozen).")
        model.unfreeze_backbone(unfreeze_layer4_only=True)
    elif should_freeze:
        print("[Q-MedCAM] Freezing ResNet backbone. Training ONLY the classical bottleneck, channel attention, and quantum head.")
        model.freeze_backbone()
    else:
        print("[Q-MedCAM] Joint end-to-end training (CNN backbone + bottleneck + quantum head).")

    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        backbone_lr=float(config["training"].get("backbone_lr", 1e-5)),
        classical_lr=float(config["training"]["classical_lr"]),
        quantum_lr=float(config["training"]["quantum_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        device=device,
        checkpoint_dir=config["training"]["checkpoint_dir"],
        log_dir=config["training"]["log_dir"]
    )

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    best_metrics = trainer.train(num_epochs=epochs)
    print("\n--- Best Validation Metrics ---")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")

    # Evaluate on held-out test split
    best_weights_path = os.path.join(config["training"]["checkpoint_dir"], "best_qmedcam.pt")
    if os.path.exists(best_weights_path):
        checkpoint = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = trainer.evaluate(epoch=0, data_loader=test_loader, split_name="Test")
        print("\n--- Final Held-out Test Metrics (Acceptance Criteria Check) ---")
        for k, v in test_metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
