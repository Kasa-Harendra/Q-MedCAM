# Checkpoints Directory

This directory stores trained model checkpoints. Due to GitHub's 100 MB file size limit and best practices for large model binaries, `.pt` files are excluded from git.

### Available Checkpoints in Development:
- `baseline_resnet18.pt`: Trained classical ResNet18 baseline (~44.8 MB)
- `best_qmedcam.pt`: Option C Residual Hybrid Quantum-Classical model (~135 MB)
- `best_qmedcam_pure_bottleneck.pt`: Pure bottleneck hybrid ablation model (~44.8 MB)

### How to Train / Reproduce:
1. **Classical Baseline:**
   ```bash
   python scripts/train_baseline.py --epochs 10
   ```
2. **Option C Residual Hybrid Model:**
   ```bash
   python scripts/train_hybrid.py --epochs 3 --freeze_backbone --backbone_checkpoint checkpoints/baseline_resnet18.pt
   ```
3. **Inference with Pretrained Weights:**
   ```bash
   python scripts/run_inference_xai.py --image data/images/ISIC_0024310.jpg --checkpoint checkpoints/best_qmedcam.pt
   ```
