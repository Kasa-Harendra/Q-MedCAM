# Q-MedCAM: Hybrid Quantum-Classical Explainable Skin Lesion Screener

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.x-6929c4.svg)](https://qiskit.org/)
[![Tests](https://img.shields.io/badge/tests-9%20passed-brightgreen.svg)]()

Q-MedCAM is an explainable hybrid quantum-classical deep learning system designed for clinical dermatological screening. It couples a classical **ResNet18** convolutional feature extractor with a 4-qubit **IBM Qiskit Parameterized Quantum Circuit (PQC)** and routes analytical parameter-shift gradients back into the classical convolutional layers to generate sub-millimeter **Grad-CAM** visual heatmaps.

---

## 1. System Architecture

```
[Lesion Image: 3 x 224 x 224]
              │
              ▼
   Pretrained ResNet18 (layer1 - layer4) ───► Hook: layer4[-1].conv2 (Grad-CAM Target)
              │
              ▼
   AdaptiveAvgPool2d (512-dim)
              │
              ▼
   Linear(512 -> 4) + Tanh Bottleneck
              │
              ▼
   4-Qubit ZZFeatureMap (reps=2, entanglement='full')
              │
              ▼
   Trainable RealAmplitudes Ansatz (reps=3, 16 parameters)
              │
              ▼
   Pauli-Z Expectation Measurement <Z_0>
              │
              ▼
   TorchConnector (Exact Parameter-Shift Rule, input_gradients=True)
              │
              ▼
   Logits & Sigmoid -> P(Malignant)
              │
              ▼ (retain_graph=True)
   Grad-CAM Heatmap + 300 DPI Clinical Composite Artifact
```

---

## 2. Key Features & PRD Compliance

1. **Zero Data Leakage (REQ-1, AC-1):** Group-stratified train/val/test splits strictly partitioned on `lesion_id` to prevent multi-image patient leakage.
2. **Medical-Grade Augmentation (REQ-2):** $\pm 180^\circ$ rotation, dermatoscope color jittering, and center cropping implemented via Albumentations.
3. **Class Balancing (REQ-3):** `WeightedRandomSampler` handling benign vs. malignant skew in HAM10000.
4. **Differentiated Learning Rates (REQ-4):** Dual parameter group AdamW optimizer ($10^{-3}$ classical, $10^{-4}$ quantum).
5. **Loss & Regularization (REQ-5):** `BCEWithLogitsLoss` + L2 bottleneck regularization.
6. **Analytical Quantum Gradients (REQ-6):** Exact Parameter-Shift gradient computation through Qiskit `EstimatorQNN`.
7. **Gradient Persistence (REQ-7):** Autograd backward traversal with `retain_graph=True` across the `TorchConnector`.
8. **Clinical Artifacts (REQ-8, AC-4):** 300 DPI side-by-side composites (`[Original Lesion] | [Jet Grad-CAM Overlay]`).
9. **Fault Tolerance & Rollback (PRD Sec 4):** Automated checkpointing, NaN loss detection, and dynamic LR decay.

---

## 3. Hardware Requirements & System Compatibility

The hybrid computational graph dynamically allocates workloads across classical and quantum simulation backends:

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | 4-core Intel / AMD / Apple Silicon | 8+ core Apple Silicon (M1/M2/M3/M4) or Intel Core i7/i9 / AMD Ryzen 7/9 |
| **Memory (RAM)** | 8 GB System RAM | 16 GB+ System RAM (for multi-worker HAM10000 image streaming) |
| **GPU Acceleration** | Optional (Runs in CPU fallback mode) | **NVIDIA CUDA:** $\ge$ 4 GB VRAM (RTX 3060/4070/A100) or **Apple Silicon:** Metal Performance Shaders (`mps`) |
| **Quantum Simulator** | CPU Statevector Simulation (`qiskit-aer`) | NVIDIA GPU-accelerated statevector (`qiskit-aer-gpu` when CUDA is detected) |
| **Storage / Disk** | ~5 GB free space | $\ge$ 10 GB SSD (HAM10000 image dataset ~3 GB + checkpoints & logs) |

### Computational Allocation:
- **Classical CNN Feature Extractor:** Automatically utilizes Apple Silicon **MPS** or **NVIDIA CUDA** for fast ResNet18 convolutions and gradient calculations.
- **Quantum Parameterized Circuit (PQC):** The 4-qubit statevector representation ($2^4 = 16$ complex amplitudes) operates with minimal memory footprint (< 1 MB). Simulates parameter-shift analytical gradients on `qiskit-aer` / `qiskit.primitives.StatevectorEstimator` or `qiskit-aer-gpu`.

---

## 4. Installation & Setup

```bash
# 1. Clone repository & enter workspace
git clone <repo-url>
cd Quantum

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

Download the HAM10000 dataset:
```bash
bash scripts/download_dataset.sh
```
Or place the files manually:
- Metadata CSV: `data/HAM10000_metadata.csv`
- Lesion images: `data/images/*.jpg`

---

## 6. Usage & Execution

### Run Full Test Suite
```bash
pytest tests/ -v
```

### Train Pure Classical ResNet18 Baseline
```bash
python scripts/train_baseline.py --config configs/config.yaml
```

### Train End-to-End Hybrid Q-MedCAM Model
```bash
python scripts/train_hybrid.py --config configs/config.yaml
```

### Run Clinical Inference & Grad-CAM Heatmap Generation
```bash
python scripts/run_inference_xai.py \
    --image data/sample/sample_lesion.jpg \
    --output_dir outputs/xai \
    --dpi 300
```
This produces a 300 DPI composite image saved at `outputs/xai/qmedcam_xai_<image_id>.png`.
