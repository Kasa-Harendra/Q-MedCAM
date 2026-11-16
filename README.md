# Q-MedCAM: Hybrid Quantum-Classical Explainable Skin Lesion Screener

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org)
[![Qiskit: 2.x](https://img.shields.io/badge/Qiskit-2.x-613197.svg)](https://qiskit.org)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![AUC-ROC: 92.91%](https://img.shields.io/badge/Test_AUC--ROC-92.91%25-success.svg)]()
[![Sensitivity: 90.91%](https://img.shields.io/badge/Sensitivity-90.91%25-blue.svg)]()

Q-MedCAM is an explainable hybrid quantum-classical deep learning system designed for dermatological screening of malignant skin lesions (Melanoma, Basal Cell Carcinoma). It integrates a deep convolutional neural network with parameterized quantum variational circuits and backward-propagated Grad-CAM explainability maps.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Input Dermoscopic Image   |
                      |       (3 x 224 x 224)       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    ResNet18 Feature Trunk   |
                      |   (Target: layer4.1.conv2)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    512-D Latent Vector      |
                      +-------+-------------+-------+
                              |             |
        Classical Residual    |             | Classical Bottleneck
        Skip Connection       |             | Linear(512, 4) + Tanh
                              |             v
                              |     +-----------------------------+
                              |     |   4-Qubit ZZFeatureMap      |
                              |     |     (Phase Encoding)        |
                              |     +--------------+--------------+
                              |                    |
                              |                    v
                              |     +-----------------------------+
                              |     |  RealAmplitudes Ansatz      |
                              |     |  (Entangled Variational PQC)|
                              |     +--------------+--------------+
                              |                    |
                              |                    v
                              |     +-----------------------------+
                              |     |   Pauli-Z Expectation       |
                              |     +--------------+--------------+
                              |                    |
                              v                    v
                      +-----------------------------+
                      | Logit = w_c * x + alpha * Q |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     P(Malignant) [0, 1]     |
                      |   Grad-CAM (retain_graph)   |
                      +-----------------------------+
```

---

## Monorepo Project Structure

```
Q-MedCAM/
|-- backend/          # Planned FastAPI application for serving inference
|-- frontend/         # Planned React / Next.js web application
|-- ml/               # Core Quantum Machine Learning engine
|   |-- configs/      # Hyperparameters and experiment configurations
|   |-- data/         # Metadata, synthetic generators, and dataset loaders
|   |-- outputs/xai/  # 300 DPI Grad-CAM composite clinical artifacts
|   |-- scripts/      # Training, inference, and benchmarking scripts
|   |-- src/          # Model architectures, quantum circuits, trainer, XAI
|   |-- tests/        # Unit test suite (12/12 passing)
|   |-- pytest.ini    # Test runner configuration
|   `-- requirements.txt
|-- docs/             # Technical specifications, PRD, and benchmark reports
|-- .gitignore
`-- README.md
```

---

## Key Benchmark Findings

Evaluated on held-out test lesions from the HAM10000 dataset:

| Clinical Metric | Baseline ResNet18 | Pure Bottleneck QNN | Option C Residual QNN |
| :--- | :---: | :---: | :---: |
| **AUC-ROC** | 92.96% | 82.65% | **92.91%** |
| **Sensitivity (Recall)** | 90.91% | 69.70% | **90.91%** |
| **Specificity** | 79.92% | 79.13% | **77.95%** |
| **Accuracy** | 82.19% | 77.19% | **80.62%** |
| **Macro F1** | 77.74% | 70.19% | **76.20%** |
| **Missed Cancers (FN)** | 6 | 20 | **6 (-70%)** |

---

## Getting Started

### 1. Installation

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Generate Grad-CAM Diagnostic Artifacts

```bash
python scripts/run_inference_xai.py \
  --image data/sample/sample_lesion.jpg \
  --checkpoint checkpoints/best_qmedcam.pt \
  --output_dir outputs/xai
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
