# Product Requirements Document (PRD): Q-MedCAM

**Project Name:** Hybrid Quantum-Classical Explainable Skin Lesion Screener (Q-MedCAM)
**Version:** 2.0 (Detailed Implementation & Scaling Specification)
**Target Domain:** Quantum Machine Learning (QML), Healthcare Diagnostics, Explainable AI (XAI)

---

## 1. Executive Summary & Clinical Objective

### 1.1 Overview & Motivation
The **Q-MedCAM** project pioneers the intersection of Noisy Intermediate-Scale Quantum (NISQ) computing and clinical dermatology. While classical Deep Convolutional Neural Networks (CNNs) achieve high accuracy in skin lesion classification, their "black-box" nature hinders clinical adoption. Furthermore, quantum machine learning theorizes an exponential advantage in mapping complex, high-dimensional feature spaces. Q-MedCAM combines a classical CNN feature extractor with an IBM Qiskit parameterized quantum circuit (PQC) acting as the classifier. Crucially, it routes gradients from the quantum measurement back to the classical layers to generate **Grad-CAM** visual heatmaps, providing clinicians with interpretable, localized diagnostic evidence.

### 1.2 Core Objectives & Hypotheses
- **Quantum Feature Encoding:** Prove that classical latent spaces can be mapped into a quantum Hilbert space (using phase-amplitude encoding) without losing spatial gradient information critical for explainability.
- **Explainable Diagnostics:** Generate sub-millimeter accurate Grad-CAM heatmaps showing exactly which pixel regions (e.g., irregular borders, color variegation) drove the quantum circuit's malignant/benign prediction.
- **Clinical Integration Standard:** Establish a modular pipeline utilizing the HAM10000 dataset that outputs both a statistical prediction and a visual artifact suitable for a dermatologist's secondary review.

---

## 2. Detailed System Architecture

### 2.1 Technology Stack & Dependencies
- **Quantum Computing:** IBM Qiskit $\ge$ 0.44, `qiskit-machine-learning`, `qiskit-aer` (for high-performance GPU-accelerated statevector simulation).
- **Deep Learning:** PyTorch $\ge$ 2.0, Torchvision.
- **XAI Framework:** `pytorch-grad-cam` (specifically targeting the hybrid architecture).
- **Data Engineering:** Pandas, OpenCV, Albumentations (for medical-grade image augmentation).

### 2.2 End-to-End Computational Graph
1. **Input Tensor:** $X \in \mathbb{R}^{3 \times 224 \times 224}$ (RGB Image).
2. **CNN Backbone:** Pretrained ResNet18 (weights truncated at `layer4`). 
   - *Target for Grad-CAM:* Hook registered at `model.layer4[-1].conv2`.
3. **Adaptive Pooling & Flattening:** Reduces feature maps to $512 \times 1 \times 1$, flattened to a 512-dimensional vector.
4. **Classical Bottleneck:** Fully Connected (Linear) Layer with Tanh activation mapping $\mathbb{R}^{512} \rightarrow \mathbb{R}^{4}$. This exact matching of 4 dimensions corresponds to a 4-qubit quantum system.
5. **Quantum Feature Map:** Qiskit `ZZFeatureMap` (4 qubits, 2 repetitions). Maps the classical 4D vector into a highly entangled quantum state space.
6. **Quantum Ansatz:** `RealAmplitudes` or `TwoLocal` circuit (entanglement='full', reps=3). Contains the trainable parameters $\theta$.
7. **Quantum Measurement:** Pauli-Z expectation values on specific qubits.
8. **TorchConnector:** PyTorch wrapper that handles forward passes (simulating the circuit) and backward passes (computing parameter gradients via the parameter-shift rule).
9. **Final Output:** Sigmoid activation mapping the expectation value to $P(\text{Malignant})$.

---

## 3. Expanded Functional Requirements

### 3.1 Advanced Data Ingestion & Augmentation
- **REQ-1 (Data Integrity):** The system must ingest HAM10000 metadata (CSV), map 7 granular diagnoses into binary classes (Class 0: `nv`, `bkl`, `df`, `vasc`; Class 1: `mel`, `bcc`, `akiec`), and strictly separate train/val/test sets by `lesion_id` to prevent data leakage from multiple images of the same lesion.
- **REQ-2 (Medical Augmentation):** The data loader must apply clinically relevant augmentations during training: random rotations ($\pm 180^\circ$), color jittering (simulating different dermatoscope lighting), and center cropping.
- **REQ-3 (Class Imbalance):** The `DataLoader` must utilize a `WeightedRandomSampler` to handle the heavy skew toward benign Nevi (`nv`) in the HAM10000 dataset.

### 3.2 Hybrid Optimization & Training Mechanics
- **REQ-4 (Differentiated Learning Rates):** The optimizer (AdamW) must support parameter groups, applying a standard learning rate (e.g., $1 \times 10^{-3}$) to classical bottleneck layers and a lower learning rate (e.g., $1 \times 10^{-4}$) to the quantum parameters to maintain quantum state stability.
- **REQ-5 (Loss Landscape):** Training must utilize Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) combined with L2 regularization to prevent overfitting in the classical bottleneck.
- **REQ-6 (Quantum Gradients):** The system must compute gradients for the quantum layers using the parameter-shift rule, ensuring exact gradient calculations rather than finite-difference approximations.

### 3.3 Explainability & Interface
- **REQ-7 (Gradient Persistence):** The pipeline must set `retain_graph=True` during the backward pass of the Grad-CAM generation to ensure the connection between the quantum output and classical convolutional layers remains intact.
- **REQ-8 (Visual Output):** The output must be a side-by-side composite image: [Original Lesion Image] | [Overlayed Grad-CAM Heatmap (Jet Colormap)], saved at 300 DPI for clinical review.

---

## 4. Non-Functional Requirements & Compliance

- **Performance & Scalability:** The statevector simulation must be optimized using `qiskit-aer-gpu` if NVIDIA hardware is detected, reducing the simulation overhead of the 4-qubit circuit by at least 40% compared to CPU execution.
- **Reproducibility & Auditing:** The system must log all hyperparameters, random seeds, and epoch metrics (Loss, Accuracy, AUC-ROC, F1-Score) to TensorBoard or Weights & Biases (W&B).
- **Patient Data Privacy (HIPAA Alignment):** All inference modules must run entirely locally or on secure isolated compute instances. The code must not transmit image data to external third-party APIs during inference.
- **Fault Tolerance:** The training loop must include automated checkpointing. If a gradient vanishing issue occurs between the `TorchConnector` and the CNN, the script must catch the `NaN` loss, rollback to the previous epoch's weights, and adjust the learning rate dynamically.

---

## 5. Implementation Roadmap & Sprints

| Sprint | Milestone | Key Engineering Deliverables | Estimated Duration |
| :--- | :--- | :--- | :--- |
| **Sprint 1** | **Data & Classical Baseline** | EDA, Lesion ID grouping, Stratified splits. Implement and train standard ResNet18 (pure classical) to establish performance baseline (Target: 85% Accuracy). | 2 Weeks |
| **Sprint 2** | **Quantum Circuit Design** | Build `ZZFeatureMap` and `TwoLocal` ansatz in Qiskit. Benchmark simulator performance. Configure `EstimatorQNN`. | 1 Week |
| **Sprint 3** | **Hybrid Integration** | Wire ResNet18 to the Quantum Layer via `TorchConnector`. Resolve tensor dimension mismatches. Implement distinct optimizer groups. | 2 Weeks |
| **Sprint 4** | **Hybrid Training & Tuning** | Execute end-to-end training. Monitor for barren plateaus in quantum gradients. Tune bottleneck dimensions (4 vs 8 qubits). | 2 Weeks |
| **Sprint 5** | **XAI / Grad-CAM** | Attach forward/backward hooks. Ensure PyTorch Autograd successfully traverses the `TorchConnector`. Generate and validate heatmap outputs. | 1.5 Weeks |

---

## 6. Strict Acceptance Criteria

1. **Data Leakage Prevented:** Zero overlap of `lesion_id` between training, validation, and test datasets.
2. **Hybrid Convergence:** The hybrid model loss curve monotonically decreases over the first 5 epochs, proving successful gradient flow between classical and quantum layers.
3. **Metric Thresholds:** The model achieves an AUC-ROC $> 0.80$ and a macro F1-score $> 0.75$ on the held-out test set.
4. **XAI Verification:** Grad-CAM heatmaps are successfully generated for 100% of the test set images. A qualitative check confirms the heatmaps localize to the actual lesion in the image, rather than background skin or image corners.