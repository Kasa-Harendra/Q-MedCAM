"""
Hybrid Quantum-Classical Neural Network (Q-MedCAM).
Connects ResNet18 backbone to Qiskit EstimatorQNN via TorchConnector.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector

from .classical_backbone import ClassicalBackbone
from .quantum_circuit import create_qmedcam_qnn

class QMedCAMHybridModel(nn.Module):
    """
    End-to-end Q-MedCAM model.
    Routes gradient from quantum expectation values back into classical convolutional layers.
    Includes:
    - Multi-Observable Pauli-Z measurement readout across all 4 qubits
    - Squeeze-and-Excitation channel attention on dermoscopic features
    - Residual quantum-classical skip connection (Option C) to eliminate the 128x bottleneck choke.
    """
    def __init__(
        self,
        pretrained: bool = True,
        bottleneck_dim: int = 4,
        ansatz_reps: int = 3,
        feature_map_reps: int = 2,
        entanglement: str = "full",
        use_residual: bool = True,
        multi_observable: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual
        self.multi_observable = multi_observable
        self.backbone = ClassicalBackbone(
            pretrained=pretrained,
            bottleneck_dim=bottleneck_dim,
            use_attention=use_attention
        )
        
        # Build Quantum EstimatorQNN with multi-qubit Pauli-Z measurement
        qnn, self.circuit = create_qmedcam_qnn(
            num_qubits=bottleneck_dim,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
            entanglement=entanglement,
            multi_observable=multi_observable
        )

        # TorchConnector enables automatic PyTorch autograd through the quantum circuit
        self.quantum_layer = TorchConnector(qnn)

        if self.multi_observable:
            # Multi-Observable Quantum Projection Head: maps 4D expectation values to scalar residual
            self.quantum_head = nn.Sequential(
                nn.Linear(bottleneck_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1)
            )
            # Gentle initialization for smooth residual learning
            nn.init.normal_(self.quantum_head[2].weight, std=0.01)
            nn.init.constant_(self.quantum_head[2].bias, 0.0)

        if self.use_residual:
            # Linear skip connection directly projecting 512-d ResNet features to logit space
            self.classical_head = nn.Linear(512, 1)
            # Learnable scaling factor for the quantum residual correction (init at 0.2)
            self.quantum_scale = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns layer4[-1].conv2 for Grad-CAM explainability hooks.
        """
        return self.backbone.get_gradcam_target_layer()

    def freeze_backbone(self, freeze_classical_head: bool = True):
        """
        Freezes ResNet CNN layers to train ONLY the classical bottleneck and quantum head.
        """
        self.backbone.freeze_backbone()
        if self.use_residual and freeze_classical_head:
            for param in self.classical_head.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_layer4_only: bool = False, unfreeze_classical_head: bool = True):
        """
        Unfreezes CNN layers for joint fine-tuning.
        """
        self.backbone.unfreeze_backbone(unfreeze_layer4_only=unfreeze_layer4_only)
        if self.use_residual and unfreeze_classical_head:
            for param in self.classical_head.parameters():
                param.requires_grad = True

    def _forward_quantum(self, latent_4d: torch.Tensor) -> torch.Tensor:
        original_device = latent_4d.device
        if original_device.type != "cpu":
            # Pass through CPU-based Qiskit Aer simulator
            latent_cpu = latent_4d.to("cpu")
            quantum_out = self.quantum_layer(latent_cpu)
            quantum_out = quantum_out.to(original_device)
        else:
            quantum_out = self.quantum_layer(latent_4d)

        if self.multi_observable:
            return self.quantum_head(quantum_out).view(-1)
        return quantum_out.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        Option C Residual:
            feat_512, latent_4d = extract_features(x)
            classical_logits = classical_head(feat_512)
            quantum_logits = quantum_layer(latent_4d)
            return classical_logits + quantum_scale * quantum_logits
        """
        if self.use_residual:
            feat_512, latent_4d = self.backbone.extract_features(x)
            classical_logits = self.classical_head(feat_512).view(-1)
            quantum_logits = self._forward_quantum(latent_4d).view(-1)
            return classical_logits + (self.quantum_scale * quantum_logits)
        else:
            latent_4d = self.backbone(x)
            quantum_logits = self._forward_quantum(latent_4d).view(-1)
            return quantum_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes malignant probability via Sigmoid.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def load_baseline_weights(self, checkpoint_path: str):
        """
        Transfers learned dermatological weights from classical baseline ResNet18.
        Loads convolutional layers into backbone and fc layer into classical_head.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone_weights = {k: v for k, v in state_dict.items() if not k.startswith("fc")}
        self.backbone.load_state_dict(backbone_weights, strict=False)

        if self.use_residual and "fc.weight" in state_dict and "fc.bias" in state_dict:
            self.classical_head.weight.data.copy_(state_dict["fc.weight"])
            self.classical_head.bias.data.copy_(state_dict["fc.bias"])
