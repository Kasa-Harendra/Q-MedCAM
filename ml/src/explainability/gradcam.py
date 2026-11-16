"""
Hybrid Grad-CAM implementation for Q-MedCAM.
Extracts gradient-weighted class activation maps across the quantum-classical boundary (REQ-7).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridGradCAM:
    """
    Grad-CAM for hybrid architectures with backward gradient flow from quantum outputs.
    Ensures retain_graph=True during backprop (REQ-7).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.get_gradcam_target_layer()

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Capture gradients flowing backwards into the target layer
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        retain_graph: bool = True
    ) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for a single image tensor (1, 3, H, W).
        
        Args:
            input_tensor: PyTorch tensor of shape (1, 3, H, W).
            target_class: 1 for malignant probability, 0 for benign.
            retain_graph: Set to True to retain computational graph (REQ-7).
            
        Returns:
            heatmap: 2D numpy array normalized to [0.0, 1.0] of shape (H, W).
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Forward pass
        logits = self.model(input_tensor)
        prob_malignant = torch.sigmoid(logits)

        # Target score for backpropagation
        score = prob_malignant if target_class == 1 else (1.0 - prob_malignant)

        # Zero gradients
        self.model.zero_grad()

        # REQ-7: Backward pass with retain_graph=True
        score.backward(retain_graph=retain_graph)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: (1, Channels, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain only features that have a positive influence on target class
        cam = F.relu(cam)

        # Upsample to input image size
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()
