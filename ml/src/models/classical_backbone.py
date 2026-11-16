"""
Classical CNN feature extractor and dimensionality bottleneck.
Supports ResNet18 and EfficientNet (B0, B2) backbones and maps latent representations to 4 dimensions.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b2, EfficientNet_B2_Weights
)

class ClassicalBackbone(nn.Module):
    """
    Modular feature extractor (ResNet18 / EfficientNet) with Linear(feature_dim, 4) + Tanh bottleneck.
    """
    def __init__(
        self,
        backbone_type: str = "resnet18",
        pretrained: bool = True,
        bottleneck_dim: int = 4,
        use_attention: bool = True
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.bottleneck_dim = bottleneck_dim
        self.use_attention = use_attention

        if self.backbone_type == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            # Individual references for fine-grained layer freezing/access
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.gradcam_target = self.layer4[-1].conv2
            self.feature_dim = 512

        elif self.backbone_type in ["efficientnet_b0", "efficientnet"]:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = efficientnet_b0(weights=weights)
            self.features = base.features
            self.gradcam_target = self.features[-1]
            self.feature_dim = 1280

        elif self.backbone_type == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            base = efficientnet_b2(weights=weights)
            self.features = base.features
            self.gradcam_target = self.features[-1]
            self.feature_dim = 1408

        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}. Supported: ['resnet18', 'efficientnet_b0', 'efficientnet_b2']")

        # Adaptive pooling & flattening
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Squeeze-and-Excitation Channel Attention
        if self.use_attention:
            hidden_dim = max(self.feature_dim // 8, 32)
            self.channel_attention = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.feature_dim),
                nn.Sigmoid()
            )
            # Initialize attention gate near 1.0 (identity passthrough at initialization)
            nn.init.constant_(self.channel_attention[2].bias, 2.0)

        # Classical bottleneck: Linear(feature_dim, 4) + Tanh (PRD Section 2.2)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, bottleneck_dim),
            nn.Tanh()
        )

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns target layer for Grad-CAM explainability hooks.
        """
        return self.gradcam_target

    def freeze_backbone(self):
        """
        Freezes convolutional feature extractor layers, leaving bottleneck and attention trainable.
        """
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_layer4_only: bool = False):
        """
        Unfreezes CNN backbone layers for joint fine-tuning.
        """
        if unfreeze_layer4_only and self.backbone_type == "resnet18":
            for param in self.layer4.parameters():
                param.requires_grad = True
        elif unfreeze_layer4_only and "efficientnet" in self.backbone_type:
            # Unfreeze only the final convolutional stage of EfficientNet
            for param in self.features[-1].parameters():
                param.requires_grad = True
        else:
            for param in self.features.parameters():
                param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts both the pooled feature representation and the 4-d latent bottleneck representation.
        
        Returns:
            Tuple of (features_flat, latent_4d)
        """
        feat = self.features(x)
        pooled = self.avgpool(feat)
        feat_flat = self.flatten(pooled)

        if self.use_attention:
            attn = self.channel_attention(feat_flat)
            attended_feat = feat_flat * attn
        else:
            attended_feat = feat_flat

        latent_4d = self.bottleneck(attended_feat)
        return feat_flat, latent_4d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, latent_4d = self.extract_features(x)
        return latent_4d
