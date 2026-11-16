"""
Classical CNN feature extractor and dimensionality bottleneck.
Truncates ResNet18 and maps latent representations to 4 dimensions.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ClassicalBackbone(nn.Module):
    """
    ResNet18 feature extractor with Linear(512, 4) + Tanh bottleneck.
    """
    def __init__(self, pretrained: bool = True, bottleneck_dim: int = 4, use_attention: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = resnet18(weights=weights)

        # CNN feature extractor up to layer4
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        # Adaptive pooling & flattening to 512 dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Squeeze-and-Excitation Channel Attention for dermatological feature salience
        self.use_attention = use_attention
        if self.use_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 512),
                nn.Sigmoid()
            )
            # Initialize attention gate near 1.0 (identity passthrough at initialization)
            nn.init.constant_(self.channel_attention[2].bias, 2.0)

        # Classical bottleneck: Linear(512, 4) + Tanh (PRD Section 2.2)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, bottleneck_dim),
            nn.Tanh()
        )

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the layer to hook for Grad-CAM: model.layer4[-1].conv2 (PRD REQ-7).
        """
        return self.layer4[-1].conv2

    def freeze_backbone(self):
        """
        Freezes ResNet convolutional layers (conv1 through layer4),
        leaving only the 512->4 classical bottleneck trainable.
        """
        for module in [self.conv1, self.bn1, self.relu, self.maxpool,
                       self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_layer4_only: bool = False):
        """
        Unfreezes CNN backbone layers for joint fine-tuning.
        """
        modules = [self.layer4] if unfreeze_layer4_only else [
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts both the 512-d pooled feature representation and the 4-d latent bottleneck representation.
        
        Returns:
            Tuple of (features_512, latent_4d)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat_512 = self.flatten(x)
        if self.use_attention:
            attn = self.channel_attention(feat_512)
            attended_feat = feat_512 * attn
        else:
            attended_feat = feat_512
        latent_4d = self.bottleneck(attended_feat)
        return feat_512, latent_4d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, latent_4d = self.extract_features(x)
        return latent_4d
