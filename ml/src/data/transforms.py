"""
Image augmentation pipelines for Q-MedCAM using Albumentations.
Implements clinical dermoscopy transforms (rotations, color jitter, center crop).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Training transformations with medical-grade augmentations (REQ-2):
    - Random rotation (±180°)
    - Color jittering (brightness, contrast, saturation) simulating dermatoscope variations
    - Horizontal and vertical flips
    - Center/Random cropping to image_size
    - Normalization with ImageNet statistics
    """
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(image_size, image_size),
        A.Rotate(limit=180, p=0.8, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05,
            p=0.7
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def get_val_test_transforms(image_size: int = 224) -> A.Compose:
    """
    Deterministic validation and test transformations:
    - Resize and center crop
    - Normalization
    """
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
