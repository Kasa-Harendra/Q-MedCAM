"""
HAM10000 PyTorch Dataset implementation.
"""

import os
from typing import Optional, Callable, Dict, Any, Union, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    """
    Dermatoscopy Dataset for HAM10000 images and metadata.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_id: str) -> str:
        # Standard filenames in HAM10000 are <image_id>.jpg
        base_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        if os.path.exists(base_path):
            return base_path
        alt_path = os.path.join(self.images_dir, f"{image_id}.png")
        if os.path.exists(alt_path):
            return alt_path
        # Direct path if image_id already includes extension
        direct_path = os.path.join(self.images_dir, image_id)
        if os.path.exists(direct_path):
            return direct_path
        return base_path

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        image_path = self._resolve_image_path(image_id)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image at: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(row["binary_label"], dtype=torch.float32)

        if self.return_metadata:
            meta = {
                "image_id": image_id,
                "lesion_id": str(row.get("lesion_id", "")),
                "dx": str(row.get("dx", "")),
                "image_path": image_path
            }
            return image_tensor, label, meta

        return image_tensor, label
