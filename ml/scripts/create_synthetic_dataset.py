"""
Synthetic HAM10000 Dataset Generator for Q-MedCAM.
Creates realistic synthetic dermoscopic lesion images and structured metadata
for instant smoke testing, CI/CD validation, and pipeline verification.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2

BENIGN_DX = ["nv", "bkl", "df", "vasc"]
MALIGNANT_DX = ["mel", "bcc", "akiec"]
LOCALIZATIONS = ["back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest"]
DX_TYPES = ["histo", "consensus", "confocal", "follow_up"]

def generate_synthetic_lesion_image(
    image_size: int = 256,
    is_malignant: bool = False,
    seed: int = 42
) -> np.ndarray:
    """
    Synthesizes a realistic dermoscopic lesion image:
    - Base dermoscopic skin background (pinkish/tan)
    - Central pigment lesion with color variegation and border irregularity
    - Dermatoscope circular vignette effect
    """
    rng = np.random.default_rng(seed)
    
    # 1. Base skin tone (RGB ~ [220-240, 180-210, 160-190])
    base_color = np.array([
        rng.integers(215, 235),
        rng.integers(180, 205),
        rng.integers(160, 185)
    ], dtype=np.float32)
    
    img = np.ones((image_size, image_size, 3), dtype=np.float32) * base_color
    
    # Add subtle skin texture noise
    skin_noise = rng.normal(0, 4.0, (image_size, image_size, 3)).astype(np.float32)
    img = np.clip(img + skin_noise, 0, 255)

    # 2. Lesion geometry
    center_y = image_size // 2 + rng.integers(-10, 11)
    center_x = image_size // 2 + rng.integers(-10, 11)
    radius_y = rng.integers(35, 65)
    radius_x = rng.integers(35, 65)
    angle = rng.uniform(0, 180)

    # Create mask for lesion
    y, x = np.ogrid[:image_size, :image_size]
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    
    # Rotated coordinates relative to center
    xr = (x - center_x) * cos_a + (y - center_y) * sin_a
    yr = -(x - center_x) * sin_a + (y - center_y) * cos_a

    if is_malignant:
        # Malignant lesions feature irregular border perturbations (high-frequency asymmetry)
        theta = np.arctan2(yr, xr)
        perturbation = (
            0.20 * np.sin(3 * theta) +
            0.15 * np.cos(5 * theta) +
            0.10 * rng.normal(0, 0.15, theta.shape)
        )
        dist = (xr / (radius_x * (1 + perturbation)))**2 + (yr / (radius_y * (1 + perturbation)))**2
        lesion_mask = np.clip(1.0 - dist, 0.0, 1.0)
        lesion_mask = cv2.GaussianBlur(lesion_mask, (15, 15), 5.0)
        
        # Color variegation: dark brown, reddish-black, uneven pigment distribution
        lesion_color = np.array([
            rng.integers(40, 75),
            rng.integers(25, 50),
            rng.integers(20, 45)
        ], dtype=np.float32)
    else:
        # Benign nevi feature smooth, symmetrical oval borders
        dist = (xr / radius_x)**2 + (yr / radius_y)**2
        lesion_mask = np.clip(1.0 - dist, 0.0, 1.0)
        lesion_mask = cv2.GaussianBlur(lesion_mask, (11, 11), 3.0)
        
        # Homogeneous light/medium brown color
        lesion_color = np.array([
            rng.integers(90, 130),
            rng.integers(55, 85),
            rng.integers(35, 60)
        ], dtype=np.float32)

    # Blend lesion into skin
    mask_3d = np.repeat(lesion_mask[:, :, np.newaxis], 3, axis=2)
    img = img * (1.0 - mask_3d) + lesion_color * mask_3d

    # 3. Dermatoscope circular aperture vignette
    aperture_dist = ((x - image_size / 2) ** 2 + (y - image_size / 2) ** 2) / ((image_size * 0.58) ** 2)
    vignette = np.clip(1.0 - aperture_dist ** 3, 0.0, 1.0)
    vignette = np.repeat(vignette[:, :, np.newaxis], 3, axis=2)
    img = img * vignette

    return np.clip(img, 0, 255).astype(np.uint8)

def create_synthetic_dataset(
    output_dir: str = "data",
    num_lesions: int = 50,
    seed: int = 42
):
    """
    Creates a full synthetic dataset directory structure:
    - data/images/*.jpg
    - data/HAM10000_metadata.csv
    - data/sample/sample_lesion.jpg
    """
    rng = np.random.default_rng(seed)
    images_dir = os.path.join(output_dir, "images")
    sample_dir = os.path.join(output_dir, "sample")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    metadata_rows = []
    image_counter = 24300

    print(f"[Dataset Generator] Generating {num_lesions} synthetic lesion groups...")

    for i in range(num_lesions):
        lesion_id = f"HAM_{10000 + i:07d}"
        
        # ~25% malignant, ~75% benign (approximating HAM10000 distribution)
        is_malignant = rng.random() < 0.25
        if is_malignant:
            dx = rng.choice(MALIGNANT_DX)
        else:
            dx = rng.choice(BENIGN_DX)

        localization = rng.choice(LOCALIZATIONS)
        age = float(rng.choice([30, 35, 40, 45, 50, 55, 60, 65, 70, 75]))
        sex = rng.choice(["male", "female"])
        dx_type = rng.choice(DX_TYPES)

        # 1 to 3 images per lesion (to test group leakage prevention)
        num_images = rng.choice([1, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])

        for _ in range(num_images):
            image_counter += 1
            image_id = f"ISIC_{image_counter:07d}"
            image_filename = f"{image_id}.jpg"
            image_path = os.path.join(images_dir, image_filename)

            # Generate synthetic image
            img_bgr = generate_synthetic_lesion_image(
                image_size=256,
                is_malignant=is_malignant,
                seed=image_counter
            )
            # Save as BGR for cv2 imwrite
            cv2.imwrite(image_path, cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR))

            metadata_rows.append({
                "lesion_id": lesion_id,
                "image_id": image_id,
                "dx": dx,
                "dx_type": dx_type,
                "age": age,
                "sex": sex,
                "localization": localization
            })

    # Save metadata CSV
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(output_dir, "HAM10000_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"[Dataset Generator] Created {len(meta_df)} images across {num_lesions} lesions.")
    print(f"[Dataset Generator] Metadata saved to: {meta_path}")

    # Generate dedicated standalone sample lesion
    sample_path = os.path.join(sample_dir, "sample_lesion.jpg")
    sample_img = generate_synthetic_lesion_image(image_size=256, is_malignant=True, seed=999)
    cv2.imwrite(sample_path, cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
    print(f"[Dataset Generator] Sample lesion saved to: {sample_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic HAM10000 Dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num_lesions", type=int, default=50, help="Number of unique lesion IDs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    create_synthetic_dataset(output_dir=args.output_dir, num_lesions=args.num_lesions, seed=args.seed)

if __name__ == "__main__":
    main()
