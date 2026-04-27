"""
Image preprocessing utilities.

Pipeline:
  1. Load PNG image (RGB) and corresponding segmentation mask (grayscale)
  2. Optionally apply mask to zero-out background (focus on ROI)
  3. Resize to IMAGE_SIZE × IMAGE_SIZE
  4. Apply augmentations (train only)
  5. Normalize with ImageNet statistics (for pretrained backbone)
"""

import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch

from src.config import (
    IMAGE_SIZE, USE_MASKS,
    IMAGENET_MEAN, IMAGENET_STD,
    AUG_H_FLIP, AUG_V_FLIP, AUG_ROTATION,
    AUG_BRIGHTNESS, AUG_CONTRAST,
    IMAGE_DIR, CLASSES,
)


def load_image(path: str) -> np.ndarray:
    """Load an image as an RGB numpy array (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    """Load a binary mask as a grayscale numpy array (H, W)."""
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    # Binarize: anything > 0 is foreground
    return (arr > 0).astype(np.uint8)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero-out pixels outside the mask.
    image: (H, W, 3), mask: (H, W) binary
    Returns: (H, W, 3)
    """
    masked = image.copy()
    masked[mask == 0] = 0
    return masked


def get_image_path(patient_id: str, class_name: str) -> str:
    return os.path.join(IMAGE_DIR, class_name, "images", f"{patient_id}.png")


def get_mask_path(patient_id: str, class_name: str) -> str:
    return os.path.join(IMAGE_DIR, class_name, "masks", f"{patient_id}.png")


# ─── torchvision Transform pipelines ──────────────────────────────────────────

def get_train_transforms() -> T.Compose:
    transforms = [
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if AUG_H_FLIP:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    if AUG_V_FLIP:
        transforms.append(T.RandomVerticalFlip(p=0.5))
    if AUG_ROTATION > 0:
        transforms.append(T.RandomRotation(degrees=AUG_ROTATION))
    if AUG_BRIGHTNESS > 0 or AUG_CONTRAST > 0:
        transforms.append(T.ColorJitter(
            brightness=AUG_BRIGHTNESS,
            contrast=AUG_CONTRAST,
        ))
    transforms += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_and_preprocess(
    patient_id: str,
    class_name: str,
    transform,
    use_mask: bool = USE_MASKS,
) -> torch.Tensor:
    """
    Full pipeline: load image (+ optionally mask) → apply transform → tensor.
    Returns tensor of shape (3, H, W).
    """
    img_path = get_image_path(patient_id, class_name)
    image = load_image(img_path)

    if use_mask:
        mask_path = get_mask_path(patient_id, class_name)
        if os.path.exists(mask_path):
            mask = load_mask(mask_path)
            # Resize mask to match image before applying
            if mask.shape[:2] != image.shape[:2]:
                mask_pil = Image.fromarray(mask * 255).resize(
                    (image.shape[1], image.shape[0]), Image.NEAREST
                )
                mask = (np.array(mask_pil) > 0).astype(np.uint8)
            image = apply_mask(image, mask)

    return transform(image)


def build_image_registry() -> dict:
    """
    Returns {patient_id: class_name} for every image in dataset1.
    """
    registry = {}
    for cls in CLASSES:
        img_dir = os.path.join(IMAGE_DIR, cls, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if fname.endswith(".png"):
                pid = fname.replace(".png", "")
                registry[pid] = cls
    return registry
