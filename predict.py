"""
Inference script: run prediction on a single image (+ optional tabular data).

Usage:
    # Image-only prediction
    python predict.py --image path/to/image.png --mode image --fold 0

    # Multimodal prediction (image + patient data)
    python predict.py --image path/to/image.png --patient-id MB-1234 --mode multimodal --fold 0

    # Batch prediction from a folder
    python predict.py --image-dir path/to/folder/ --mode image --fold 0
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image

from src.config import CHECKPOINT_DIR, CLASSES, SEED
from src.preprocessing.image_preprocessing import get_val_transforms, load_image, load_mask, apply_mask
from src.preprocessing.tabular_preprocessing import load_tabular_data, TabularPreprocessor, PREPROCESSOR_PATH
from src.models.multimodal_model import build_model
from src.training.trainer import get_device

torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Medical Image AI — Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to a single PNG image")
    parser.add_argument("--mask", type=str, default=None, help="Path to corresponding mask (optional)")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID for tabular lookup")
    parser.add_argument("--image-dir", type=str, default=None, help="Folder of PNG images (batch mode)")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "tabular", "multimodal"])
    parser.add_argument("--fold", type=int, default=0)
    return parser.parse_args()


def load_model(mode: str, fold: int, tabular_input_dim: int) -> torch.nn.Module:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold{fold}_{mode}.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model = build_model(mode, tabular_input_dim=tabular_input_dim, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(img_path: str, mask_path: str = None) -> torch.Tensor:
    img = load_image(img_path)
    if mask_path and os.path.exists(mask_path):
        mask = load_mask(mask_path)
        if mask.shape[:2] != img.shape[:2]:
            from PIL import Image as PILImage
            m_pil = PILImage.fromarray(mask * 255).resize(
                (img.shape[1], img.shape[0]), PILImage.NEAREST
            )
            mask = (np.array(m_pil) > 0).astype(np.uint8)
        img = apply_mask(img, mask)
    transform = get_val_transforms()
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def predict_single(model, image_tensor, tabular_tensor, mode, device):
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if mode == "image":
            logits = model(image_tensor)
        elif mode == "tabular":
            logits = model(tabular_tensor.to(device))
        else:
            logits = model(image_tensor, tabular_tensor.to(device))

    proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(proba.argmax())
    return CLASSES[pred_idx], proba


def main():
    args = parse_args()
    device = get_device()

    # Load tabular preprocessor if needed
    tabular_input_dim = 1  # placeholder for image-only
    preprocessor = None
    if args.mode in ("multimodal", "tabular"):
        assert os.path.exists(PREPROCESSOR_PATH), (
            f"Preprocessor not found at {PREPROCESSOR_PATH}. "
            "Run train.py first to generate it."
        )
        preprocessor = TabularPreprocessor.load(PREPROCESSOR_PATH)
        tabular_input_dim = preprocessor.input_dim

    model = load_model(args.mode, args.fold, tabular_input_dim)
    model = model.to(device)

    def get_tabular_tensor(patient_id):
        if preprocessor is None or patient_id is None:
            return torch.zeros(1, tabular_input_dim, dtype=torch.float32)
        df = load_tabular_data()
        row = df[df["Patient ID"] == patient_id]
        if row.empty:
            raise ValueError(f"Patient ID '{patient_id}' not found in dataset.")
        X, _, _ = preprocessor.transform(row)
        return torch.tensor(X, dtype=torch.float32)

    if args.image:
        # Single image prediction
        img_tensor = preprocess_image(args.image, args.mask)
        tab_tensor = get_tabular_tensor(args.patient_id)
        cls_name, proba = predict_single(model, img_tensor, tab_tensor, args.mode, device)

        print(f"\nPrediction: {cls_name.upper()}")
        print("Class probabilities:")
        for c, p in zip(CLASSES, proba):
            bar = "█" * int(p * 30)
            print(f"  {c:<12}: {p:.4f}  {bar}")

    elif args.image_dir:
        # Batch prediction
        print(f"\nBatch prediction from: {args.image_dir}")
        header = f"{'Patient ID':<15} {'Prediction':<12} " + " ".join(f"{c:>10}" for c in CLASSES)
        print(header)
        print("─" * len(header))
        for fname in sorted(os.listdir(args.image_dir)):
            if not fname.endswith(".png"):
                continue
            pid = fname.replace(".png", "")
            img_path = os.path.join(args.image_dir, fname)
            img_tensor = preprocess_image(img_path)
            try:
                tab_tensor = get_tabular_tensor(pid)
            except ValueError as e:
                print(f"  WARNING: {e} — using zero tabular vector", file=sys.stderr)
                tab_tensor = torch.zeros(1, tabular_input_dim, dtype=torch.float32)
            cls_name, proba = predict_single(model, img_tensor, tab_tensor, args.mode, device)
            proba_str = " ".join(f"{p:>10.4f}" for p in proba)
            print(f"  {pid:<13} {cls_name:<12} {proba_str}")
    else:
        print("Provide --image or --image-dir. See --help for usage.")


if __name__ == "__main__":
    main()
