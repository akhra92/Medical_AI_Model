"""
Evaluation script: loads a saved checkpoint and evaluates on the test set.

Usage:
    python evaluate.py --mode multimodal --fold 0
    python evaluate.py --mode image --fold 2
    python evaluate.py --mode tabular --fold 0
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from src.config import CHECKPOINT_DIR, RESULTS_DIR, SEED
from src.preprocessing.data_loader import prepare_data, get_test_loader
from src.models.multimodal_model import build_model
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Medical Image AI")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "image", "tabular"])
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold's checkpoint to load (0-indexed)")
    parser.add_argument("--no-mask", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    use_mask = not args.no_mask

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold{args.fold}_{mode}.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print(f"\nLoading checkpoint: {ckpt_path}")
    data = prepare_data(mode=mode)
    tabular_input_dim = data["full_preprocessor"].input_dim

    model = build_model(mode, tabular_input_dim=tabular_input_dim, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    trainer = Trainer(model=model, mode=mode, fold=args.fold)

    test_loader = get_test_loader(
        data, preprocessor=data["full_preprocessor"],
        mode=mode, use_mask=use_mask,
    )
    y_true, y_pred, y_proba = trainer.predict(test_loader)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, label=f"Test Set (fold={args.fold}, mode={mode})")

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(RESULTS_DIR, f"eval_cm_{mode}_fold{args.fold}.png"),
        title=f"Evaluation — {mode}",
    )
    plot_roc_curves(
        y_true, y_proba,
        save_path=os.path.join(RESULTS_DIR, f"eval_roc_{mode}_fold{args.fold}.png"),
        title=f"ROC — {mode}",
    )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, f"eval_metrics_{mode}_fold{args.fold}.csv"), index=False
    )


if __name__ == "__main__":
    main()
