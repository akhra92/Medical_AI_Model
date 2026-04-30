"""
Main training script.

Usage:
    python train.py --mode multimodal   # recommended
    python train.py --mode image
    python train.py --mode tabular

Runs N_FOLDS cross-validation, saves best checkpoints, prints per-fold metrics.
"""

import argparse
import os
import sys

import numpy as np
import torch

from src.config import N_FOLDS, SEED, RESULTS_DIR, CHECKPOINT_DIR
from src.preprocessing.data_loader import prepare_data, get_fold_loaders, get_test_loader
from src.models.multimodal_model import build_model
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves,
    plot_training_history, save_fold_summary,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Medical Image AI")
    parser.add_argument(
        "--mode", type=str, default="multimodal",
        choices=["multimodal", "image", "tabular"],
        help="Which model to train",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Disable mask-based ROI cropping for images",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Train image backbone from scratch (not recommended)",
    )
    parser.add_argument(
        "--folds", type=int, default=N_FOLDS,
        help="Number of cross-validation folds",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    use_mask = not args.no_mask
    pretrained = not args.no_pretrained

    print(f"\n{'#'*60}")
    print(f"  MEDICAL IMAGE AI — Training")
    print(f"  Mode: {mode} | Folds: {args.folds} | Mask: {use_mask}")
    print(f"{'#'*60}\n")

    # ── 1. Prepare data ───────────────────────────────────────────────────────
    print("Loading and preparing data...")
    data = prepare_data(mode=mode)
    tabular_input_dim = data["full_preprocessor"].input_dim
    print(f"  Tabular features: {tabular_input_dim}")
    print(f"  Train/val samples: {len(data['train_val_ids'])}")
    print(f"  Test samples: {len(data['test_ids'])}")

    all_fold_metrics = []
    fold_trainers = []
    fold_preprocessors = []

    # ── 2. Cross-validation ────────────────────────────────────────────────────
    for fold in range(args.folds):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold + 1} / {args.folds}")
        print(f"{'─'*60}")

        train_loader, val_loader, preprocessor = get_fold_loaders(
            data, fold_idx=fold, mode=mode, use_mask=use_mask
        )

        model = build_model(mode, tabular_input_dim=tabular_input_dim, pretrained=pretrained)
        trainer = Trainer(model=model, mode=mode, fold=fold)

        history = trainer.train(train_loader, val_loader, warmup_epochs=5)

        # ── Evaluate on validation set ─────────────────────────────────────
        y_true, y_pred, y_proba = trainer.predict(val_loader)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        print_metrics(metrics, label=f"Fold {fold+1} Validation")
        all_fold_metrics.append(metrics)
        fold_trainers.append(trainer)
        fold_preprocessors.append(preprocessor)

        # ── Plots ──────────────────────────────────────────────────────────
        plot_training_history(
            history, fold=fold+1, mode=mode,
            save_path=os.path.join(RESULTS_DIR, f"history_fold{fold+1}_{mode}.png"),
        )
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(RESULTS_DIR, f"cm_fold{fold+1}_{mode}.png"),
            title=f"Fold {fold+1} Validation",
        )

    # ── 3. Cross-validation summary ────────────────────────────────────────────
    save_fold_summary(all_fold_metrics, mode=mode)

    # ── 4. Final test evaluation (using best fold by val macro_f1) ─────────────
    best_fold = int(np.argmax([m["macro_f1"] for m in all_fold_metrics]))
    print(f"\nBest fold: {best_fold + 1} — evaluating on held-out test set...")

    test_loader = get_test_loader(
        data, preprocessor=fold_preprocessors[best_fold], mode=mode, use_mask=use_mask
    )
    y_true, y_pred, y_proba = fold_trainers[best_fold].predict(test_loader)
    test_metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(test_metrics, label="TEST SET")

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(RESULTS_DIR, f"cm_test_{mode}.png"),
        title="Test Set",
    )
    plot_roc_curves(
        y_true, y_proba,
        save_path=os.path.join(RESULTS_DIR, f"roc_test_{mode}.png"),
        title=f"Test ROC Curves [{mode}]",
    )

    # Save test metrics
    import pandas as pd
    pd.DataFrame([test_metrics]).to_csv(
        os.path.join(RESULTS_DIR, f"test_metrics_{mode}.csv"), index=False
    )
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
