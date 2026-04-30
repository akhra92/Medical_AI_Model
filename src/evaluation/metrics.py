"""
Evaluation utilities: metrics, plots, reports.

Computes:
  - Accuracy, per-class Precision / Recall / F1 (macro + weighted)
  - ROC-AUC (one-vs-rest, macro)
  - Confusion matrix (absolute + normalized)
  - Training history plots
  - Per-fold summary table
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

from src.config import CLASSES, RESULTS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Returns a dict of scalar metrics.
    y_proba: (N, num_classes) softmax probabilities
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
    }

    # ROC-AUC (multi-class one-vs-rest)
    try:
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["roc_auc_macro"] = float("nan")

    # Per-class metrics
    for cls in CLASSES:
        metrics[f"{cls}_f1"] = report[cls]["f1-score"]
        metrics[f"{cls}_precision"] = report[cls]["precision"]
        metrics[f"{cls}_recall"] = report[cls]["recall"]

    return metrics


def print_metrics(metrics: dict, label: str = ""):
    header = f"{'─'*50}\n  Metrics {label}\n{'─'*50}"
    print(header)
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print(f"  ROC-AUC (macro) : {metrics['roc_auc_macro']:.4f}")
    print(f"\n  Per-class F1:")
    for cls in CLASSES:
        print(f"    {cls:<12}: {metrics[f'{cls}_f1']:.4f}")
    print()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, t in zip(axes, [cm, cm_plot], ["Count", "Normalized"]):
        sns.heatmap(
            data, annot=True, fmt=".2f" if t == "Normalized" else "d",
            cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} ({t})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix → {save_path}")
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = None,
    title: str = "ROC Curves",
):
    """One-vs-rest ROC for each class."""
    n_classes = y_proba.shape[1]
    plt.figure(figsize=(8, 6))
    colors = ["steelblue", "tomato", "seagreen"]

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curves → {save_path}")
    plt.close()


def plot_training_history(history: dict, fold: int = 0, mode: str = "", save_path: str = None):
    """Plot train/val loss and val accuracy over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss Curve (Fold {fold}, {mode})")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="seagreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Validation Accuracy (Fold {fold}, {mode})")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history → {save_path}")
    plt.close()


def save_fold_summary(all_fold_metrics: list, mode: str):
    """Print and save a summary table across all folds."""
    df = pd.DataFrame(all_fold_metrics)
    df.index = [f"Fold {i}" for i in range(len(df))]
    mean_row = df.mean().rename("Mean")
    std_row = df.std().rename("Std")
    df = pd.concat([df, mean_row.to_frame().T, std_row.to_frame().T])

    path = os.path.join(RESULTS_DIR, f"fold_summary_{mode}.csv")
    df.to_csv(path)
    print(f"\n{'='*60}")
    print(f"  Cross-validation summary [{mode}]")
    print(f"{'='*60}")
    print(df[["accuracy", "macro_f1", "roc_auc_macro"]].to_string())
    print(f"\nSaved to {path}")
