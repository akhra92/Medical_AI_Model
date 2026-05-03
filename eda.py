"""
Exploratory Data Analysis (EDA) script.
Generates plots and statistics for all three datasets.

Run: python eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

from src.config import (
    IMAGE_DIR, PATIENT_CSV, BIOMARKER_CSV,
    CLASSES, CLASS_TO_IDX, RESULTS_DIR,
)
from src.preprocessing.tabular_preprocessing import load_tabular_data
from src.preprocessing.image_preprocessing import build_image_registry

EDA_DIR = os.path.join(RESULTS_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")


# ─── 1. Class distribution ────────────────────────────────────────────────────

def plot_class_distribution():
    counts = {}
    registry = build_image_registry()
    for cls in CLASSES:
        counts[cls] = sum(1 for v in registry.values() if v == cls)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["steelblue", "tomato", "seagreen"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontsize=11)
    ax.set_title("Class Distribution (Dataset 1 — Ultrasound Images)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("  Class counts:", counts)


# ─── 2. Sample images ─────────────────────────────────────────────────────────

def plot_sample_images(n_per_class: int = 3):
    fig, axes = plt.subplots(len(CLASSES), n_per_class * 2,
                             figsize=(n_per_class * 4, len(CLASSES) * 2.5))

    for row, cls in enumerate(CLASSES):
        img_dir = os.path.join(IMAGE_DIR, cls, "images")
        mask_dir = os.path.join(IMAGE_DIR, cls, "masks")
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])[:n_per_class]

        for col, fname in enumerate(files):
            img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
            ax_img = axes[row, col * 2]
            ax_img.imshow(img)
            ax_img.axis("off")
            # Use text() instead of set_ylabel so it shows with axis("off")
            if col == 0:
                ax_img.text(-0.15, 0.5, cls, transform=ax_img.transAxes,
                            fontsize=12, va="center", ha="right", fontweight="bold")

            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert("L")
                if mask_pil.size != (img.shape[1], img.shape[0]):
                    mask_pil = mask_pil.resize((img.shape[1], img.shape[0]), Image.NEAREST)
                mask = np.array(mask_pil)
                overlay = img.copy()
                overlay[mask > 0] = [255, 0, 0]
                axes[row, col * 2 + 1].imshow(overlay)
            else:
                axes[row, col * 2 + 1].imshow(np.zeros_like(img))
            axes[row, col * 2 + 1].axis("off")

    # Title each pair of columns, not just the first
    for col in range(n_per_class):
        axes[0, col * 2].set_title("Image", fontsize=10)
        axes[0, col * 2 + 1].set_title("Mask overlay", fontsize=10)
    plt.suptitle("Sample Images per Class (red = mask ROI)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "sample_images.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ─── 3. Tabular feature distributions ─────────────────────────────────────────

def plot_tabular_distributions():
    df = load_tabular_data()

    numeric_cols = [
        "Age at Diagnosis", "Tumor Size", "Lymph nodes examined positive",
        "Nottingham prognostic index", "Overall Survival (Months)",
        "Mutation Count", "Tumor Stage",
    ]
    n = len(numeric_cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for cls in CLASSES:
            subset = df[df["class"] == cls][col].dropna().astype(float)
            axes[i].hist(subset, alpha=0.6, bins=20, label=cls, edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        if i == 0:
            axes[i].legend(fontsize=8)

    axes[-1].axis("off")
    plt.suptitle("Numeric Feature Distributions by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "feature_distributions.png"), dpi=150)
    plt.close()


def plot_categorical_features():
    df = load_tabular_data()
    cat_cols = [
        "Type of Breast Surgery", "ER Status", "HER2 Status",
        "Inferred Menopausal State", "Relapse Free Status",
        "Pam50 + Claudin-low subtype",
    ]
    n_cols = 3
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    colors = {"benign": "steelblue", "malignant": "tomato", "normal": "seagreen"}

    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df["class"], normalize="index")
        ct.plot(kind="bar", ax=axes[i], color=[colors[c] for c in ct.columns],
                edgecolor="white", legend=(i == 0))
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Categorical Features (row-normalized by category value)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "categorical_distributions.png"), dpi=150)
    plt.close()


# ─── 4. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap():
    df = load_tabular_data()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3)
    plt.title("Feature Correlation Matrix (numeric)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


# ─── 5. Image size statistics ─────────────────────────────────────────────────

def plot_image_size_stats():
    registry = build_image_registry()
    widths, heights = [], []
    for pid, cls in registry.items():
        img_path = os.path.join(IMAGE_DIR, cls, "images", f"{pid}.png")
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[1].hist(heights, bins=30, color="tomato", edgecolor="white")
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "image_size_stats.png"), dpi=150)
    plt.close()
    print(f"  Image sizes — W: {np.mean(widths):.0f}±{np.std(widths):.0f}, "
          f"H: {np.mean(heights):.0f}±{np.std(heights):.0f}")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running EDA...")
    print("\n1. Class distribution")
    plot_class_distribution()

    print("2. Sample images")
    plot_sample_images(n_per_class=3)

    print("3. Tabular feature distributions")
    plot_tabular_distributions()
    plot_categorical_features()

    print("4. Correlation heatmap")
    plot_correlation_heatmap()

    print("5. Image size statistics")
    plot_image_size_stats()

    print(f"\nEDA complete. Plots saved to: {EDA_DIR}")
