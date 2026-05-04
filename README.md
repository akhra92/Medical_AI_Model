# Medical Image AI

A multimodal breast cancer classification system that fuses ultrasound imaging with clinical and molecular tabular data to predict whether a case is **benign**, **malignant**, or **normal**.

---

## Overview

| | |
|---|---|
| **Task** | 3-class classification: benign / malignant / normal |
| **Modalities** | Ultrasound images + clinical history + molecular biomarkers |
| **Architecture** | Late-fusion: EfficientNet-B3 + MLP → concatenation → classifier |
| **Training** | 5-fold stratified cross-validation with held-out test set (15%) |
| **Dataset** | 780 patients — 437 benign, 210 malignant, 133 normal |

---

## Datasets

This project combines three datasets covering 780 breast cancer patients.

### Dataset 1 — Breast Ultrasound Images (BUSI)
Ultrasound images with pixel-level segmentation masks collected in 2018 from female patients aged 25–75. Images are PNG format, average size ~500×500px.

- 437 benign, 210 malignant, 133 normal cases
- Each case includes an image and a binary segmentation mask (ROI)
- **Source:** [Breast Ultrasound Images Dataset — Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Paper:** [Dataset of Breast Ultrasound Images — PubMed](https://pubmed.ncbi.nlm.nih.gov/31867417/)

### Dataset 2 & 3 — METABRIC Clinical & Molecular Data
The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) database provides clinical history and molecular biomarker profiles for the same 780 patients.

- **Dataset 2** (25 features): age, tumor size, tumor stage, ER/HER2 status, survival months, Nottingham prognostic index, hormone therapy, and more.
- **Dataset 3** (10 features): Pam50 subtype, integrative cluster, cellularity, chemotherapy, PR status, and more.
- **Source:** [Breast Cancer Gene Expression Profiles (METABRIC) — Kaggle](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)

> **Note:** The two tabular datasets are merged on `Patient ID` before preprocessing.

---

## Repository Structure

```
Medical_Image_AI/
├── train.py                  # Main training script (cross-validation)
├── evaluate.py               # Evaluation on held-out test set
├── predict.py                # Single-image and batch inference
├── train_baselines.py        # Classical ML baselines (RF, XGBoost, LR)
├── eda.py                    # Exploratory data analysis & visualizations
│
├── src/
│   ├── config.py             # All hyperparameters and paths
│   ├── models/
│   │   ├── multimodal_model.py   # Late-fusion multimodal classifier
│   │   ├── image_model.py        # EfficientNet-B3 image encoder
│   │   └── tabular_model.py      # MLP for tabular features
│   ├── preprocessing/
│   │   ├── data_loader.py        # Dataset classes & data splitting
│   │   ├── image_preprocessing.py    # Image loading, masking, augmentation
│   │   └── tabular_preprocessing.py  # Feature encoding & scaling
│   ├── training/
│   │   └── trainer.py        # Training loop, early stopping, two-stage training
│   └── evaluation/
│       └── metrics.py        # Metrics computation and plotting utilities
│
├── dataset/
│   ├── dataset1/             # Ultrasound images + segmentation masks
│   │   ├── benign/images/ & masks/
│   │   ├── malignant/images/ & masks/
│   │   └── normal/images/ & masks/
│   ├── dataset2/
│   │   └── patient_history_dataset.csv   # 25 clinical features
│   └── dataset3/
│       └── molecular_biomarker_dataset.csv  # 10 molecular features
│
├── checkpoints/              # Saved model weights: best_fold{0-4}_{mode}.pt
└── results/                  # Plots, metrics CSVs, EDA outputs
```

---

## Model Architecture

```
Ultrasound Image (224×224×3)      Tabular Features (N-dim)
          │                                  │
  EfficientNet-B3                      MLP [256, 128, 64]
  GAP → Dropout                              │
  Linear → 256-dim                     64-dim embedding
          │                                  │
          └──────────── Concat (320-dim) ────┘
                               │
                     Fusion Head: Linear(320→128)
                     BatchNorm → ReLU → Dropout
                     Linear(128→3 classes)
                               │
                           Logits (B, 3)
```

**Training strategy:**
1. **Warmup (5 epochs):** Backbone frozen, head trained at 5× higher learning rate.
2. **Fine-tune (50 epochs):** All weights unfrozen, end-to-end training with cosine LR decay.

---

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `torchvision`, `efficientnet_pytorch`, `scikit-learn`, `xgboost`, `pandas`, `Pillow`, `matplotlib`, `seaborn`, `tqdm`

---

## Dataset Setup

Place datasets in the following structure:

```
dataset/
├── dataset1/
│   ├── benign/
│   │   ├── images/    # MB-XXXX.png  (437 files)
│   │   └── masks/     # MB-XXXX.png  (binary segmentation masks)
│   ├── malignant/
│   │   ├── images/    # (210 files)
│   │   └── masks/
│   └── normal/
│       ├── images/    # (133 files)
│       └── masks/
├── dataset2/patient_history_dataset.csv   # Columns: Patient ID, Age, Tumor Size, ...
└── dataset3/molecular_biomarker_dataset.csv  # Columns: Patient ID, Cancer Type, ...
```

Image files are PNG, named by Patient ID (e.g., `MB-0002.png`). Mask files share the same names as images.

---

## Usage

### Exploratory Data Analysis

```bash
python eda.py
```

Generates plots in `results/eda/`: class distribution, sample images with mask overlays, feature distributions, correlation heatmap.

### Training

```bash
# Multimodal (recommended)
python train.py --mode multimodal

# Image only
python train.py --mode image

# Tabular only
python train.py --mode tabular

# Options
python train.py --mode multimodal --no-mask       # Disable segmentation masks
python train.py --mode multimodal --no-pretrained  # Train from scratch
python train.py --mode multimodal --folds 5        # Number of CV folds
```

**Outputs saved to `results/` and `checkpoints/`:**
- `checkpoints/best_fold{0-4}_{mode}.pt` — Best checkpoint per fold
- `results/fold_summary_{mode}.csv` — Cross-validation metrics summary
- `results/test_metrics_{mode}.csv` — Final test set metrics
- `results/cm_fold{1-5}_{mode}.png` — Per-fold confusion matrices
- `results/history_fold{1-5}_{mode}.png` — Per-fold training/validation loss curves
- `results/cm_test_{mode}.png` — Test set confusion matrix
- `results/roc_test_{mode}.png` — Test set ROC curves

> The final test set is evaluated using the checkpoint and preprocessor from the best-performing fold (by validation macro F1), ensuring consistent feature scaling between training and test.

### Evaluation

```bash
# Evaluate fold 0 checkpoint on held-out test set
python evaluate.py --mode multimodal --fold 0
```

**Outputs:**
- `results/eval_cm_{mode}_fold{fold}.png` — Confusion matrix
- `results/eval_roc_{mode}_fold{fold}.png` — ROC curves (one-vs-rest)
- `results/eval_metrics_{mode}_fold{fold}.csv` — Per-class and aggregate metrics

### Inference

```bash
# Single image (image-only mode)
python predict.py --image path/to/image.png --mode image --fold 0

# Single image with optional mask
python predict.py --image path/to/image.png --mask path/to/mask.png --mode image --fold 0

# Multimodal (image + patient data from CSV)
python predict.py --image path/to/image.png --patient-id MB-1234 --mode multimodal --fold 0

# Batch prediction from a folder of PNG images
python predict.py --image-dir path/to/folder/ --mode image --fold 0
```

**Example single-image output:**
```
Prediction: MALIGNANT
Class probabilities:
  benign      : 0.0234  ██
  malignant   : 0.9412  ██████████████████████████████
  normal      : 0.0354  ███
```

### Baseline Comparison

```bash
python train_baselines.py
```

Trains Logistic Regression, Random Forest (300 trees), and XGBoost (300 trees, `mlogloss`) on tabular features only. Results saved to `results/baseline_results.csv`.

---

## Results

### Exploratory Data Analysis

*Class Distributions.*

![Class Distribution Visualization](results/eda/class_distribution.png)

*Sample Images.*

![Sample Images](results/eda/sample_images.png)

---

### Performance Summary

**5-Fold Cross-Validation (multimodal mode):**

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 98.6% | ±1.2% |
| Macro F1 | 98.7% | ±1.2% |
| Weighted F1 | 98.6% | ±1.2% |
| ROC-AUC (macro) | 99.9% | ±0.1% |

**Held-Out Test Set (15%, 117 patients):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Benign | 97.1% | 100% | 98.5% |
| Malignant | 100% | 93.5% | 96.7% |
| Normal | 100% | 100% | 100% |
| **Overall** | **Accuracy: 98.3%** | **Macro F1: 98.4%** | **AUC: 1.000** |

---

### Learning Curves

*Representative training curve (Fold 3) showing clean convergence with no overfitting.*

![Learning Curves Fold 3](results/history_fold3_multimodal.png)

---

### Confusion Matrix

*Test set confusion matrix (count and normalized). Only 2 malignant cases misclassified as benign.*

![Confusion Matrix](results/cm_test_multimodal.png)

---

### ROC Curves

*Per-class ROC curves (one-vs-rest) on the test set. All three classes achieve AUC = 1.000.*

![ROC Curves](results/roc_test_multimodal.png)

---

## Configuration

All hyperparameters are centralized in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 224 | Input image resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 50 | Max epochs per fold |
| `LEARNING_RATE` | 1e-4 | Base learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `PATIENCE` | 10 | Early stopping patience |
| `N_FOLDS` | 5 | Cross-validation folds |
| `TEST_SPLIT` | 0.15 | Held-out test fraction |
| `DROPOUT` | 0.3 | Dropout rate |
| `IMAGE_BACKBONE` | `efficientnet_b3` | CNN backbone |
| `IMAGE_FEATURE_DIM` | 256 | Image embedding size |
| `TABULAR_HIDDEN_DIMS` | [256, 128, 64] | MLP hidden layers |
| `CLASS_WEIGHTS` | [0.596, 1.238, 1.754] | Inverse-frequency class weights |

---

## Metrics

Each fold and the final test set are evaluated on:

- **Accuracy**
- **Macro F1** (unweighted average across classes)
- **Weighted F1** (weighted by support)
- **ROC-AUC** (one-vs-rest, macro-averaged)
- **Per-class precision, recall, F1**

---

## License

This project is for research and educational purposes.
