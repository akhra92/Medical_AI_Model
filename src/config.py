"""
Central configuration for the Medical Image AI project.
All hyperparameters, paths, and settings are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

IMAGE_DIR = os.path.join(DATASET_DIR, "dataset1")          # ultrasound images + masks
PATIENT_CSV = os.path.join(DATASET_DIR, "dataset2", "patient_history_dataset.csv")
BIOMARKER_CSV = os.path.join(DATASET_DIR, "dataset3", "molecular_biomarker_dataset.csv")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASSES = ["benign", "malignant", "normal"]          # alphabetical → index 0,1,2
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = 3

# Class weights (inverse frequency) to handle imbalance
# benign=437, malignant=210, normal=133  → total=780
CLASS_WEIGHTS = [780 / (3 * 437), 780 / (3 * 210), 780 / (3 * 133)]

# ─── Image settings ────────────────────────────────────────────────────────────
IMAGE_SIZE = 224                # resize all images to IMAGE_SIZE × IMAGE_SIZE
USE_MASKS = True                # multiply image by segmentation mask (focus on ROI)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Tabular settings ──────────────────────────────────────────────────────────
# Columns to drop from merged tabular dataframe (non-informative or leakage-prone)
TABULAR_DROP_COLS = [
    "Patient ID",
    "Cancer Type",          # constant "Breast Cancer"
    "Sex",                  # constant "Female"
    "class",                # this is the label
    # survival & relapse columns could be strong leakage in a real setting;
    # kept here for academic completeness but flagged
]

# Numeric columns to scale
NUMERIC_COLS = [
    "Age at Diagnosis",
    "Cohort",
    "Lymph nodes examined positive",
    "Tumor Size",
    "Tumor Stage",
    "Mutation Count",
    "Nottingham prognostic index",
    "Overall Survival (Months)",
    "Relapse Free Status (Months)",
    "Neoplasm Histologic Grade",
]

# ─── Training ─────────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10               # early-stopping patience (epochs)

# Cross-validation
N_FOLDS = 5
VAL_SPLIT = 0.15            # fraction held out for validation within each fold
TEST_SPLIT = 0.15           # held-out test set (stratified, fixed across all runs)

# ─── Model ────────────────────────────────────────────────────────────────────
IMAGE_BACKBONE = "efficientnet_b3"    # torchvision model name
IMAGE_FEATURE_DIM = 256               # size of image embedding head
TABULAR_HIDDEN_DIMS = [256, 128, 64]  # MLP hidden layer sizes
TABULAR_FEATURE_DIM = 64             # output embedding from tabular MLP
FUSION_HIDDEN_DIM = 128              # fusion classifier hidden size
DROPOUT = 0.3

# ─── Augmentation (training only) ────────────────────────────────────────────
AUG_H_FLIP = True
AUG_V_FLIP = True
AUG_ROTATION = 15            # degrees
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
