"""
PyTorch Dataset classes and DataLoader factories.

Three Dataset types:
  - MultimodalDataset  : image tensor + tabular features + label
  - ImageOnlyDataset   : image tensor + label
  - TabularOnlyDataset : tabular features + label

Splitting strategy:
  - A fixed 15% stratified test set is held out before any cross-validation.
  - The remaining 85% is split into N_FOLDS for cross-validation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    CLASS_TO_IDX, CLASSES, SEED, BATCH_SIZE, N_FOLDS, TEST_SPLIT,
)
from src.preprocessing.image_preprocessing import (
    load_and_preprocess, build_image_registry,
    get_train_transforms, get_val_transforms,
)
from src.preprocessing.tabular_preprocessing import (
    load_tabular_data, TabularPreprocessor,
)


# ─── Dataset classes ──────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    """Returns (image_tensor, tabular_tensor, label) for each patient."""

    def __init__(
        self,
        patient_ids: list,
        labels: np.ndarray,
        tabular_X: np.ndarray,
        image_registry: dict,   # {patient_id: class_name}
        transform,
        use_mask: bool = True,
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.tabular_X = tabular_X
        self.image_registry = image_registry
        self.transform = transform
        self.use_mask = use_mask

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = int(self.labels[idx])
        cls_name = self.image_registry[pid]

        # Image
        image = load_and_preprocess(pid, cls_name, self.transform, self.use_mask)

        # Tabular
        tab = torch.tensor(self.tabular_X[idx], dtype=torch.float32)

        return image, tab, label


class ImageOnlyDataset(Dataset):
    """Returns (image_tensor, label)."""

    def __init__(
        self,
        patient_ids: list,
        labels: np.ndarray,
        image_registry: dict,
        transform,
        use_mask: bool = True,
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.image_registry = image_registry
        self.transform = transform
        self.use_mask = use_mask

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = int(self.labels[idx])
        cls_name = self.image_registry[pid]
        image = load_and_preprocess(pid, cls_name, self.transform, self.use_mask)
        return image, label


class TabularOnlyDataset(Dataset):
    """Returns (tabular_tensor, label)."""

    def __init__(self, tabular_X: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(tabular_X, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(mode: str = "multimodal") -> dict:
    """
    Load + preprocess all data. Returns a dict with everything needed for
    cross-validation: splits, preprocessor, image_registry, labels, etc.

    mode: "multimodal" | "image" | "tabular"
    """
    # 1. Load tabular data
    df = load_tabular_data()

    # 2. Extract labels and patient IDs directly — no preprocessor needed yet
    patient_ids_all = df["Patient ID"].tolist()
    y_all = np.array([CLASS_TO_IDX[c] for c in df["class"]])

    # 3. Build image registry
    image_registry = build_image_registry()

    # 4. Stratified train/test split
    indices = np.arange(len(patient_ids_all))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, stratify=y_all, random_state=SEED
    )

    test_ids = [patient_ids_all[i] for i in test_idx]
    test_labels = y_all[test_idx]
    # Store raw test DataFrame — will be transformed by the fold preprocessor
    # in get_test_loader() to prevent data leakage from test statistics.
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_val_ids = [patient_ids_all[i] for i in train_val_idx]
    train_val_labels = y_all[train_val_idx]
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)

    # Fit reference preprocessor on train_val only — used solely to determine
    # input_dim for model construction. Test set is NOT included here.
    preprocessor_full = TabularPreprocessor()
    preprocessor_full.fit_transform(train_val_df)

    return {
        "train_val_ids": train_val_ids,
        "train_val_labels": train_val_labels,
        "train_val_df": train_val_df,
        "test_ids": test_ids,
        "test_labels": test_labels,
        "test_df": test_df,
        "image_registry": image_registry,
        "full_preprocessor": preprocessor_full,
        "mode": mode,
    }


def get_fold_loaders(
    data: dict,
    fold_idx: int,
    mode: str = "multimodal",
    use_mask: bool = True,
) -> tuple[DataLoader, DataLoader, TabularPreprocessor]:
    """
    For a given fold, refit tabular preprocessor on train split only,
    return (train_loader, val_loader, preprocessor).
    """
    train_val_ids = data["train_val_ids"]
    train_val_labels = data["train_val_labels"]
    train_val_df = data["train_val_df"]
    image_registry = data["image_registry"]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train_val_ids, train_val_labels))
    train_idx, val_idx = splits[fold_idx]

    # Split DataFrames for refitting preprocessor
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    preprocessor = TabularPreprocessor()
    X_train_tab, y_train, train_ids = preprocessor.fit_transform(train_df)
    X_val_tab, y_val, val_ids = preprocessor.transform(val_df)

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    if mode == "multimodal":
        train_ds = MultimodalDataset(train_ids, y_train, X_train_tab, image_registry, train_transform, use_mask)
        val_ds = MultimodalDataset(val_ids, y_val, X_val_tab, image_registry, val_transform, use_mask)
    elif mode == "image":
        train_ds = ImageOnlyDataset(train_ids, y_train, image_registry, train_transform, use_mask)
        val_ds = ImageOnlyDataset(val_ids, y_val, image_registry, val_transform, use_mask)
    elif mode == "tabular":
        train_ds = TabularOnlyDataset(X_train_tab, y_train)
        val_ds = TabularOnlyDataset(X_val_tab, y_val)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, preprocessor


def get_test_loader(data: dict, preprocessor: TabularPreprocessor, mode: str = "multimodal", use_mask: bool = True) -> DataLoader:
    """Return DataLoader for the held-out test set.

    Args:
        preprocessor: The fold-specific preprocessor fitted on training data only.
                      This prevents test statistics from leaking into preprocessing.
    """
    test_ids = data["test_ids"]
    test_labels = data["test_labels"]
    image_registry = data["image_registry"]

    # Transform test set using the fold's preprocessor (fitted on training data only).
    # This ensures no test-set statistics influence the scaler/encoders.
    X_test_tab, _, _ = preprocessor.transform(data["test_df"])

    val_transform = get_val_transforms()

    if mode == "multimodal":
        test_ds = MultimodalDataset(test_ids, test_labels, X_test_tab, image_registry, val_transform, use_mask)
    elif mode == "image":
        test_ds = ImageOnlyDataset(test_ids, test_labels, image_registry, val_transform, use_mask)
    elif mode == "tabular":
        test_ds = TabularOnlyDataset(X_test_tab, test_labels)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
