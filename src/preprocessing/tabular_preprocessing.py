"""
Tabular preprocessing utilities.

Steps:
  1. Load patient_history_dataset.csv (dataset2) + molecular_biomarker_dataset.csv (dataset3)
  2. Merge on Patient ID
  3. Drop irrelevant / constant columns
  4. Encode categorical features (LabelEncoder / OrdinalEncoder)
  5. Scale numeric features (StandardScaler)
  6. Return processed feature matrix + patient IDs + labels

The fitted encoder/scaler are returned so they can be reused at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import os

from src.config import (
    PATIENT_CSV, BIOMARKER_CSV,
    TABULAR_DROP_COLS, NUMERIC_COLS,
    CLASS_TO_IDX, CLASSES,
    RESULTS_DIR,
)

PREPROCESSOR_PATH = os.path.join(RESULTS_DIR, "tabular_preprocessor.pkl")


def load_tabular_data() -> pd.DataFrame:
    """
    Load and merge the two CSV files on Patient ID.
    Returns a single DataFrame with all features + 'class' label.
    """
    df_hist = pd.read_csv(PATIENT_CSV)
    df_bio = pd.read_csv(BIOMARKER_CSV)

    # Merge on Patient ID (inner join — all 780 IDs are present in both)
    df = pd.merge(df_hist, df_bio, on="Patient ID", how="inner")
    return df


def _identify_categorical_cols(df: pd.DataFrame, numeric_cols: list, drop_cols: list) -> list:
    """Return column names that are non-numeric and not dropped."""
    cat_cols = []
    for col in df.columns:
        if col in drop_cols:
            continue
        if col not in numeric_cols and not pd.api.types.is_numeric_dtype(df[col]):
            cat_cols.append(col)
    return cat_cols


class TabularPreprocessor:
    """
    Stateful preprocessor: fit on training data, transform on val/test.
    """

    def __init__(self):
        self.label_encoders: dict = {}   # col → LabelEncoder
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.numeric_cols_: list = []
        self.cat_cols_: list = []
        self.feature_cols_: list = []    # final ordered list of feature columns
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Fit on df, return (X, y, patient_ids).
        X: float32 array (N, F)
        y: int array (N,) — class index
        patient_ids: list of str
        """
        df = df.copy()
        patient_ids = df["Patient ID"].tolist()
        y = np.array([CLASS_TO_IDX[c] for c in df["class"]])

        # Drop non-feature columns
        df.drop(columns=[c for c in TABULAR_DROP_COLS if c in df.columns], inplace=True)

        # Identify column types
        self.numeric_cols_ = [c for c in NUMERIC_COLS if c in df.columns]
        self.cat_cols_ = _identify_categorical_cols(df, self.numeric_cols_, [])

        # Impute (fills any NaN)
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns,
        )

        # Encode categoricals
        for col in self.cat_cols_:
            le = LabelEncoder()
            df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
            self.label_encoders[col] = le

        # Ensure numerics are float
        for col in self.numeric_cols_:
            df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce").fillna(0)

        self.feature_cols_ = list(df_imputed.columns)
        X = df_imputed[self.feature_cols_].values.astype(np.float32)

        # Scale all features (numerics get proper scaling; encoded cats are also scaled)
        X = self.scaler.fit_transform(X).astype(np.float32)
        self.fitted = True
        return X, y, patient_ids

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Apply fitted preprocessor to new data. Returns (X, y, patient_ids).
        """
        assert self.fitted, "Call fit_transform first."
        df = df.copy()
        patient_ids = df["Patient ID"].tolist()
        y = np.array([CLASS_TO_IDX[c] for c in df["class"]])

        df.drop(columns=[c for c in TABULAR_DROP_COLS if c in df.columns], inplace=True)

        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns,
        )

        for col in self.cat_cols_:
            le = self.label_encoders[col]
            df_imputed[col] = df_imputed[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )

        for col in self.numeric_cols_:
            df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce").fillna(0)

        X = df_imputed[self.feature_cols_].values.astype(np.float32)
        X = self.scaler.transform(X).astype(np.float32)
        return X, y, patient_ids

    def save(self, path: str = PREPROCESSOR_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = PREPROCESSOR_PATH) -> "TabularPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols_)


def get_feature_importance_names(preprocessor: TabularPreprocessor) -> list:
    """Return human-readable feature names after encoding."""
    return preprocessor.feature_cols_
