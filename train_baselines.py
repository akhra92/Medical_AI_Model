"""
Classical ML baselines for tabular data only.
Models: Random Forest, XGBoost, Logistic Regression
Uses stratified 5-fold CV and reports results.

Run: python train_baselines.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, roc_auc_score,
    make_scorer, f1_score,
)
from xgboost import XGBClassifier

from src.config import N_FOLDS, SEED, RESULTS_DIR, CLASS_TO_IDX, CLASSES
from src.preprocessing.tabular_preprocessing import load_tabular_data, TabularPreprocessor

os.makedirs(RESULTS_DIR, exist_ok=True)


def prepare_tabular():
    df = load_tabular_data()
    preprocessor = TabularPreprocessor()
    X, y, _ = preprocessor.fit_transform(df)
    preprocessor.save()   # save for use in predict.py
    return X, y, preprocessor


def evaluate_model(name, clf, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
    }
    cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=False)

    acc_mean = cv_results["test_accuracy"].mean()
    acc_std = cv_results["test_accuracy"].std()
    f1_mean = cv_results["test_macro_f1"].mean()
    f1_std = cv_results["test_macro_f1"].std()
    wf1_mean = cv_results["test_weighted_f1"].mean()

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy     : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Macro F1     : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Weighted F1  : {wf1_mean:.4f}")

    return {
        "model": name,
        "accuracy": acc_mean, "accuracy_std": acc_std,
        "macro_f1": f1_mean, "macro_f1_std": f1_std,
        "weighted_f1": wf1_mean,
    }


def main():
    print("Loading tabular data...")
    X, y, preprocessor = prepare_tabular()
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Class distribution: { {c: int((y==i).sum()) for i, c in enumerate(CLASSES)} }")

    classifiers = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced",
                multi_class="multinomial", solver="lbfgs", random_state=SEED,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1,
            ),
        ),
    ]

    results = []
    for name, clf in classifiers:
        metrics = evaluate_model(name, clf, X, y)
        results.append(metrics)

    df_res = pd.DataFrame(results).set_index("model")
    print(f"\n{'='*50}")
    print("  Baseline Summary")
    print(f"{'='*50}")
    print(df_res[["accuracy", "macro_f1", "weighted_f1"]].to_string())

    out_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    df_res.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
