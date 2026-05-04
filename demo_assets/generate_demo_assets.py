"""
One-time script to generate demo assets for the Streamlit app.

Outputs:
  results/tabular_preprocessor.pkl   — fitted TabularPreprocessor (needed at inference)
  demo_assets/mean_tabular.npy       — population-mean tabular feature vector (float32)

Run from project root:
  python demo_assets/generate_demo_assets.py
"""

import os
import sys
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.tabular_preprocessing import load_tabular_data, TabularPreprocessor, PREPROCESSOR_PATH

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Loading tabular data...")
    df = load_tabular_data()
    print(f"  Loaded {len(df)} patients.")

    print("Fitting TabularPreprocessor on full dataset...")
    preprocessor = TabularPreprocessor()
    X_all, y_all, _ = preprocessor.fit_transform(df)
    print(f"  Feature matrix shape: {X_all.shape}")

    print(f"Saving preprocessor → {PREPROCESSOR_PATH}")
    preprocessor.save(PREPROCESSOR_PATH)

    mean_vec = X_all.mean(axis=0).astype(np.float32)
    out_path = os.path.join(OUT_DIR, "mean_tabular.npy")
    np.save(out_path, mean_vec)
    print(f"Saving mean tabular vector ({len(mean_vec)} features) → {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
