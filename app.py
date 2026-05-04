"""
Streamlit demo for the Breast Cancer Ultrasound Classifier.

Model: Multimodal EfficientNet-B3 + MLP (fold 3 checkpoint, 98.3% test accuracy)
Tabular branch: population-average clinical features (demo mode)
"""

import io
import os

import numpy as np
import torch
from PIL import Image
import streamlit as st

from src.config import CLASSES
from src.preprocessing.image_preprocessing import get_val_transforms, apply_mask
from src.models.multimodal_model import build_model
from src.training.trainer import get_device

# ─── Paths ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join("checkpoints", "best_fold0_multimodal.pt")
MEAN_TAB_PATH   = os.path.join("demo_assets", "mean_tabular.npy")

DEMO_IMAGES = {
    "Benign (example)"    : os.path.join("demo_images", "benign.png"),
    "Malignant (example)" : os.path.join("demo_images", "malignant.png"),
    "Normal (example)"    : os.path.join("demo_images", "normal.png"),
}

CLASS_COLORS = {
    "benign"    : "#2196F3",   # blue
    "malignant" : "#F44336",   # red
    "normal"    : "#4CAF50",   # green
}

# ─── Cached model loader ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model_cached():
    mean_tab = np.load(MEAN_TAB_PATH).astype(np.float32)
    tabular_input_dim = len(mean_tab)
    model = build_model("multimodal", tabular_input_dim=tabular_input_dim, pretrained=False)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, mean_tab


# ─── Inference ────────────────────────────────────────────────────────────────

def predict(pil_image: Image.Image, model, mean_tab: np.ndarray) -> tuple[str, np.ndarray]:
    img_rgb = np.array(pil_image.convert("RGB"))
    transform = get_val_transforms()
    img_tensor = transform(img_rgb).unsqueeze(0)          # (1, 3, 224, 224)
    tab_tensor = torch.tensor(mean_tab).unsqueeze(0)      # (1, F)

    with torch.no_grad():
        logits = model(img_tensor, tab_tensor)

    proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(proba.argmax())
    return CLASSES[pred_idx], proba


# ─── UI ───────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            """
**1. Select an example** from the dropdown, or
**2. Upload your own** breast ultrasound image.

---

**Accepted image types**
- PNG or JPG
- Any resolution (auto-resized to 224 × 224)
- Grayscale or RGB ultrasound scans

**What is a breast ultrasound image?**
High-frequency sound waves create cross-sectional images of breast tissue. The model was trained on the [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) (780 patients).

---

**Model details**
- Architecture: EfficientNet-B3 + MLP late-fusion
- Checkpoint: fold 0 (97.7 % val accuracy)
- Test accuracy: **98.3 %**, AUC: **1.000**
- Tabular branch: population-average clinical values

---

> ⚠️ **Research demo only.**
> This tool is not approved for clinical use.
> Always consult a licensed physician.
            """
        )


def render_results(cls_name: str, proba: np.ndarray):
    color = CLASS_COLORS[cls_name]
    st.markdown(
        f"<h2 style='color:{color}; margin-top:0'>Prediction: {cls_name.upper()}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("**Confidence scores**")
    for c, p in zip(CLASSES, proba):
        bar_color = CLASS_COLORS[c]
        label = f"**{c.capitalize()}**"
        pct = f"{p * 100:.1f} %"
        st.markdown(
            f"{label} &nbsp; `{pct}`",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:{bar_color}; width:{p * 100:.1f}%; "
            f"height:18px; border-radius:4px; margin-bottom:8px;'></div>",
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Breast Cancer Ultrasound Classifier",
        page_icon="🩺",
        layout="wide",
    )

    render_sidebar()

    st.title("🩺 Breast Cancer Ultrasound Classifier")
    st.markdown(
        "Upload a breast ultrasound image — or pick a built-in example — "
        "to classify it as **benign**, **malignant**, or **normal**."
    )
    st.divider()

    # ── Image selection ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Input Image")

        source = st.radio(
            "Image source",
            ["Use an example image", "Upload my own image"],
            horizontal=True,
            label_visibility="collapsed",
        )

        pil_image = None

        if source == "Use an example image":
            choice = st.selectbox("Select example", list(DEMO_IMAGES.keys()))
            pil_image = Image.open(DEMO_IMAGES[choice])
            st.caption(f"Showing: {choice}")
        else:
            uploaded = st.file_uploader(
                "Upload a PNG or JPG breast ultrasound image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                pil_image = Image.open(uploaded)
            else:
                st.info(
                    "No image uploaded yet. "
                    "Select **'Use an example image'** above or upload a file."
                )
                # Show the benign default as a preview
                st.markdown("**Default preview (benign example):**")
                pil_image = Image.open(DEMO_IMAGES["Benign (example)"])

        if pil_image is not None:
            st.image(pil_image, use_container_width=True, clamp=True)

    # ── Prediction ───────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Classification Result")

        if pil_image is not None:
            try:
                model, mean_tab = load_model_cached()
                with st.spinner("Running inference…"):
                    cls_name, proba = predict(pil_image, model, mean_tab)
                render_results(cls_name, proba)
            except Exception as e:
                st.error(f"Inference failed: {e}")
        else:
            st.info("Awaiting image input.")

    st.divider()
    st.caption(
        "Model trained on the BUSI + METABRIC datasets. "
        "Tabular branch uses population-average clinical features in demo mode. "
        "For research and educational use only."
    )


if __name__ == "__main__":
    main()
