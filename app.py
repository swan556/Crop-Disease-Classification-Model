import os
import glob
from pathlib import Path

import streamlit as st  # pyrefly: ignore [missing-import]
from PIL import Image  # pyrefly: ignore [missing-import]

from src.predict import CropDiseasePredictor

# Find default model checkpoint
CHECKPOINT_DIR = Path("checkpoints")
default_ckpt = None
if CHECKPOINT_DIR.exists():
    ckpts = list(CHECKPOINT_DIR.glob("*.pth"))
    if ckpts:
        default_ckpt = str(ckpts[0])


@st.cache_resource
def load_predictor(ckpt_path: str):
    """Load and cache the predictor so it persists across Streamlit reruns."""
    if os.path.exists(ckpt_path):
        predictor = CropDiseasePredictor(ckpt_path)
        predictor.checkpoint_path = ckpt_path
        return predictor
    return None


def diagnose_leaf(image: Image.Image, checkpoint_path: str):
    if image is None:
        return "Please upload a crop leaf image.", {}, ""

    if not os.path.exists(checkpoint_path):
        return (
            f"Checkpoint file '{checkpoint_path}' not found. Please train a model first using `python src/train.py`.",
            {},
            "No model loaded."
        )

    p = load_predictor(checkpoint_path)
    if p is None:
        return "Failed to load model checkpoint.", {}, ""

    res = p.predict(image, top_k=5)

    # Format confidence probabilities
    confidences = {
        item["class"].replace("___", " - ").replace("_", " "): item["confidence"] / 100.0
        for item in res["top_k_predictions"]
    }

    top_name = res["top_prediction"].replace("___", " - ").replace("_", " ")
    top_conf = res["top_confidence"]

    # OOD / Low Confidence Detection
    if res.get("is_low_confidence", False):
        status_markdown = f"""
### ⚠️ Low Confidence Detection

**The model is uncertain about this diagnosis.** Please ensure:
- The leaf fills the frame clearly and is in focus
- The image is well-lit without heavy shadows
- The leaf is from a supported crop species

---

### 🩺 Best Guess: **{top_name}**
**Confidence Score**: `{top_conf:.1f}%` *(below reliability threshold)*
**Prediction Uncertainty**: `{res.get('normalized_entropy', 0):.2f}` *(0 = certain, 1 = random)*

#### 🚜 Tentative Guidance:
> {res['treatment_advice']}
        """
    else:
        status_markdown = f"""
### 🩺 Diagnosis: **{top_name}**
**Confidence Score**: `{top_conf:.1f}%`

#### 🚜 Actionable Farmer Guidance:
> {res['treatment_advice']}
        """

    return top_name, confidences, status_markdown


def main():
    st.set_page_config(
        page_title="Crop Leaf Disease Diagnosis",
        page_icon="🌾",
        layout="wide",
    )

    st.title("🌾 Crop Leaf Disease Diagnosis System")
    st.markdown(
        "### Powered by PyTorch & Computer Vision for Small-Scale Farmers\n"
        "Upload a clear close-up photograph of a plant leaf to detect diseases early "
        "and receive actionable agricultural treatment guidance."
    )

    # Determine initial checkpoint
    ckpt_files = glob.glob("checkpoints/*.pth")
    initial_ckpt = ckpt_files[0] if ckpt_files else "checkpoints/best_model_resnet18.pth"

    col_input, col_output = st.columns(2)

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Leaf Photo",
            type=["jpg", "jpeg", "png", "webp"],
        )
        ckpt_input = st.text_input(
            "Model Checkpoint Path (.pth)",
            value=initial_ckpt,
            placeholder="checkpoints/best_model_resnet18.pth",
        )
        run_btn = st.button("🔍 Run AI Diagnosis", type="primary", use_container_width=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
        else:
            image = None

    with col_output:
        if run_btn:
            if image is None:
                st.warning("Please upload a crop leaf image first.")
            else:
                with st.spinner("Running diagnosis..."):
                    top_name, confidences, advice_md = diagnose_leaf(image, ckpt_input)

                st.subheader("Top Diagnosis")
                st.info(top_name)

                if confidences:
                    st.subheader("Top Predicted Disease Probabilities")
                    # Display as a horizontal bar chart
                    import pandas as pd  # pyrefly: ignore [missing-import]
                    df = pd.DataFrame(
                        {"Disease": list(confidences.keys()),
                         "Confidence": list(confidences.values())}
                    )
                    df = df.sort_values("Confidence", ascending=True)
                    st.bar_chart(df, x="Disease", y="Confidence", horizontal=True)

                st.subheader("Treatment Advice")
                st.markdown(advice_md)

    st.divider()
    st.caption(
        "Built with PyTorch, Torchvision & Streamlit. "
        "Designed for deployment in agricultural extension programs."
    )


if __name__ == "__main__":
    main()
