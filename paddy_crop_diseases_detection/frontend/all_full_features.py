
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import datetime
import os
import time
import matplotlib.pyplot as plt

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(
    page_title="Paddy Crop Disease Detection Dashboard",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# AGRICULTURE THEME CSS + FOOTER
# -----------------------------
agri_css = """
<style>
body { background-color: #f4f8f2; }

/* Title */
.main-title {
    font-size: 50px;
    font-weight: 900;
    color: #2d6a4f;
    text-align: center;
    margin-top:0;
    padding-top:10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #1b4332;
    margin-top: -10px;
    margin-bottom: 18px;
}

/* Prediction Cards */
.card {
    padding: 16px;
    border-radius: 12px;
    background: #d8f3dc;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
}

.result-card {
    padding: 18px;
    border-radius: 12px;
    background: #b7e4c7;
    border-left: 8px solid #2e8b57;
}

/* Danger / low confidence */
.danger-card {
    padding: 18px;
    border-radius: 12px;
    background: #ffd6d6;
    border-left: 8px solid #b30000;
}

.sidebar .sidebar-content {
    background-color: #eaf5e3 !important;
    font-size: 25px;
}

/* Footer */
.footer {
    margin-top: 30px;
    border-top: 1px solid #e0e0e0;
    padding-top: 12px;
    color: #6b7280;
    font-size: 13px;
    text-align: center;
}
.logo {
    height: 34px;
    margin-right: 10px;
    vertical-align: middle;
}
</style>
"""

st.markdown(agri_css, unsafe_allow_html=True)

# -----------------------------
# SETTINGS
# -----------------------------
CONF_THRESHOLD = 55.0

# -----------------------------
# LOAD MODELS (cached)
# -----------------------------
@st.cache_resource
def load_models():
    vgg = None
    densenet = None
    try:
        vgg = tf.keras.models.load_model("../model/rice_model_vgg19.keras")
    except Exception as e:
        st.warning("VGG19 model could not be loaded: " + str(e))
    try:
        densenet = tf.keras.models.load_model("../model/rice_model_densenet.keras")
    except Exception as e:
        st.warning("DenseNet model could not be loaded: " + str(e))
    return vgg, densenet

vgg19_model, densenet_model = load_models()

# -----------------------------
# CLASS LABELS & TREATMENTS
# -----------------------------
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

DISPLAY_NAMES = {
    "bacterial_leaf_blight": "Bacterial Leaf Blight",
    "bacterial_leaf_streak": "Bacterial Leaf Streak",
    "bacterial_panicle_blight": "Bacterial Panicle Blight",
    "blast": "Rice Blast",
    "brown_spot": "Brown Spot",
    "dead_heart": "Dead Heart (Stem Borer)",
    "downy_mildew": "Downy Mildew",
    "hispa": "Rice Hispa",
    "normal": "Healthy Leaf",
    "tungro": "Rice Tungro Virus"
}

TREATMENT_GUIDE = {
    "bacterial_leaf_blight": "Apply Streptocycline + Copper Oxychloride; avoid standing water.",
    "bacterial_leaf_streak": "Use validamycin or propiconazole; maintain field hygiene.",
    "bacterial_panicle_blight": "Use bactericides; improve soil drainage.",
    "blast": "Use Tricyclazole or Isoprothiolane; avoid excess nitrogen.",
    "brown_spot": "Apply Mancozeb; increase potash in soil.",
    "dead_heart": "Apply Carbofuran granules; destroy affected tillers.",
    "downy_mildew": "Spray Metalaxyl or Mancozeb; avoid moisture accumulation.",
    "hispa": "Use Chlorpyrifos or Neem oil; avoid over-fertilizing.",
    "normal": "Leaf is healthy. No treatment required.",
    "tungro": "Control leafhoppers; remove infected plants immediately."
}

# -----------------------------
# IMAGE PREDICTION HELPERS
# -----------------------------
def preprocess_for_model(img: Image.Image, size=(224,224)):
    img = img.convert("RGB").resize(size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def model_predict_safe(model, img):
    """Return numpy probabilities or None if model missing."""
    if model is None:
        return None
    x = preprocess_for_model(img)
    preds = model.predict(x)[0]
    return preds


def is_rice_or_leaf(img: Image.Image):
    """
    Lightweight heuristic to check if image *looks* like a leaf (green-dominant).
    This is a heuristic and not perfect — it's intended to reduce obvious non-leaf uploads.
    """
    try:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb.resize((224,224))) / 255.0
        # Ensure shape (H,W,3)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False
        green_ratio = np.mean(arr[:, :, 1])
        # Also check overall saturation-ish: green channel significantly larger than others
        rg_diff = np.mean(arr[:, :, 1] - arr[:, :, 0])
        gb_diff = np.mean(arr[:, :, 1] - arr[:, :, 2])
        # Heuristic thresholds (tweakable)
        if green_ratio < 0.25:
            return False
        if rg_diff < 0.02 and gb_diff < 0.02:
            # Not strongly green compared to red/blue
            return False
        return True
    except Exception:
        return False


def predict_image(model, img):
    # Defensive: ensure model exists
    if model is None:
        return None, "Model Not Loaded", 0.0

    # Check whether image looks like a leaf first
    if not is_rice_or_leaf(img):
        return None, "Not a paddy image!", 0.0

    preds = model_predict_safe(model, img)
    if preds is None:
        return None, "Model Not Loaded", 0.0

    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf = float(np.max(preds) * 100.0)
    return preds, label, conf

# -----------------------------
# SESSION STORAGE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# SIDEBAR NAV
# -----------------------------
# st.sidebar.image("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='50' height='50'><rect width='100%' height='100%' fill='%232e8b57' rx='12'/><text x='80%' y='70%' font-size='50' text-anchor='middle' fill='white' font-family='Arial' dy='.3em'>🌾</text></svg>", width=60)
st.sidebar.markdown(
    """<h2 style="font-size: 35px;'font-weight: 500; font-family: 'Arial', sans-serif;text-shadow: 1px 1px 2px #aaa;">🌾 Navigation</h2>  <hr style="border: 1px solid #2e8b57; margin-top: -10px; margin-bottom: 20px;">""", 
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* Radio option font size, color, and weight */
    div[role="radiogroup"] label {
        font-size: 25px;
        font-weight: 500;
        color: #2e8b57;
        padding: 5px 0;
    }
    /* Hover effect */
    div[role="radiogroup"] label:hover {
        color: #1e5f3c;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Radio Menu
# -----------------------------
menu = st.sidebar.radio(
    "Go to:",
    ["🏠 Dashboard", "📊 Model Comparison", "📚 Prevention Guide", "📈 Session History"]
)


# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "🏠 Dashboard":
    st.markdown("<div class='main-title',margin-top:-20px>🌾 Paddy Crop Disease Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI dual-model predictions + treatment & history</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a paddy plant image", type=["jpg", "jpeg", "png"]) 
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error("Cannot open image: " + str(e))
            img = None

        if img is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### 📸 Uploaded Image")
                # resized preview for UI (small)
                preview_img = img.resize((400, 300))
                st.image(preview_img, use_container_width=False)

            with col2:
                st.markdown("### 🤖 Model Predictions")

                # Run predictions after the upload and check
                vgg_preds, vgg_label, vgg_acc = predict_image(vgg19_model, img)
                den_preds, den_label, den_acc = predict_image(densenet_model, img)

                # If either model indicated 'Not a rice image!' show error and stop
                if vgg_label == "Not a paddy image!" or den_label == "Not a paddy image!":
                    st.error("🚫 This does not look like a paddy image. Please upload a paddy leaf.")
                    st.stop()

                # VGG card
                st.markdown("#### 🧠 VGG19 Result")
                if vgg_label == "Model Not Loaded":
                    st.markdown("<div class='danger-card'><b>VGG19:</b> Not available</div>", unsafe_allow_html=True)
                else:
                    pretty_vgg = DISPLAY_NAMES.get(vgg_label, vgg_label)
                    st.markdown(f"<div class='card'><b>Prediction:</b> {pretty_vgg}<br><b>Confidence:</b> {vgg_acc:.2f}%</div>", unsafe_allow_html=True)

                # DenseNet card
                st.markdown("#### 🧠 DenseNet Result")
                if den_label == "Model Not Loaded":
                    st.markdown("<div class='danger-card'><b>DenseNet:</b> Not available</div>", unsafe_allow_html=True)
                else:
                    pretty_den = DISPLAY_NAMES.get(den_label, den_label)
                    st.markdown(f"<div class='card'><b>Prediction:</b> {pretty_den}<br><b>Confidence:</b> {den_acc:.2f}%</div>", unsafe_allow_html=True)

            # --- Final decision logic with CONFIDENCE THRESHOLD ---
            if (vgg_acc is None or vgg_acc == 0.0) and (den_acc is None or den_acc == 0.0):
                final_label = "Model(s) not loaded"
                final_conf = 0.0
                treatment_text = "Models are not available. Check model files."
                st.markdown(f"<div class='danger-card'><h4>⚠️ {final_label}</h4>{treatment_text}</div>", unsafe_allow_html=True)
            else:
                valid_vgg = (vgg_acc is not None and vgg_acc >= CONF_THRESHOLD)
                valid_den = (den_acc is not None and den_acc >= CONF_THRESHOLD)

                if not valid_vgg and not valid_den:
                    final_label = None
                    final_conf = max(vgg_acc or 0.0, den_acc or 0.0)
                    treatment_text = "Image does not clearly show a paddy leaf or disease. Please upload a clear paddy plant."
                    st.markdown(
                        f"<div class='danger-card'><h4>⚠️ Uncertain — Not a Paddy Leaf / Low Confidence</h4>{treatment_text}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # choose the best valid model
                    if valid_vgg and (not valid_den or vgg_acc >= den_acc):
                        final_label = vgg_label
                        final_conf = vgg_acc
                    else:
                        final_label = den_label
                        final_conf = den_acc

                    pretty_label = DISPLAY_NAMES.get(final_label, final_label)
                    treatment_text = TREATMENT_GUIDE.get(final_label, "No treatment available.")
                    card_class = "result-card" if final_conf >= CONF_THRESHOLD else "danger-card"
                    st.markdown(
                        f"<div class='{card_class}'><h4>🌱 Final Identified Condition: {pretty_label} ({final_conf:.2f}%)</h4>"
                        f"<b>Treatment / Remediation:</b><br>{treatment_text}</div>",
                        unsafe_allow_html=True
                    )

            # safe defaults before saving history
            if 'final_label' not in locals() or final_label is None:
                stored_final = "Uncertain"
            else:
                stored_final = final_label
            stored_conf = float(final_conf) if 'final_conf' in locals() else 0.0

            st.session_state.history.append({
                "Timestamp": str(datetime.datetime.now()),
                "Source": "Upload",
                "Final Prediction": stored_final,
                "Final Confidence %": stored_conf,
                "VGG19 Prediction": vgg_label if vgg_label is not None else "",
                "VGG19 Confidence %": float(vgg_acc) if vgg_acc is not None else 0.0,
                "DenseNet Prediction": den_label if den_label is not None else "",
                "DenseNet Confidence %": float(den_acc) if den_acc is not None else 0.0,
            })




# -----------------------------
# MODEL COMPARISON CHART
# -----------------------------
elif menu == "📊 Model Comparison":
    st.markdown("<div class='main-title'>📊 Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Visual summary of session predictions and per-class counts</div>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No history yet — make some predictions first (Upload.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.write("Recent predictions (most recent first):")
        st.dataframe(df_hist.sort_values("Timestamp", ascending=False).head(50))

        # per-class counts (final prediction)
        counts = df_hist["Final Prediction"].value_counts().reindex(CLASS_NAMES).fillna(0)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.barh(counts.index, counts.values)
        ax.set_xlabel("Number of Predictions")
        ax.set_title("Predicted Class Counts (Final Decision)")
        plt.tight_layout()
        st.pyplot(fig)

        # model accuracy-ish: average confidence by model (approx)
        vgg_conf = df_hist["VGG19 Confidence %"].dropna().astype(float)
        den_conf = df_hist["DenseNet Confidence %"].dropna().astype(float)
        fig2, ax2 = plt.subplots(figsize=(6,2))
        labels = ["VGG19", "DenseNet"]
        means = [vgg_conf.mean() if not vgg_conf.empty else 0.0, den_conf.mean() if not den_conf.empty else 0.0]
        ax2.bar(labels, means)
        ax2.set_ylabel("Average Confidence (%)")
        ax2.set_ylim(0,100)
        ax2.set_title("Average Model Confidence (historical)")
        st.pyplot(fig2)

# -----------------------------
# PREVENTION GUIDE
# -----------------------------
elif menu == "📚 Prevention Guide":
    st.markdown("<div class='main-title'>📚 Disease Prevention & Best Practices</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Practical steps farmers can take to reduce disease risk</div>", unsafe_allow_html=True)

    st.markdown("### 🌱 Cultural Practices")
    st.write("""
    - Use certified disease-free seeds and resistant varieties when available.
    - Maintain recommended spacing to reduce humidity and disease spread.
    - Rotate crops when possible and avoid continuous paddy cropping in same field.
    - Manage irrigation to avoid prolonged standing water except where needed.
    """)

    st.markdown("### 🧪 Nutrient & Chemical Management")
    st.write("""
    - Apply balanced fertilization; avoid excess nitrogen which can favor some pathogens.
    - Follow label instructions and local extension advice when applying fungicides/insecticides.
    - Use integrated pest management (IPM): combine biological, cultural, and chemical controls.
    """)

# -----------------------------
# SESSION HISTORY & EXPORT
# -----------------------------
elif menu == "📈 Session History":
    st.markdown("<div class='main-title'>📈 Session History</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>All recorded predictions during this session</div>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df.sort_values("Timestamp", ascending=False))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV History",
            data=csv,
            file_name="session_history.csv",
            mime="text/csv"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(f"""
<div class="footer">
    <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28'><rect width='100%' height='100%' fill='%232e8b57' rx='6'/><text x='50%' y='56%' font-size='16' text-anchor='middle' fill='white' font-family='Arial' dy='.3em'>🌾</text></svg>" class="logo" />
    Paddy Crop Disease Detection • Built with Streamlit • {datetime.datetime.now().year}
</div>
""", unsafe_allow_html=True)
