import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from skimage.feature import hog # NEW: For texture/edge detection

# UI imports
from ui import inject_css, header, controls, show_preview, show_result

APP_TITLE = "VasteAI"
MODEL_PATH = "model.pkl"

# -----------------------------
# UPDATED CLASS LABELS (Added Cardboard)
# -----------------------------
# -----------------------------
# EXPANDED CLASS LABELS (10 Categories)
# -----------------------------
CLASS_LABELS = {
    "battery": {
        "title": "Battery (E-Waste)",
        "emoji": "🔋",
        "color": "#FEE2E2", # Red-ish warning
        "border": "#B91C1C",
        "bin": "E-Waste / Hazardous",
        "instruction": "DO NOT put in normal bins. Take to a battery recycling drop-off.",
    },
    "biological": {
        "title": "Biological / Organic",
        "emoji": "🍎",
        "color": "#ECFCCB", # Green/Yellow
        "border": "#65A30D",
        "bin": "Compost / Organics",
        "instruction": "Place in your compost or green waste bin.",
    },
    "cardboard": {
        "title": "Cardboard",
        "emoji": "📦",
        "color": "#FEF3C7", # Warm brown/yellow
        "border": "#D97706",
        "bin": "Paper recycling stream",
        "instruction": "Keep dry and flatten before recycling.",
    },
    "clothes": {
        "title": "Clothes / Textiles",
        "emoji": "👕",
        "color": "#F3E8FF", # Purple
        "border": "#7E22CE",
        "bin": "Textile donation / Recycling",
        "instruction": "Donate if usable, or take to a textile recycling point.",
    },
    "glass": {
        "title": "Glass",
        "emoji": "🍶",
        "color": "#E0F7FA", # Cyan
        "border": "#0891B2",
        "bin": "Glass recycling stream",
        "instruction": "Rinse out liquids before recycling. Do not break.",
    },
    "metal": {
        "title": "Metal",
        "emoji": "🔩",
        "color": "#E2E8F0", # Gray
        "border": "#475569",
        "bin": "Metal recycling stream",
        "instruction": "Place clean metal items in recycling.",
    },
    "paper": {
        "title": "Paper",
        "emoji": "📄",
        "color": "#F8FAFC", # White/Gray
        "border": "#94A3B8",
        "bin": "Paper recycling stream",
        "instruction": "Keep dry. Shred confidential documents if necessary.",
    },
    "plastic": {
        "title": "Plastic",
        "emoji": "♻️",
        "color": "#EFF6FF", # Blue
        "border": "#2563EB",
        "bin": "Plastic recycling stream",
        "instruction": "Rinse and recycle. Check local rules for film plastics.",
    },
    "shoes": {
        "title": "Shoes",
        "emoji": "👟",
        "color": "#FFEDD5", # Orange
        "border": "#EA580C",
        "bin": "Donation / Textile bin",
        "instruction": "Tie pairs together and donate if in good condition.",
    },
    "trash": {
        "title": "General Trash",
        "emoji": "🗑️",
        "color": "#FCE7F3", # Pink/Red
        "border": "#BE185D",
        "bin": "General waste",
        "instruction": "Dispose in general waste. Cannot be recycled.",
    },
}
# -----------------------------
# LANGUAGE CONFIG
# -----------------------------
LANGUAGES = {
    "English": "en",
    "हिंदी": "hi",
    "ਪੰਜਾਬੀ": "pa",
}

INSTRUCTIONS = {
    "en": {
        "upload": "Upload a waste image",
        "camera": "Or take a photo",
        "confidence": "Confidence",
        "not_loaded": "model.pkl not found",
        "no_image": "Upload or capture an image",
    },
    "hi": {
        "upload": "छवि अपलोड करें",
        "camera": "फोटो लें",
        "confidence": "विश्वास",
        "not_loaded": "model.pkl नहीं मिला",
        "no_image": "छवि अपलोड करें",
    },
    "pa": {
        "upload": "ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ",
        "camera": "ਫੋਟੋ ਲਵੋ",
        "confidence": "ਭਰੋਸਾ",
        "not_loaded": "model.pkl ਨਹੀਂ ਮਿਲੀ",
        "no_image": "ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ",
    },
}

# -----------------------------
# IMAGE HELPERS
# -----------------------------
def ensure_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(image: Image.Image):
    image = image.resize((64, 64))
    
    # 1. Shape & Texture Features (HOG)
    gray_image = np.array(image.convert("L"))
    hog_features = hog(
        gray_image, 
        orientations=8, 
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        feature_vector=True
    )
    
    # 2. Advanced Color Features (HSV Histograms)
    hsv_image = image.convert("HSV")
    h, s, v = hsv_image.split()
    
    h_hist, _ = np.histogram(np.array(h).flatten(), bins=16, range=(0, 256))
    s_hist, _ = np.histogram(np.array(s).flatten(), bins=8, range=(0, 256))
    v_hist, _ = np.histogram(np.array(v).flatten(), bins=8, range=(0, 256))
    
    color_features = np.concatenate([h_hist, s_hist, v_hist]) / (64 * 64)

    features = np.concatenate([hog_features, color_features])
    
    return features.reshape(1, -1)


# -----------------------------
# MODEL LOADING
# -----------------------------
@dataclass
class LoadedBundle:
    model: object
    scaler: object
    classes: list


@st.cache_resource
def load_bundle(path=MODEL_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return LoadedBundle(
        model=data["model"],
        scaler=data["scaler"],
        classes=list(data["classes"]),
    )


# -----------------------------
# PREDICTION
# -----------------------------
def predict_image(image, bundle: LoadedBundle) -> Tuple[str, float]:
    features = extract_features(image)
    scaled = bundle.scaler.transform(features)

    pred = bundle.model.predict(scaled)[0]

    if hasattr(bundle.model, "predict_proba"):
        probs = bundle.model.predict_proba(scaled)[0]
        idx = list(bundle.model.classes_).index(pred)
        confidence = float(probs[idx])
    else:
        confidence = 0.5

    return str(pred), confidence


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")

    inject_css()
    header()

    img_source, lang_key = controls(LANGUAGES, INSTRUCTIONS)

    if img_source is None:
        st.info(INSTRUCTIONS[lang_key]["no_image"])
        return

    if not os.path.exists(MODEL_PATH):
        st.error(INSTRUCTIONS[lang_key]["not_loaded"])
        return

    bundle = load_bundle(MODEL_PATH)

    image = Image.open(img_source)
    image = ensure_rgb(image)

    show_preview(image)

    with st.spinner("Analyzing..."):
        pred, confidence = predict_image(image, bundle)

    label_data = CLASS_LABELS.get(pred.lower(), CLASS_LABELS["trash"])

    show_result(pred, confidence, label_data, lang_key, INSTRUCTIONS)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()