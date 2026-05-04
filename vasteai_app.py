import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# UI imports (keep ONLY one inject_css from ui.py)
from ui import inject_css, header, controls, show_preview, show_result

APP_TITLE = "VasteAI"
MODEL_PATH = "model.pkl"

# -----------------------------
# CLASS LABELS
# -----------------------------
CLASS_LABELS = {
    "glass": {
        "title": "Glass",
        "emoji": "🍶",
        "color": "#E0F7FA",
        "border": "#0891B2",
        "bin": "Glass recycling stream",
        "instruction": "Rinse and place in the glass recycling bin.",
    },
    "metal": {
        "title": "Metal",
        "emoji": "🔩",
        "color": "#F3E8FF",
        "border": "#9333EA",
        "bin": "Metal recycling stream",
        "instruction": "Place clean metal items in recycling.",
    },
    "plastic": {
        "title": "Plastic",
        "emoji": "♻️",
        "color": "#EFF6FF",
        "border": "#2563EB",
        "bin": "Plastic recycling stream",
        "instruction": "Rinse and recycle if accepted locally.",
    },
    "trash": {
        "title": "Trash",
        "emoji": "🗑️",
        "color": "#FEE2E2",
        "border": "#DC2626",
        "bin": "General waste",
        "instruction": "Dispose in general waste.",
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
def extract_features(image):
    image = image.resize((43, 43))
    gray = image.convert("L")
    gray_arr = np.array(gray) / 255.0

    gray_flat = gray_arr.flatten()

    rgb = np.array(image) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    color_features = np.array([
        r.mean(), g.mean(), b.mean(),
        r.std(), g.std(), b.std()
    ])

    extra_features = np.array([
        gray_arr.mean(),
        gray_arr.std(),
        gray_arr.min(),
        gray_arr.max()
    ])

    features = np.concatenate([gray_flat, color_features, extra_features])

    if len(features) < 1871:
        features = np.pad(features, (0, 1871 - len(features)))
    else:
        features = features[:1871]

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