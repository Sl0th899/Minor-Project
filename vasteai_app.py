import os
import json
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

from ui import inject_css, header, show_preview, show_result

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------

APP_TITLE = "VasteAI"

MODEL_PATH = "custom_waste_model.keras"
CLASSES_PATH = "class_names.json"

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------

CLASS_LABELS = {

    "battery": {
        "title": "Battery",
        "emoji": "🔋",
        "color": "#FEE2E2",
        "border": "#B91C1C",
        "bin": "E-Waste",
        "instruction": "Take to an e-waste collection center."
    },

    "biological": {
        "title": "Biological",
        "emoji": "🍎",
        "color": "#ECFCCB",
        "border": "#65A30D",
        "bin": "Compost",
        "instruction": "Dispose in compost or organic waste."
    },

    "cardboard": {
        "title": "Cardboard",
        "emoji": "📦",
        "color": "#FEF3C7",
        "border": "#D97706",
        "bin": "Paper Recycling",
        "instruction": "Flatten before recycling."
    },

    "clothes": {
        "title": "Clothes",
        "emoji": "👕",
        "color": "#F3E8FF",
        "border": "#7E22CE",
        "bin": "Donation / Textile Recycling",
        "instruction": "Donate reusable clothes."
    },

    "glass": {
        "title": "Glass",
        "emoji": "🍶",
        "color": "#E0F7FA",
        "border": "#0891B2",
        "bin": "Glass Recycling",
        "instruction": "Rinse before recycling."
    },

    "metal": {
        "title": "Metal",
        "emoji": "🔩",
        "color": "#E2E8F0",
        "border": "#475569",
        "bin": "Metal Recycling",
        "instruction": "Recycle clean metal items."
    },

    "paper": {
        "title": "Paper",
        "emoji": "📄",
        "color": "#F8FAFC",
        "border": "#94A3B8",
        "bin": "Paper Recycling",
        "instruction": "Keep dry before disposal."
    },

    "plastic": {
        "title": "Plastic",
        "emoji": "♻️",
        "color": "#EFF6FF",
        "border": "#2563EB",
        "bin": "Plastic Recycling",
        "instruction": "Rinse plastic containers before recycling."
    },

    "shoes": {
        "title": "Shoes",
        "emoji": "👟",
        "color": "#FFEDD5",
        "border": "#EA580C",
        "bin": "Donation",
        "instruction": "Donate wearable shoes."
    },

    "trash": {
        "title": "Trash",
        "emoji": "🗑️",
        "color": "#FCE7F3",
        "border": "#BE185D",
        "bin": "General Waste",
        "instruction": "Dispose in general waste."
    }
}

# --------------------------------------------------
# LANGUAGE CONFIG
# --------------------------------------------------

LANGUAGES = {
    "English": "en",
    "हिंदी": "hi",
    "ਪੰਜਾਬੀ": "pa",
}

INSTRUCTIONS = {

    "en": {
        "upload": "Upload 1 or 2 images",
        "confidence": "Confidence",
        "not_loaded": "Model not found",
        "no_image": "Please upload image(s)"
    },

    "hi": {
        "upload": "1 या 2 छवियां अपलोड करें",
        "confidence": "विश्वास",
        "not_loaded": "मॉडल नहीं मिला",
        "no_image": "कृपया छवि अपलोड करें"
    },

    "pa": {
        "upload": "1 ਜਾਂ 2 ਤਸਵੀਰਾਂ ਅੱਪਲੋਡ ਕਰੋ",
        "confidence": "ਭਰੋਸਾ",
        "not_loaded": "ਮਾਡਲ ਨਹੀਂ ਮਿਲਿਆ",
        "no_image": "ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ"
    }
}

# --------------------------------------------------
# IMAGE UTILS
# --------------------------------------------------

def ensure_rgb(image: Image.Image):

    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_keras_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names

# --------------------------------------------------
# PREDICT IMAGE
# --------------------------------------------------

def predict_image(images, model, class_names):

    combined_predictions = None

    for image in images:
        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array, verbose=0)[0]

        if combined_predictions is None:
            combined_predictions = predictions
        else:
            combined_predictions += predictions

    combined_predictions /= len(images)

    print("\nPrediction Scores:")

    for i, prob in enumerate(combined_predictions):
        print(f"{class_names[i]}: {prob:.4f}")

    top_indices = np.argsort(combined_predictions)[-2:][::-1]

    top_results = []

    for idx in top_indices:

        top_results.append({
            "class": class_names[idx],
            "confidence": float(combined_predictions[idx])
        })

    return top_results

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

def main():

    st.set_page_config(
        page_title=APP_TITLE,
        layout="centered"
    )

    inject_css()

    header()

    # -----------------------------
    # LANGUAGE SELECT
    # -----------------------------

    language = st.selectbox(
        "Select Language",
        list(LANGUAGES.keys())
    )

    lang_key = LANGUAGES[language]

    # -----------------------------
    # IMAGE UPLOAD
    # -----------------------------

    uploaded_files = st.file_uploader(
        INSTRUCTIONS[lang_key]["upload"],
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    # -----------------------------
    # CHECK MODEL
    # -----------------------------

    if not os.path.exists(MODEL_PATH):

        st.error(INSTRUCTIONS[lang_key]["not_loaded"])

        return

    model, class_names = load_keras_model()

    # -----------------------------
    # NO IMAGE
    # -----------------------------

    if not uploaded_files:

        st.info(INSTRUCTIONS[lang_key]["no_image"])

        return

    # -----------------------------
    # LOAD IMAGES
    # -----------------------------

    images = []

    st.markdown("## Uploaded Images")

    for uploaded_file in uploaded_files[:2]:

        image = Image.open(uploaded_file)

        image = ensure_rgb(image)

        images.append(image)

        show_preview(image)

    # -----------------------------
    # PREDICTION
    # -----------------------------

    with st.spinner("Analyzing waste..."):

        results = predict_image(
            images,
            model,
            class_names
        )

    primary = results[0]

    secondary = results[1]

    # -----------------------------
    # PRIMARY RESULT
    # -----------------------------

    primary_data = CLASS_LABELS.get(
        primary["class"].lower(),
        CLASS_LABELS["trash"]
    )

    st.markdown("## Primary Prediction")

    show_result(
        primary["class"],
        primary["confidence"],
        primary_data,
        lang_key,
        INSTRUCTIONS
    )

    # -----------------------------
    # SECONDARY RESULT
    # -----------------------------

    st.markdown("---")

    secondary_data = CLASS_LABELS.get(
        secondary["class"].lower(),
        CLASS_LABELS["trash"]
    )

    st.markdown("## Also Looks Like")

    show_result(
        secondary["class"],
        secondary["confidence"],
        secondary_data,
        lang_key,
        INSTRUCTIONS
    )

    # -----------------------------
    # COMBINED RESULT
    # -----------------------------

    st.markdown("---")

    combined_text = (
        f"This waste most likely belongs to "
        f"'{primary['class'].title()}' "
        f"but may also resemble "
        f"'{secondary['class'].title()}'."
    )

    st.success(combined_text)

# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":
    main()