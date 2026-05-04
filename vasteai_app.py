import os
import json
from typing import Tuple
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

# UI imports
from ui import inject_css, header, controls, show_preview, show_result

APP_TITLE = "VasteAI"
MODEL_PATH = "custom_waste_model.keras"
CLASSES_PATH = "class_names.json"

# -----------------------------
# CLASS LABELS (Your 10 Categories)
# -----------------------------
CLASS_LABELS = {
    "battery": {"title": "Battery", "emoji": "🔋", "color": "#FEE2E2", "border": "#B91C1C", "bin": "E-Waste / Hazardous", "instruction": "DO NOT put in normal bins. Take to a drop-off."},
    "biological": {"title": "Biological", "emoji": "🍎", "color": "#ECFCCB", "border": "#65A30D", "bin": "Compost", "instruction": "Place in your compost bin."},
    "cardboard": {"title": "Cardboard", "emoji": "📦", "color": "#FEF3C7", "border": "#D97706", "bin": "Paper recycling", "instruction": "Keep dry and flatten."},
    "clothes": {"title": "Clothes", "emoji": "👕", "color": "#F3E8FF", "border": "#7E22CE", "bin": "Textile donation", "instruction": "Donate or textile recycle."},
    "glass": {"title": "Glass", "emoji": "🍶", "color": "#E0F7FA", "border": "#0891B2", "bin": "Glass recycling", "instruction": "Rinse out liquids. Do not break."},
    "metal": {"title": "Metal", "emoji": "🔩", "color": "#E2E8F0", "border": "#475569", "bin": "Metal recycling", "instruction": "Place clean metal items in recycling."},
    "paper": {"title": "Paper", "emoji": "📄", "color": "#F8FAFC", "border": "#94A3B8", "bin": "Paper recycling", "instruction": "Keep dry. Shred confidential docs."},
    "plastic": {"title": "Plastic", "emoji": "♻️", "color": "#EFF6FF", "border": "#2563EB", "bin": "Plastic recycling", "instruction": "Rinse and recycle."},
    "shoes": {"title": "Shoes", "emoji": "👟", "color": "#FFEDD5", "border": "#EA580C", "bin": "Donation", "instruction": "Tie pairs together and donate."},
    "trash": {"title": "Trash", "emoji": "🗑️", "color": "#FCE7F3", "border": "#BE185D", "bin": "General waste", "instruction": "Dispose in general waste."},
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
    "en": {"upload": "Upload image", "camera": "Or take photo", "confidence": "Confidence", "not_loaded": "Model not found", "no_image": "Upload an image"},
    "hi": {"upload": "अपलोड करें", "camera": "फोटो लें", "confidence": "विश्वास", "not_loaded": "मॉडल नहीं मिला", "no_image": "छवि अपलोड करें"},
    "pa": {"upload": "ਅੱਪਲੋਡ ਕਰੋ", "camera": "ਫੋਟੋ ਲਵੋ", "confidence": "ਭਰੋਸਾ", "not_loaded": "ਮਾਡਲ ਨਹੀਂ ਮਿਲਿਆ", "no_image": "ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ"},
}

def ensure_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_keras_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

# -----------------------------
# PREDICTION
# -----------------------------
def predict_image(image: Image.Image, model, class_names) -> Tuple[str, float]:
    # Resize to match the 128x128 we used in training
    img = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

    predictions = model.predict(img_array)[0]
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

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

    model, class_names = load_keras_model()

    image = Image.open(img_source)
    image = ensure_rgb(image)
    show_preview(image)

    with st.spinner("Analyzing with CNN..."):
        pred, confidence = predict_image(image, model, class_names)

    label_data = CLASS_LABELS.get(pred.lower(), CLASS_LABELS["trash"])
    show_result(pred, confidence, label_data, lang_key, INSTRUCTIONS)

if __name__ == "__main__":
    main()