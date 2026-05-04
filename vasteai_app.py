
import io
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple
from ui import inject_css, header, controls, show_preview, show_result
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

# Optional imports used only if available
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from skimage.feature import local_binary_pattern  # type: ignore
except Exception:
    local_binary_pattern = None


APP_TITLE = "VasteAI"
MODEL_PATH = "model.pkl"

CLASS_LABELS = {
    "glass": {
        "title": "Glass",
        "emoji": "🍶",
        "color": "#E0F7FA",
        "border": "#0891B2",
        "bin": "Glass recycling stream",
        "instruction": "Rinse and place in the glass recycling bin. Remove non-glass attachments when possible.",
        "tip": "Broken glass should be wrapped safely before disposal.",
    },
    "metal": {
        "title": "Metal",
        "emoji": "🔩",
        "color": "#F3E8FF",
        "border": "#9333EA",
        "bin": "Metal recycling stream",
        "instruction": "Place clean metal items in the recycling bin. Empty containers first.",
        "tip": "Metal is highly recyclable, so keep it out of regular trash when possible.",
    },
    "plastic": {
        "title": "Plastic",
        "emoji": "♻️",
        "color": "#EFF6FF",
        "border": "#2563EB",
        "bin": "Plastic recycling stream",
        "instruction": "Rinse plastic containers and place them in the recycling bin if accepted locally.",
        "tip": "Remove caps and lids if your local recycling rules ask for it.",
    },
    "trash": {
        "title": "Trash",
        "emoji": "🗑️",
        "color": "#FEE2E2",
        "border": "#DC2626",
        "bin": "General waste stream",
        "instruction": "This item should go to general trash unless your local facility accepts it.",
        "tip": "When in doubt, keep contaminated or mixed-material items out of recycling.",
    },
}

LANGUAGES = {
    "English": "en",
    "हिंदी": "hi",
    "ਪੰਜਾਬੀ": "pa",
}

INSTRUCTIONS = {
    "en": {
        "upload": "Upload a waste image",
        "camera": "Or take a photo",
        "predict": "Classify waste",
        "result": "Prediction",
        "confidence": "Confidence",
        "model": "Model",
        "input_hint": "Use a clear photo with the item centered in frame.",
        "not_loaded": "Model file not found. Put `model.pkl` in the same folder as this script.",
        "no_image": "Upload a photo or use the camera to classify waste.",
        "how_it_works": "The app loads a local `.pkl` model, extracts image features, scales them, and predicts the waste class.",
        "footer": "Built for a local VS Code + Streamlit workflow. No API keys required.",
    },
    "hi": {
        "upload": "अपशिष्ट की छवि अपलोड करें",
        "camera": "या फोटो लें",
        "predict": "वर्गीकरण करें",
        "result": "परिणाम",
        "confidence": "विश्वास स्तर",
        "model": "मॉडल",
        "input_hint": "स्पष्ट फोटो लें, जिसमें वस्तु बीच में हो।",
        "not_loaded": "मॉडल फ़ाइल नहीं मिली। `model.pkl` को इस script के same folder में रखें।",
        "no_image": "वर्गीकरण के लिए फोटो अपलोड करें या कैमरा उपयोग करें।",
        "how_it_works": "ऐप local `.pkl` model load करता है, image features निकालता है, scale करता है, और waste class predict करता है।",
        "footer": "VS Code + Streamlit local workflow के लिए बनाया गया। API keys की आवश्यकता नहीं है।",
    },
    "pa": {
        "upload": "ਕਚਰੇ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ",
        "camera": "ਜਾਂ ਫੋਟੋ ਲਵੋ",
        "predict": "ਵਰਗੀਕਰਨ ਕਰੋ",
        "result": "ਨਤੀਜਾ",
        "confidence": "ਭਰੋਸਾ",
        "model": "ਮਾਡਲ",
        "input_hint": "ਸਾਫ਼ ਫੋਟੋ ਲਵੋ, ਆਈਟਮ ਨੂੰ ਵਿਚਕਾਰ ਰੱਖੋ।",
        "not_loaded": "ਮਾਡਲ ਫ਼ਾਈਲ ਨਹੀਂ ਮਿਲੀ। `model.pkl` ਨੂੰ ਇਸ script ਦੇ same folder ਵਿੱਚ ਰੱਖੋ।",
        "no_image": "ਵਰਗੀਕਰਨ ਲਈ ਫੋਟੋ upload ਕਰੋ ਜਾਂ camera ਵਰਤੋ।",
        "how_it_works": "ਐਪ local `.pkl` model load ਕਰਦਾ ਹੈ, image features ਕੱਢਦਾ ਹੈ, scale ਕਰਦਾ ਹੈ, ਅਤੇ waste class predict ਕਰਦਾ ਹੈ।",
        "footer": "VS Code + Streamlit local workflow ਲਈ ਬਣਾਇਆ ਗਿਆ। API keys ਦੀ ਲੋੜ ਨਹੀਂ।",
    },
}


def ensure_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _safe_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0.0, 1.0), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def extract_features(image):
    import numpy as np

    # --- Step 1: Resize ---
    image = image.resize((43, 43))

    # --- Step 2: Grayscale (main features) ---
    gray = image.convert("L")
    gray_arr = np.array(gray) / 255.0
    gray_flat = gray_arr.flatten()  # 43x43 = 1849

    rgb = np.array(image) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    color_features = np.array([
        r.mean(), g.mean(), b.mean(),
        r.std(), g.std(), b.std()
    ])  # 6 features

    brightness = gray_arr.mean()
    contrast = gray_arr.std()
    min_val = gray_arr.min()
    max_val = gray_arr.max()

    extra_features = np.array([
        brightness,
        contrast,
        min_val,
        max_val
    ])  # 4 features

    features = np.concatenate([
        gray_flat,         # 1849
        color_features,    # +6 = 1855
        extra_features     # +4 = 1859
    ])

    if len(features) < 1871:
        features = np.pad(features, (0, 1871 - len(features)))
    else:
        features = features[:1871]

    return features.reshape(1, -1)

@dataclass
class LoadedBundle:
    model: object
    scaler: object
    classes: list


@st.cache_resource(show_spinner=False)
def load_bundle(model_path: str = MODEL_PATH) -> LoadedBundle:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "model" not in bundle or "scaler" not in bundle or "classes" not in bundle:
        raise ValueError("Unexpected model bundle format. Expected dict with model, scaler, classes.")

    return LoadedBundle(
        model=bundle["model"],
        scaler=bundle["scaler"],
        classes=list(bundle["classes"]),
    )


def predict_image(image: Image.Image, bundle: LoadedBundle) -> Tuple[str, float]:
    features = extract_features(image)
    scaled = bundle.scaler.transform(features)
    pred = bundle.model.predict(scaled)[0]
    proba = None

    if hasattr(bundle.model, "predict_proba"):
        try:
            proba = bundle.model.predict_proba(scaled)[0]
        except Exception:
            proba = None

    if proba is not None and len(proba) > 0:
        class_list = list(bundle.model.classes_) if hasattr(bundle.model, "classes_") else bundle.classes
        idx = class_list.index(pred) if pred in class_list else int(np.argmax(proba))
        confidence = float(proba[idx])
    elif hasattr(bundle.model, "decision_function"):
        confidence = 0.5
    else:
        confidence = 0.5

    return str(pred), float(confidence)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 22%),
                radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 20%),
                linear-gradient(180deg, #f8fafc 0%, #f8fafc 100%);
        }

        .hero {
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 52%, #0f766e 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            border: 1px solid rgba(255,255,255,0.12);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0.65rem 0 0 0;
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
            max-width: 64ch;
        }

        .glass-card {
            background: rgba(255,255,255,0.82);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 24px;
            padding: 1.25rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }

        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.65rem;
        }

        .soft-note {
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            color: #334155;
        }

        .result-wrap {
            border-radius: 24px;
            padding: 1.35rem 1.25rem;
            border-left: 6px solid;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
        }

        .badge {
            display: inline-block;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.85);
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid rgba(15,23,42,0.08);
        }

        .meter {
            height: 10px;
            width: 100%;
            background: rgba(15,23,42,0.08);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.35rem 0 0.9rem 0;
        }
        .meter > div {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #2563eb);
        }

        .mini-stat {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.86rem;
            padding: 0.6rem 0 1.2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    inject_css()
    header()

    img_source, lang_key = controls(LANGUAGES, INSTRUCTIONS)

    if img_source is None:
        st.info(INSTRUCTIONS[lang_key]["no_image"])
        st.stop()

    # LOAD MODEL (you forgot this)
    if not os.path.exists(MODEL_PATH):
        st.error(INSTRUCTIONS[lang_key]["not_loaded"])
        st.stop()

    try:
        bundle = load_bundle(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    image = Image.open(img_source)
    image = ensure_rgb(image)

    show_preview(image)

    with st.spinner("Analyzing..."):
        pred, confidence = predict_image(image, bundle)

    label_data = CLASS_LABELS.get(pred.lower(), CLASS_LABELS["trash"])

    show_result(pred, confidence, label_data, lang_key, INSTRUCTIONS)
