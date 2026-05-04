import streamlit as st

APP_TITLE = "VasteAI"

# -----------------------------
# GLOBAL CSS
# -----------------------------
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
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
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0.6rem 0 0 0;
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.85);
        border-radius: 24px;
        padding: 1.25rem;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    }

    .result-box {
        border-radius: 20px;
        padding: 1.25rem;
        margin-top: 1rem;
        border-left: 6px solid;
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------
def header():
    st.markdown(f"""
    <div class="hero">
        <h1>♻️ {APP_TITLE}</h1>
        <p>Smart Waste Classifier (Local ML)</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# INPUT CONTROLS
# -----------------------------
def controls(languages, instructions):
    col1, col2 = st.columns(2)

    with col1:
        lang = st.selectbox("Language", list(languages.keys()))
        lang_key = languages[lang]

    with col2:
        mode = st.radio("Input Mode", ["Upload", "Camera"])

    if mode == "Upload":
        img = st.file_uploader(
            instructions[lang_key]["upload"],
            type=["jpg", "png", "jpeg"]
        )
    else:
        img = st.camera_input(instructions[lang_key]["camera"])

    return img, lang_key


# -----------------------------
# IMAGE PREVIEW
# -----------------------------
def show_preview(image):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# RESULT DISPLAY
# -----------------------------
def show_result(pred, confidence, label_data, lang_key, instructions):
    st.markdown(f"""
    <div class="result-box" style="background:{label_data['color']}; border-left-color:{label_data['border']}">
        <h2>{label_data['emoji']} {label_data['title']}</h2>
        <p><b>Bin:</b> {label_data['bin']}</p>
        <p>{label_data['instruction']}</p>
        <p><b>{instructions[lang_key]['confidence']}:</b> {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)