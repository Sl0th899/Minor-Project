import streamlit as st

APP_TITLE = "VasteAI"

def inject_css():
    st.markdown("""
    <style>
    body { font-family: 'Inter', sans-serif; }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1d4ed8, #0f766e);
        color: white;
    }

    .glass-card {
        background: rgba(255,255,255,0.85);
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .result-box {
        padding: 1rem;
        border-radius: 15px;
        border-left: 5px solid;
    }
    </style>
    """, unsafe_allow_html=True)


def header():
    st.markdown(f"""
    <div class="hero">
        <h1>♻️ {APP_TITLE}</h1>
        <p>Smart Waste Classifier (Local ML)</p>
    </div>
    """, unsafe_allow_html=True)


def controls(languages, instructions):
    col1, col2 = st.columns(2)

    with col1:
        lang = st.selectbox("Language", list(languages.keys()))
        lang_key = languages[lang]

    with col2:
        mode = st.radio("Input Mode", ["Upload", "Camera"])

    if mode == "Upload":
        img = st.file_uploader(instructions[lang_key]["upload"], type=["jpg", "png", "jpeg"])
    else:
        img = st.camera_input(instructions[lang_key]["camera"])

    return img, lang_key


def show_preview(image):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_result(pred, confidence, label_data, lang_key, instructions):
    st.markdown(f"""
    <div class="result-box" style="background:{label_data['color']}; border-left-color:{label_data['border']}">
        <h2>{label_data['emoji']} {label_data['title']}</h2>
        <p><b>Bin:</b> {label_data['bin']}</p>
        <p>{label_data['instruction']}</p>
        <p><b>{instructions[lang_key]['confidence']}:</b> {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)