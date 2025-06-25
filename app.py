import streamlit as st
from dashboard.components.uploader import upload_images
from dashboard.components.visual_blocks import display_visuals
from dashboard.inference.inference_runner import InferenceRunner
from dashboard.components.role_selector import get_user_role

# Page config
st.set_page_config(
    page_title="AI Retinal Diagnostic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 👁️ Title Block (Centered)
st.markdown("""
    <div style='text-align: center;'>
        <h1>👁️ Akeso AI-Powered Retinal Diagnostic</h1>
        <a href="https://wandb.ai/diganta-dutta-097/akeso-eyecare" target="_blank">
            <button style="
                background-color: #FFBE0B;
                border: none;
                color: black;
                padding: 10px 24px;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
            ">
                🔗 View Model Dashboard (W&B)
            </button>
        </a>
    </div>
    <hr style='margin-top: 20px;'>
""", unsafe_allow_html=True)

# Sidebar
user_role = get_user_role()
show_legend = st.sidebar.checkbox("Show Overlay Legend", value=True)

# 🧠 Inference Engine
inference_engine = InferenceRunner(
    s3_bucket="akeso-eyecare",
    model_keys={
        "grading": "outputs/checkpoints/grading.pt",
        "segmentation": "outputs/checkpoints/segmentor.pt",
        "localization": "outputs/checkpoints/localizer_od_fovea_best.pt"
    }
)

# 📤 Upload Block (Centered)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h3 style='text-align: center;'>📤 Upload Retinal Image(s)</h3>", unsafe_allow_html=True)
    uploaded_files = upload_images()

# 📊 Inference Results
if uploaded_files:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align: center;'>📊 Diagnostic Results</h3>", unsafe_allow_html=True)

    for file in uploaded_files:
        name, image = file
        inference_results = inference_engine.infer_all(image)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            display_visuals(
                file_data=file,
                user_role=user_role,
                show_legend=show_legend,
                inference_results=inference_results
            )
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: gray;'>Upload a retinal image or a zip of images to begin.</div>",
            unsafe_allow_html=True
        )