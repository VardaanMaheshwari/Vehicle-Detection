"""
Vehicle Detection & Tracking — Main Application
================================================
Object detection and multi-object tracking for 6 vehicle classes,
powered by a custom fine-tuned YOLO11n model and Streamlit.
"""

import streamlit as st
import config
import image_service
import video_service

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption(f"v{config.APP_VERSION}")
    st.markdown("---")

    # 1️⃣  Inference mode ─────────────────────────────────────────────────────
    mode = st.radio("Inference Mode", config.MODES_LIST, key="mode")

    # 2️⃣  Task selection ─────────────────────────────────────────────────────
    task = st.radio("Task", config.TASKS_LIST, key="task")

    st.markdown("---")

    # 3️⃣  Model selector ─────────────────────────────────────────────────────
    catalog = config.get_model_catalog(task)
    model_label = st.selectbox(
        "🧠 Model",
        options=list(catalog.keys()),
        index=0,
        help="Custom Indian Vehicle Detector (fine-tuned YOLO11n) or a general COCO model.",
        key="model_select",
    )
    selected_model = catalog[model_label]

    st.markdown("---")

    # 4️⃣  Confidence slider ──────────────────────────────────────────────────
    confidence = (
        st.slider(
            "Model Confidence (%)",
            min_value=config.MIN_CONFIDENCE,
            max_value=config.MAX_CONFIDENCE,
            value=int(config.DEFAULT_CONFIDENCE * 100),
        )
        / 100.0
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.title(config.APP_TITLE)
st.markdown(f"*{config.APP_DESCRIPTION}*")

if mode == config.MODE_IMAGE:
    image_service.render(task, confidence, selected_model)
elif mode == config.MODE_VIDEO:
    video_service.render(task, confidence, selected_model)
else:
    st.error("Please select a valid inference mode.")
