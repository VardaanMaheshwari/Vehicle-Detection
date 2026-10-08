"""
Centralized model loading with Streamlit caching.
Models are loaded once and reused across reruns.
"""

import streamlit as st
from ultralytics import YOLO

import config


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load a YOLO detection model.

    Checks the local ``weights/`` directory first (via
    ``config.resolve_model_path``); falls back to ultralytics
    auto-download, then sweeps stray weights into ``weights/``.
    """
    path = config.resolve_model_path(model_name)
    model = YOLO(path)
    config.sweep_stray_weights()
    return model


def get_model_for_task(task: str, model_name: str | None = None) -> YOLO | None:
    """Return the detection model (cached).

    Parameters
    ----------
    task : str
        One of the ``config.TASKS_LIST`` values (detection only here).
    model_name : str | None
        Specific model filename chosen via the sidebar.
        Falls back to ``config.DETECTION_MODEL`` if ``None``.
    """
    name = model_name or config.DETECTION_MODEL
    try:
        return load_model(name)
    except Exception as exc:
        st.error(f"❌ Failed to load model for **{task}**: {exc}")
        return None


def load_fresh_model(task: str, model_name: str | None = None) -> YOLO:
    """Load a **fresh** (uncached) model instance.

    Used by multi-video mode so each video gets isolated tracking
    state (ByteTrack / BoTSORT state lives on the model instance).
    """
    name = model_name or config.DETECTION_MODEL
    path = config.resolve_model_path(name)
    m = YOLO(path)
    config.sweep_stray_weights()
    return m
