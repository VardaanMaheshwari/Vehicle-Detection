"""
Image inference service — vehicle object detection.

Provides a single ``render()`` entry-point called from *app.py*.
"""

from __future__ import annotations

import PIL.Image
import pandas as pd
import streamlit as st

import config
from model_loader import get_model_for_task


# ── Public API ────────────────────────────────────────────────────────────────


def render(task: str, confidence: float, selected_model: str | None = None) -> None:
    """Render the image-inference page for the chosen *task*."""
    st.header(f"📷 Image · {task}")

    model = get_model_for_task(task, model_name=selected_model)
    if model is None:
        return

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
    )

    col1, col2 = st.columns(2)

    with col1:
        if uploaded:
            image = PIL.Image.open(uploaded)
            st.image(image, caption="Original Image", width="stretch")
        elif config.DEFAULT_IMAGE.exists():
            st.image(
                str(config.DEFAULT_IMAGE),
                caption="Default Image",
                width="stretch",
            )

    with col2:
        if uploaded:
            if st.button(f"🚀 Run {task}", type="primary", width="stretch"):
                _run_inference(model, image, confidence, task)
        elif config.DEFAULT_DETECT_IMAGE.exists():
            st.image(
                str(config.DEFAULT_DETECT_IMAGE),
                caption="Detected Image",
                width="stretch",
            )


# ── Inference logic ───────────────────────────────────────────────────────────


def _run_inference(model, image: PIL.Image.Image, confidence: float, task: str) -> None:
    with st.spinner(f"Running {task}…"):
        results = model.predict(image, conf=confidence)
        result = results[0]

        annotated = result.plot()[:, :, ::-1]  # BGR → RGB
        st.image(annotated, caption=f"{task} Result", width="stretch")

    _display_results(result)


def _display_results(result) -> None:
    """Show structured detection results below the annotated image."""
    with st.expander("📊 Results", expanded=True):
        if result.boxes is not None and len(result.boxes):
            names = result.names
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            # Summary metrics (count + avg confidence per class)
            summary: dict[str, dict] = {}
            for cls_id, conf in zip(classes, confs):
                name = names[int(cls_id)]
                entry = summary.setdefault(name, {"count": 0, "total_conf": 0.0})
                entry["count"] += 1
                entry["total_conf"] += float(conf)

            cols = st.columns(min(len(summary), 4) or 1)
            for idx, (name, data) in enumerate(summary.items()):
                avg = data["total_conf"] / data["count"]
                cols[idx % len(cols)].metric(
                    label=name.capitalize(),
                    value=data["count"],
                    delta=f"{avg:.0%} avg conf",
                )

            # Detailed table
            rows = [
                {"Class": names[int(c)], "Confidence": f"{cf:.2%}"}
                for c, cf in zip(classes, confs)
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")
