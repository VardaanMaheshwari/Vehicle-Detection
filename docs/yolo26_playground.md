# YOLO26 Playground — Complete Functional Reference

> **YOLO Vision Studio v2.1.0** — Real-time Object Detection, Segmentation, Pose Estimation & Tracking powered by YOLO26, YOLO World v2, RT-DETR & Streamlit.

This document is a hands-on playground guide covering every YOLO26 capability available in this project, mapped to the actual weight files in `weights/` and the configuration in `config.py`.

---

## Table of Contents

1. [Available Weights & Model Catalog](#1-available-weights--model-catalog)
2. [Task 1: Object Detection](#2-task-1-object-detection)
3. [Task 2: Instance Segmentation](#3-task-2-instance-segmentation)
4. [Task 3: Pose Estimation](#4-task-3-pose-estimation)
5. [Task 4: YOLO World v2 — Open-Vocabulary Detection](#5-task-4-yolo-world-v2--open-vocabulary-detection)
6. [Task 5: YOLOE — Open-Vocabulary Detection + Segmentation](#6-task-5-yoloe--open-vocabulary-detection--segmentation)
7. [Tracking: ByteTrack & BoTSORT](#7-tracking-bytetrack--botsort)
8. [Inference Modes: Image vs Video](#8-inference-modes-image-vs-video)
9. [Video Sources: Stored, Webcam, RTSP, YouTube](#9-video-sources-stored-webcam-rtsp-youtube)
10. [Performance Tuning](#10-performance-tuning)
11. [Architecture Deep Dive](#11-architecture-deep-dive)
12. [Python API Quick Reference](#12-python-api-quick-reference)
13. [CLI Quick Reference](#13-cli-quick-reference)

---

## 1. Available Weights & Model Catalog

All model weights are stored in the `weights/` directory. Models not yet downloaded will be auto-fetched by Ultralytics on first use.

### 1.1 Currently Available Weight Files

| File | Task | Size Class | Notes |
|------|------|-----------|-------|
| `yolo26n.pt` | Detection | Nano | Fastest, lowest accuracy |
| `yolo26l.pt` | Detection | Large | Balanced speed/accuracy |
| `yolo26x.pt` | Detection | Xtra | Best accuracy, heaviest |
| `yolo26n-seg.pt` | Segmentation | Nano | Fastest segmentation |
| `yolo26l-seg.pt` | Segmentation | Large | Balanced segmentation |
| `yolo26n-pose.pt` | Pose Estimation | Nano | Fastest pose |
| `yolo26l-pose.pt` | Pose Estimation | Large | Balanced pose |
| `yoloe-26l-seg.pt` | YOLOE (Open-Vocab Seg) | Large | Text-prompted detection + segmentation |
| `yoloe-26x-seg.pt` | YOLOE (Open-Vocab Seg) | Xtra | Best YOLOE accuracy |
| `rtdetr-l.pt` | Detection (Transformer) | Large | RT-DETR transformer-based |
| `yolov8n.pt` | Detection (Legacy) | Nano | YOLOv8 baseline |
| `yolov8n-seg.pt` | Segmentation (Legacy) | Nano | YOLOv8 segmentation baseline |
| `yolov8n-cls.pt` | Classification (Legacy) | Nano | YOLOv8 classification |
| `yolov8s-worldv2.pt` | YOLO World v2 | Small | Open-vocab detection |
| `yolov8l-worldv2.pt` | YOLO World v2 | Large | Open-vocab detection (recommended) |
| `yolov8x-worldv2.pt` | YOLO World v2 | Xtra | Best open-vocab accuracy |

### 1.2 Full Model Catalog (config.py)

The app's `config.py` defines the complete catalog. Models not yet in `weights/` will auto-download:

#### Detection Models

| Display Label | Filename | Variant |
|--------------|----------|---------|
| YOLO26-nano (fastest) | `yolo26n.pt` | Nano |
| YOLO26-small | `yolo26s.pt` | Small |
| YOLO26-medium | `yolo26m.pt` | Medium |
| YOLO26-large | `yolo26l.pt` | Large |
| YOLO26-xlarge (best accuracy) | `yolo26x.pt` | Xtra |
| RT-DETR-Large (transformer) | `rtdetr-l.pt` | Large |
| RT-DETR-XLarge (transformer) | `rtdetr-x.pt` | Xtra |

#### Segmentation Models

| Display Label | Filename | Variant |
|--------------|----------|---------|
| YOLO26-nano-seg (fastest) | `yolo26n-seg.pt` | Nano |
| YOLO26-small-seg | `yolo26s-seg.pt` | Small |
| YOLO26-medium-seg | `yolo26m-seg.pt` | Medium |
| YOLO26-large-seg | `yolo26l-seg.pt` | Large |
| YOLO26-xlarge-seg (best accuracy) | `yolo26x-seg.pt` | Xtra |

#### Pose Estimation Models

| Display Label | Filename | Variant |
|--------------|----------|---------|
| YOLO26-nano-pose (fastest) | `yolo26n-pose.pt` | Nano |
| YOLO26-small-pose | `yolo26s-pose.pt` | Small |
| YOLO26-medium-pose | `yolo26m-pose.pt` | Medium |
| YOLO26-large-pose | `yolo26l-pose.pt` | Large |
| YOLO26-xlarge-pose (best accuracy) | `yolo26x-pose.pt` | Xtra |

#### YOLO World v2 Models (Open-Vocabulary Detection)

| Display Label | Filename | Variant |
|--------------|----------|---------|
| YOLOv8-small-worldv2 | `yolov8s-worldv2.pt` | Small |
| YOLOv8-medium-worldv2 | `yolov8m-worldv2.pt` | Medium |
| YOLOv8-large-worldv2 (recommended) | `yolov8l-worldv2.pt` | Large |
| YOLOv8-xlarge-worldv2 (best accuracy) | `yolov8x-worldv2.pt` | Xtra |

#### YOLOE Models (Open-Vocabulary Detection + Segmentation)

| Display Label | Filename | Variant |
|--------------|----------|---------|
| YOLOE-26n-seg (fastest) | `yoloe-26n-seg.pt` | Nano |
| YOLOE-26s-seg | `yoloe-26s-seg.pt` | Small |
| YOLOE-26m-seg | `yoloe-26m-seg.pt` | Medium |
| YOLOE-26l-seg (recommended) | `yoloe-26l-seg.pt` | Large |
| YOLOE-26x-seg (best accuracy) | `yoloe-26x-seg.pt` | Xtra |

---

## 2. Task 1: Object Detection

### Overview

YOLO26 detection uses an **anchor-free, NMS-free dual-head architecture**:

- **One-to-One Head (default):** Outputs `(N, 300, 6)` — up to 300 objects per frame, zero NMS overhead, ~43% CPU latency reduction vs YOLO11.
- **One-to-Many Head (training):** Outputs `(N, nc+4, 8400)` — higher accuracy but requires NMS post-processing.

### Playground Steps

1. **Select Task:** Choose `Detection` from the sidebar task radio.
2. **Select Model:** Pick from the detection catalog (Nano → Xtra, or RT-DETR).
3. **Set Confidence:** Adjust the confidence slider (default 40%, range 10–100%).
4. **Image Mode:** Upload an image or use the default → click `🚀 Run Detection`.
5. **Video Mode:** Choose a video source → frames are processed in real-time.

### What You Get

| Output | Description |
|--------|-------------|
| `result.boxes` | Bounding box coordinates (xyxy format) |
| `result.boxes.cls` | Class IDs per detection |
| `result.boxes.conf` | Confidence scores per detection |
| `result.names` | Class name dictionary `{id: name}` |
| Annotated image | BGR overlay with boxes, labels, and confidence % |

### Model Selection Guide

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| Edge / mobile deployment | `yolo26n.pt` | 2.4M params, 5.4B FLOPs, fastest |
| Real-time video (30+ FPS) | `yolo26s.pt` or `yolo26m.pt` | Good speed/accuracy trade-off |
| High-accuracy offline | `yolo26l.pt` or `yolo26x.pt` | 55–57.5 mAP, best COCO scores |
| Small objects (transformer) | `rtdetr-l.pt` or `rtdetr-x.pt` | Transformer attention excels at small targets |

### COCO Benchmarks (Detection)

| Model | mAP 50-95 (Std) | mAP 50-95 (E2E) | Params (M) | FLOPs (B) |
|-------|-----------------|------------------|-------------|------------|
| YOLO26n | 40.9 | 40.1 | 2.4 | 5.4 |
| YOLO26s | 48.6 | 47.8 | 9.5 | 20.7 |
| YOLO26m | 53.1 | 52.5 | 20.4 | 68.2 |
| YOLO26l | 55.0 | 54.4 | 24.8 | 86.4 |
| YOLO26x | 57.5 | 56.9 | 55.7 | 193.9 |

---

## 3. Task 2: Instance Segmentation

### Overview

YOLO26 segmentation adds a **lightweight mask branch** alongside bounding-box regression, featuring an upgraded **Multi-Scale Proto Module** that extracts structural data across multiple resolutions. This preserves crisp object boundaries and minimizes leakage around tightly packed instances.

### Playground Steps

1. **Select Task:** Choose `Segmentation` from the sidebar.
2. **Select Model:** Pick from the segmentation catalog (`yolo26n-seg.pt` → `yolo26x-seg.pt`).
3. **Set Confidence:** Adjust the confidence threshold.
4. **Run Inference:** Upload an image or process video frames.

### What You Get

| Output | Description |
|--------|-------------|
| `result.boxes` | Bounding boxes for each segmented object |
| `result.masks` | Instance segmentation masks (binary arrays) |
| `result.masks.xy` | Polygon coordinates for each mask |
| `result.masks.data` | Raw mask tensors |
| Annotated image | Overlay with colored masks + box outlines |

### Key Differences from Detection

- **Masks are pixel-level:** Each object gets a precise boundary polygon, not just a bounding rectangle.
- **Multi-Scale Proto:** Processes features at $P_3$, $P_4$, $P_5$ scales for better edge fidelity.
- **Semantic Segmentation Loss:** Acts as a regularizer during training for more stable feature maps.

---

## 4. Task 3: Pose Estimation

### Overview

YOLO26 pose models detect human body keypoints (17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) and draw skeleton connections between them.

### Playground Steps

1. **Select Task:** Choose `Pose Estimation` from the sidebar.
2. **Select Model:** Pick from the pose catalog (`yolo26n-pose.pt` → `yolo26x-pose.pt`).
3. **Set Confidence:** Lower thresholds help detect partially visible persons.
4. **Run Inference:** Best on images/videos with visible human subjects.

### What You Get

| Output | Description |
|--------|-------------|
| `result.keypoints` | Keypoint coordinates for each detected person |
| `result.keypoints.xy` | `(N, 17, 2)` — x,y coordinates per keypoint |
| `result.keypoints.conf` | Confidence per keypoint |
| `result.boxes` | Bounding boxes around each person |
| Annotated image | Skeleton overlay with colored joints and limb connections |

### Tips

- Use `yolo26n-pose.pt` for real-time webcam (fastest).
- Use `yolo26l-pose.pt` for offline analysis where accuracy matters.
- Lower confidence to ~25–30% for crowded or occluded scenes.

---

## 5. Task 4: YOLO World v2 — Open-Vocabulary Detection

### Overview

YOLO World v2 enables **zero-shot detection** using natural language text prompts. Unlike standard YOLO models that are limited to their 80 COCO training classes, YOLO World can detect **any object** you describe in plain English.

> **Key Feature:** Supports **descriptive phrases** like `"person in black"`, `"red car"`, `"wooden chair"`.

### Playground Steps

1. **Select Task:** Choose `YOLO World v2 (Text Prompt)` from the sidebar.
2. **Select Model:** Pick from the World catalog (`yolov8s-worldv2.pt` → `yolov8x-worldv2.pt`).
3. **Enter Text Prompt:** Type comma-separated classes or descriptive phrases.
   - Default: `person, car, dog, cat, chair, table, laptop, phone`
   - Try: `person in black, red car, wooden chair`
4. **Set Confidence:** Adjust threshold.
5. **Run Inference:** The model detects only the classes you specified.

### How It Works Internally

```python
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8l-worldv2.pt")
model.set_classes(["person in black", "red car"])  # text embeddings created
results = model.predict(image, conf=0.4)
```

> ⚠️ **Device Handling:** The app automatically handles CPU/CUDA mismatch when calling `set_classes()` by moving the model to CPU first, setting classes, then moving back to the best device.

### What You Get

| Output | Description |
|--------|-------------|
| `result.boxes` | Bounding boxes for matched text prompts |
| `result.names` | Maps class IDs to your custom text prompts |
| Annotated image | Boxes labeled with your custom text |

### Tips

- Use **descriptive phrases** for better accuracy (e.g., `"red car"` > just `"car"`).
- More specific prompts yield fewer false positives.
- `yolov8l-worldv2.pt` is the recommended balance of speed and accuracy.

---

## 6. Task 5: YOLOE — Open-Vocabulary Detection + Segmentation

### Overview

YOLOE combines **open-vocabulary detection with instance segmentation** — you provide category-level text prompts and get both bounding boxes AND pixel-level masks. It's the most feature-rich task in the studio.

> **Important:** YOLOE supports **category-level labels only** (e.g., `person`, `car`, `dog`). It does **NOT** support descriptive phrases like `"person in red shirt"`.

### Playground Steps

1. **Select Task:** Choose `YOLOE (Text → Segmentation)` from the sidebar.
2. **Select Model:** Pick from the YOLOE catalog (`yoloe-26n-seg.pt` → `yoloe-26x-seg.pt`).
3. **Enter Categories:** Type comma-separated category names.
   - Default: `person, car, dog, cat, chair, table, laptop, phone`
   - Use simple nouns only — no adjectives or phrases.
4. **Set Confidence:** Adjust threshold.
5. **Run Inference:** Get both boxes and masks for your categories.

### How It Works Internally

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26l-seg.pt")
model.set_classes(["person", "car", "dog"])  # category-level only
results = model.predict(image, conf=0.4)

for result in results:
    boxes = result.boxes    # bounding boxes
    masks = result.masks     # segmentation masks
```

### What You Get

| Output | Description |
|--------|-------------|
| `result.boxes` | Bounding boxes for each detected category |
| `result.masks` | Instance segmentation masks per detection |
| `result.names` | Maps class IDs to your category labels |
| Annotated image | Colored masks + box outlines with category labels |

### YOLO World v2 vs YOLOE

| Feature | YOLO World v2 | YOLOE |
|----------|--------------|-------|
| Detection (boxes) | ✅ | ✅ |
| Segmentation (masks) | ❌ | ✅ |
| Descriptive phrases | ✅ `"person in black"` | ❌ |
| Category-level only | ✅ | ✅ |
| Model family | YOLOv8-based | YOLO26-based |
| Best for | Flexible text prompts | Pixel-precise open-vocab |

---

## 7. Tracking: ByteTrack & BoTSORT

### Overview

The video inference pipeline supports **multi-object tracking** with persistent track IDs across frames. Two trackers are available:

| Tracker | Config | Best For |
|---------|--------|----------|
| **ByteTrack** | `bytetrack.yaml` | General tracking, handles occlusions well, simpler |
| **BoTSORT** | `botsort.yaml` | More robust re-identification, better for crowded scenes |

### How Tracking Works in the App

When tracking is enabled in video mode:

1. Each detected object gets a **persistent track ID** (e.g., `ID:5`).
2. The annotation label format becomes: `class | 87% | ID:5`.
3. Each track ID gets a **distinct color** from a 16-color palette.
4. **Local metrics** (per-frame) and **global metrics** (cumulative) are tracked.

### Track ID Color Palette

The app uses 16 distinct BGR colors for track visualization:

```
emerald, peter river, alizarin, sun flower, amethyst, turquoise,
carrot, wet asphalt, green sea, nephritis, belize hole, wisteria,
orange, pumpkin, pomegranate, asbestos
```

### Custom Annotation

The video service uses a custom `_annotate_with_ids()` function that renders:

- **Detection:** `class | conf%` labels on bounding boxes
- **Tracking:** `class | conf% | ID:N` labels with per-track colors
- **Segmentation:** Masks rendered first, then custom box labels overlaid
- **Pose:** Keypoints/skeleton rendered first, then custom box labels overlaid

---

## 8. Inference Modes: Image vs Video

### Image Inference (`📷 Image Inference`)

| Feature | Details |
|---------|---------|
| Input | Upload JPG, JPEG, PNG, BMP, WebP |
| Default image | `images/office_4.jpg` |
| Output | Side-by-side original vs annotated |
| Results panel | Expandable section with class counts, confidence metrics, and data table |
| Tracking | Not applicable (single frame) |

### Video Inference (`🎬 Video Inference`)

| Feature | Details |
|---------|---------|
| Sources | Stored video, Webcam, RTSP stream, YouTube |
| Tracking | ByteTrack / BoTSORT with persistent IDs |
| Skip frames | 1–8 (process every Nth frame for speed) |
| Output | Real-time frame-by-frame display |
| Metrics | Per-class count, average confidence, cumulative tracking stats |

---

## 9. Video Sources: Stored, Webcam, RTSP, YouTube

### 9.1 Stored Video

- Place video files in the `videos/` directory.
- Supported formats: `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.webm`.
- The app auto-scans the directory — newly added files appear without restart.

### 9.2 Webcam

- Uses browser-based webcam via `streamlit-webrtc`.
- Device index `0` (default webcam).
- Real-time processing with frame skipping for performance.

### 9.3 RTSP Stream

- Enter an RTSP URL (e.g., `rtsp://admin:password@192.168.1.100:554/stream`).
- Useful for IP cameras and surveillance systems.
- Frame skipping is recommended for high-bitrate streams.

### 9.4 YouTube

- Uses `yt-dlp` to fetch video streams.
- Enter any YouTube URL.
- Processing happens on the downloaded stream.

---

## 10. Performance Tuning

### 10.1 Confidence Threshold

| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| Model Confidence | 40% | 10–100% | Lower = more detections (more false positives), Higher = fewer detections (more precision) |

**Recommendations:**
- **Detection:** 25–40% for general use, 50–70% for precision-critical tasks.
- **Segmentation:** 30–50% — masks are already filtered by box confidence.
- **Pose:** 25–35% — lower thresholds help with occluded persons.
- **YOLO World / YOLOE:** 20–40% — open-vocab models benefit from lower thresholds.

### 10.2 Skip Frames (Video Only)

| Value | Use Case |
|-------|----------|
| 1 | Process every frame (smoothest, slowest) |
| 2 | Good balance for 30 FPS video |
| 3–4 | Fast processing for long videos |
| 5–8 | Maximum speed, choppy output |

### 10.3 Model Size Selection

```
Speed ←————————————————————————————→ Accuracy

yolo26n    ████████░░░░░░░░░░░░░░░░  (2.4M params,  5.4B FLOPs)
yolo26s    ████████████░░░░░░░░░░░░  (9.5M params, 20.7B FLOPs)
yolo26m    ████████████████░░░░░░░░  (20.4M params, 68.2B FLOPs)
yolo26l    ██████████████████░░░░░░  (24.8M params, 86.4B FLOPs)
yolo26x    ████████████████████████  (55.7M params, 193.9B FLOPs)
```

### 10.4 Device Selection

- **CUDA GPU:** Automatically used when available (`cuda:0`).
- **CPU fallback:** Used when no GPU is detected.
- The app handles CPU/CUDA mismatch for YOLO World and YOLOE models automatically.

---

## 11. Architecture Deep Dive

### 11.1 Core Innovations (YOLO26 vs YOLO11)

| Feature | YOLO11 | YOLO26 |
|---------|--------|--------|
| Post-processing | NMS required | NMS-free (one-to-one head) |
| Bounding box loss | DFL (Distribution Focal Loss) | DFL removed |
| Small object handling | Grid-quantization limited | STAL (Small-Target-Aware Label Assignment) |
| Loss balancing | Static | ProgLoss (Progressive Loss Balancing) |
| Optimizer | SGD / AdamW | MuSGD (SGD + Muon hybrid) |
| Output shape (detect) | `(N, nc+4, 8400)` | `(N, 300, 6)` (E2E) |
| CPU latency | Baseline | Up to 43% reduction |

### 11.2 Backbone & Neck

- **Backbone:** CSP-Darknet — extracts multi-scale features at $P_3$, $P_4$, $P_5$.
- **Neck:** PAN-FPN (Path Aggregation Network) — fuses features across scales for rich multi-scale representation.

### 11.3 Dual-Head Detection Architecture

```
Input Image
    │
    ▼
┌──────────────┐
│  CSP-Darknet  │  (Backbone)
│   Backbone    │
└──────┬───────┘
       │ P3, P4, P5
       ▼
┌──────────────┐
│   PAN-FPN     │  (Neck)
│     Neck      │
└──────┬───────┘
       │
    ┌──┴──┐
    ▼     ▼
┌────────┐  ┌────────────┐
│ 1-to-1  │  │  1-to-Many  │
│  Head   │  │    Head     │
│(deploy) │  │  (training) │
└────┬────┘  └─────┬──────┘
     │              │
     ▼              ▼
 (N,300,6)    (N,nc+4,8400)
  NMS-free      Needs NMS
```

### 11.4 Segmentation Architecture

```
Backbone Features (P3, P4, P5)
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────────┐
│  Box    │  │  Multi-Scale      │
│  Head   │  │  Proto Module     │
│(1-to-1) │  │  (P3+P4+P5 fuse)  │
└────┬────┘  └────────┬─────────┘
     │                │
     ▼                ▼
  Bounding       Instance
   Boxes          Masks
```

---

## 12. Python API Quick Reference

### 12.1 Loading Models

```python
from ultralytics import YOLO, YOLOWorld, YOLOE

# Standard YOLO26 models (auto-download if not in weights/)
det_model   = YOLO("weights/yolo26n.pt")        # Detection
seg_model   = YOLO("weights/yolo26n-seg.pt")    # Segmentation
pose_model  = YOLO("weights/yolo26n-pose.pt")   # Pose Estimation

# Open-vocabulary models
world_model = YOLOWorld("weights/yolov8l-worldv2.pt")  # YOLO World v2
yoloe_model = YOLOE("weights/yoloe-26l-seg.pt")       # YOLOE

# RT-DETR (transformer-based detection)
rtdetr_model = YOLO("weights/rtdetr-l.pt")
```

### 12.2 Running Inference

```python
# ── Detection ──
results = det_model.predict("image.jpg", conf=0.4)
for r in results:
    boxes = r.boxes          # Bounding boxes
    print(r.names)           # {0: 'person', 1: 'bicycle', ...}

# ── Segmentation ──
results = seg_model.predict("image.jpg", conf=0.4)
for r in results:
    masks = r.masks          # Instance masks
    boxes = r.boxes         # Bounding boxes

# ── Pose Estimation ──
results = pose_model.predict("image.jpg", conf=0.3)
for r in results:
    keypoints = r.keypoints  # (N, 17, 2) keypoints
    boxes = r.boxes

# ── YOLO World v2 (open-vocab detection) ──
world_model.set_classes(["person in black", "red car"])
results = world_model.predict("image.jpg", conf=0.4)

# ── YOLOE (open-vocab detection + segmentation) ──
yoloe_model.set_classes(["person", "car", "dog"])
results = yoloe_model.predict("image.jpg", conf=0.4)
for r in results:
    boxes = r.boxes
    masks = r.masks
```

### 12.3 Video Inference with Tracking

```python
# ── Video with ByteTrack ──
results = det_model.track(
    source="video.mp4",
    conf=0.4,
    tracker="bytetrack.yaml",
    persist=True,       # maintain state across frames
)

# ── Video with BoTSORT ──
results = det_model.track(
    source="video.mp4",
    conf=0.4,
    tracker="botsort.yaml",
    persist=True,
)

# ── Access track IDs ──
for r in results:
    if r.boxes.id is not None:
        track_ids = r.boxes.id.cpu().numpy().astype(int)
```

### 12.4 Model Export

```python
# Export to ONNX (for deployment)
det_model.export(format="onnx")

# Export to TensorRT (GPU deployment)
det_model.export(format="engine")  # requires TensorRT

# Export to CoreML (iOS)
det_model.export(format="coreml")

# Available formats: onnx, engine, coreml, tflite, edgetpu, openvino, etc.
```

---

## 13. CLI Quick Reference

### 13.1 Detection

```bash
# Basic detection
yolo task=detect mode=predict model=yolo26n.pt source="image.jpg"

# With confidence threshold
yolo task=detect mode=predict model=yolo26l.pt source="video.mp4" conf=0.4

# On GPU
yolo task=detect mode=predict model=yolo26x.pt source="video.mp4" device=0

# With tracking
yolo task=detect mode=track model=yolo26n.pt source="video.mp4" tracker=bytetrack.yaml
```

### 13.2 Segmentation

```bash
# Instance segmentation
yolo task=segment mode=predict model=yolo26n-seg.pt source="image.jpg"

# Video segmentation with tracking
yolo task=segment mode=track model=yolo26l-seg.pt source="video.mp4" tracker=botsort.yaml
```

### 13.3 Pose Estimation

```bash
# Pose estimation on image
yolo task=pose mode=predict model=yolo26n-pose.pt source="image.jpg" conf=0.3

# Video pose with tracking
yolo task=pose mode=track model=yolo26l-pose.pt source="video.mp4" tracker=bytetrack.yaml
```

### 13.4 Training (Fine-Tuning)

```bash
# Fine-tune detection on custom dataset
yolo task=detect mode=train model=yolo26s.pt data="custom.yaml" epochs=100 imgsz=640

# Fine-tune segmentation with MuSGD optimizer
yolo task=segment mode=train model=yolo26s-seg.pt data="custom.yaml" epochs=100 imgsz=640 optimizer=MuSGD

# Fine-tune pose estimation
yolo task=pose mode=train model=yolo26s-pose.pt data="custom_pose.yaml" epochs=200 imgsz=640
```

### 13.5 Export

```bash
# Export to ONNX
yolo export model=yolo26n.pt format=onnx

# Export to TensorRT
yolo export model=yolo26n.pt format=engine

# Export with dynamic batch size
yolo export model=yolo26n.pt format=onnx dynamic=True
```

---

## Quick-Start: Running the App

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501` with:

- **Sidebar:** Task selection, model picker, confidence slider, video source, tracking options, skip-frames.
- **Main area:** Image upload / video display with real-time inference results.
- **Results panel:** Expandable section with per-class metrics, confidence averages, and detailed data tables.

---

*Document generated for YOLO Vision Studio v2.1.0 — Last updated: June 2026*