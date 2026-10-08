"""
Configuration hub for the Vehicle Detection & Tracking app.
All paths, model config, UI settings, and constants are defined here.
"""

from pathlib import Path
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ASSETS_DIR = ROOT / "assets"
IMAGES_DIR = ROOT / "images"
VIDEOS_DIR = ROOT / "videos"
WEIGHTS_DIR = ROOT / "weights"

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(exist_ok=True)

# ─── App Metadata ────────────────────────────────────────────────────────────
APP_TITLE = "Vehicle Detection & Tracking"
APP_ICON = "🚗"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Object detection and multi-object tracking for 6 vehicle classes, "
    "powered by a custom fine-tuned YOLO11n model."
)

# ─── Inference Modes ─────────────────────────────────────────────────────────
MODE_IMAGE = "📷 Image Inference"
MODE_VIDEO = "🎬 Video Inference"
MODES_LIST = [MODE_IMAGE, MODE_VIDEO]

# ─── Tasks ───────────────────────────────────────────────────────────────────
TASK_DETECT = "Detection"
TASKS_LIST = [TASK_DETECT]

# ─── Video Sources ───────────────────────────────────────────────────────────
SOURCE_STORED = "Stored Video"
SOURCE_YOUTUBE = "YouTube"
VIDEO_SOURCES = [SOURCE_STORED, SOURCE_YOUTUBE]

# ─── Model Catalog ────────────────────────────────────────────────────────────
# {display_label: model_filename}. Bare filenames — resolve_model_path() adds
# the weights/ prefix. Ultralytics auto-downloads any model not already present.
DETECTION_MODELS = {
    "Indian Vehicle Detector": "indian_vehicle_yolo11n.pt",
    "YOLO26-nano (COCO, general)": "yolo26n.pt",
}

# Default model filename
DETECTION_MODEL = "indian_vehicle_yolo11n.pt"

# ─── Default Assets ──────────────────────────────────────────────────────────
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# ─── Video Catalog ───────────────────────────────────────────────────────────
_VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.webm")


def get_videos_dict() -> dict[str, Path]:
    """Scan ``videos/`` directory each time so newly added files appear."""
    if not VIDEOS_DIR.exists():
        return {}
    vids: dict[str, Path] = {}
    for ext in _VIDEO_EXTENSIONS:
        for p in sorted(VIDEOS_DIR.glob(ext)):
            vids[p.stem] = p
    return dict(sorted(vids.items()))


# Kept for backward compat — but prefer get_videos_dict()
VIDEOS_DICT = get_videos_dict()

# ─── Inference Defaults ──────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.50
MIN_CONFIDENCE = 10  # slider min (%)
MAX_CONFIDENCE = 100  # slider max (%)
VIDEO_DISPLAY_WIDTH = 720

# ─── Skip Frames ─────────────────────────────────────────────────────────────
DEFAULT_SKIP_FRAMES = 1  # process every frame
MIN_SKIP_FRAMES = 1
MAX_SKIP_FRAMES = 8

# ─── Tracker Config ──────────────────────────────────────────────────────────
TRACKER_BYTETRACK = "bytetrack.yaml"
TRACKER_BOTSORT = "botsort.yaml"
TRACKERS_LIST = [TRACKER_BYTETRACK, TRACKER_BOTSORT]


def resolve_model_path(model_name: str) -> str:
    """Return local weights path if it exists, else the bare name for auto-download.

    After auto-download, call ``sweep_stray_weights()`` to move any
    ``.pt`` files that landed in the project root into ``weights/``.
    """
    local = WEIGHTS_DIR / model_name
    if local.exists():
        return str(local)
    # Not in weights/ yet — check project root (old download location)
    root_copy = ROOT / model_name
    if root_copy.exists():
        root_copy.rename(local)
        return str(local)
    # Will be auto-downloaded to CWD; return bare name
    return model_name


def sweep_stray_weights() -> None:
    """Move any ``.pt`` files from the project root into ``weights/``."""
    for pt_file in ROOT.glob("*.pt"):
        dest = WEIGHTS_DIR / pt_file.name
        if not dest.exists():
            pt_file.rename(dest)


def get_model_catalog(task: str) -> dict[str, str]:
    """Return ``{display_label: filename}`` for the given *task*.

    The app is detection-only, so this always returns the detection catalog.
    """
    return DETECTION_MODELS
