"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
YOLO_DIR = DATA / "yolo"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
WEIGHTS_DIR = ROOT / "weights"  # pretrained checkpoints downloaded from GitHub releases

CLASSES = ["Bacterial_spot", "Black_rot", "Healthy", "Late_Blight", "Powder_Mildew"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 128        # classifier input size (PlantVillage originals are 256x256)
DET_IMG_SIZE = 256    # detector input size
SEED = 42

_REL = "https://github.com/ultralytics/assets/releases/download/"
PRETRAINED_URLS = {
    "yolo26n-cls.pt": _REL + "v8.4.0/yolo26n-cls.pt",   # ImageNet classification
    "yolo11n-cls.pt": _REL + "v8.4.0/yolo11n-cls.pt",
    "yolov8n-cls.pt": _REL + "v8.3.0/yolov8n-cls.pt",
    "yolo26n.pt": _REL + "v8.4.0/yolo26n.pt",           # COCO detection
    "yolo11n.pt": _REL + "v8.4.0/yolo11n.pt",
    "yolov5nu.pt": _REL + "v8.3.0/yolov5nu.pt",
}
