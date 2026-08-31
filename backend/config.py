import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"
SAMPLES_DIR = BASE_DIR / "static" / "samples"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def find_model_path() -> str:
    """Auto-discovers the YOLO-ESI ONNX model path from env, candidates, or recursive scan."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path

    candidates = [
        PROJECT_ROOT / "models" / "yolo_esi_fp16.onnx",
        PROJECT_ROOT / "models" / "best.onnx",
        PROJECT_ROOT / "yolo_esi_fp16.onnx",
        PROJECT_ROOT / "best.onnx",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    for search_dir in [PROJECT_ROOT / "models", PROJECT_ROOT / "runs", PROJECT_ROOT]:
        if search_dir.exists():
            onnx_files = list(search_dir.glob("**/*.onnx"))
            if onnx_files:
                return str(onnx_files[0])

    return str(PROJECT_ROOT / "models" / "yolo_esi_fp16.onnx")

MODEL_PATH = find_model_path()

DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
DEFAULT_IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MODEL_INPUT_SIZE = (256, 256)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "150"))
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

MODEL_CLASSES = {0: "marine_debris"}
