import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_annotations(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    theme: str = "tactical_cyan"
) -> np.ndarray:
    """
    Draws professional remote-sensing / geospatial annotations on the image.
    Uses clean bounding boxes, target badges, and center point markers.
    """
    annotated = img.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    elif annotated.shape[2] == 4:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR)

    h, w = annotated.shape[:2]
    scale_factor = max(0.6, min(w, h) / 600.0)

    # Color palette (BGR)
    COLOR_PRIMARY = (255, 180, 0)      # Bright Cyan / Aqua
    COLOR_BG = (15, 23, 42)            # Slate 900
    COLOR_TEXT = (255, 255, 255)       # White
    COLOR_ACCENT = (0, 215, 255)       # Amber

    for det in detections:
        det_id = det["id"]
        cname = det["class_name"]
        conf = det["confidence"]
        box = det["bbox"]
        cx = int(round(det["center_pixel"]["x"]))
        cy = int(round(det["center_pixel"]["y"]))

        x1 = int(round(box["x1"]))
        y1 = int(round(box["y1"]))
        x2 = int(round(box["x2"]))
        y2 = int(round(box["y2"]))

        # Bounding box line thickness
        line_thick = max(2, int(round(2 * scale_factor)))

        # Draw main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PRIMARY, line_thick)

        # Draw corner crosshairs / brackets
        corner_len = max(6, int(min(x2 - x1, y2 - y1) * 0.25))
        # Top-left
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), COLOR_ACCENT, line_thick + 1)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), COLOR_ACCENT, line_thick + 1)
        # Top-right
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), COLOR_ACCENT, line_thick + 1)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), COLOR_ACCENT, line_thick + 1)
        # Bottom-left
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), COLOR_ACCENT, line_thick + 1)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), COLOR_ACCENT, line_thick + 1)
        # Bottom-right
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), COLOR_ACCENT, line_thick + 1)
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), COLOR_ACCENT, line_thick + 1)

        # Center target marker (subtle crosshair)
        cross_size = max(4, int(4 * scale_factor))
        cv2.line(annotated, (cx - cross_size, cy), (cx + cross_size, cy), COLOR_ACCENT, 1)
        cv2.line(annotated, (cx, cy - cross_size), (cx, cy + cross_size), COLOR_ACCENT, 1)
        cv2.circle(annotated, (cx, cy), max(2, int(2 * scale_factor)), COLOR_ACCENT, -1)

        # Label badge
        label_text = f"[ID {det_id:02d}] {cname} {conf * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, 0.45 * scale_factor)
        font_thick = max(1, int(round(1 * scale_factor)))

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

        # Place label above box if space permits, else below
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = badge_y1 + th + 8
        badge_x1 = x1
        badge_x2 = min(w, x1 + tw + 10)

        # Draw filled background pill for label
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), COLOR_BG, -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), COLOR_PRIMARY, 1)

        # Draw text
        text_pos = (badge_x1 + 5, badge_y2 - 5)
        cv2.putText(annotated, label_text, text_pos, font, font_scale, COLOR_TEXT, font_thick, cv2.LINE_AA)

    return annotated

def generate_detection_only_view(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    dim_factor: float = 0.2
) -> np.ndarray:
    """
    Generates a detection-focused view where background is dimmed
    and detected targets are highlighted with bounding overlays.
    """
    if len(img.shape) == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        base = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        base = img.copy()

    # Dim the entire background
    dimmed = (base.astype(np.float32) * dim_factor).astype(np.uint8)

    # Cut out the detected regions in full brightness
    for det in detections:
        box = det["bbox"]
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(base.shape[1], int(round(box["x2"])))
        y2 = min(base.shape[0], int(round(box["y2"])))

        # Add slight margin around crop
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(base.shape[1], x2 + pad_x)
        py2 = min(base.shape[0], y2 + pad_y)

        dimmed[py1:py2, px1:px2] = base[py1:py2, px1:px2]

    # Draw the annotation overlays
    return draw_annotations(dimmed, detections)
