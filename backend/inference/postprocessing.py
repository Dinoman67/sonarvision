import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def decode_detections(
    output_tensor: np.ndarray,
    conf_threshold: float = 0.25,
    class_map: Dict[int, str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes raw YOLOv8 ONNX output tensor (1, 84, 1344) -> (boxes_xyxy, scores, class_ids).
    """
    if class_map is None:
        class_map = {0: "marine_debris"}

    # Transpose from (1, 84, 1344) to (1344, 84)
    if len(output_tensor.shape) == 3:
        predictions = output_tensor[0].T
    else:
        predictions = output_tensor.T

    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    # For YOLOv8-ESI marine debris detector, class 0 is the marine debris score
    if class_scores.shape[1] == 1:
        scores = class_scores[:, 0]
        class_ids = np.zeros(len(scores), dtype=int)
    else:
        # In case of multiple class columns
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

    # Filter by confidence threshold
    valid_mask = scores >= conf_threshold
    boxes_valid = boxes_cxcywh[valid_mask]
    scores_valid = scores[valid_mask]
    class_ids_valid = class_ids[valid_mask]

    if len(boxes_valid) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    # Convert cx, cy, w, h to x1, y1, x2, y2
    x1 = boxes_valid[:, 0] - boxes_valid[:, 2] / 2.0
    y1 = boxes_valid[:, 1] - boxes_valid[:, 3] / 2.0
    x2 = boxes_valid[:, 0] + boxes_valid[:, 2] / 2.0
    y2 = boxes_valid[:, 1] + boxes_valid[:, 3] / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return boxes_xyxy, scores_valid, class_ids_valid

def apply_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Applies Non-Maximum Suppression using OpenCV NMS.
    Returns indices of kept boxes.
    """
    if len(boxes_xyxy) == 0:
        return []

    # OpenCV NMSBoxes expects [x, y, w, h] format in integers or floats
    x = boxes_xyxy[:, 0]
    y = boxes_xyxy[:, 1]
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    boxes_xywh = np.stack([x, y, w, h], axis=1).tolist()
    scores_list = scores.tolist()

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores_list,
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return []
    return [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]

def unletterbox_boxes(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Transforms boxes from letterboxed space back to original image coordinate frame.
    """
    if len(boxes) == 0:
        return boxes

    boxes_scaled = boxes.copy()
    dw, dh = pad
    # Remove padding
    boxes_scaled[:, [0, 2]] -= dw
    boxes_scaled[:, [1, 3]] -= dh
    # Scale by ratio
    boxes_scaled[:, :] /= ratio

    # Clip to original image boundaries
    orig_h, orig_w = orig_shape[:2]
    boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, orig_w)
    boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, orig_h)

    return boxes_scaled
