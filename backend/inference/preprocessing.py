import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (256, 256),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    Returns (padded_image, ratio, (dw, dh)).
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def preprocess_tensor(img: np.ndarray) -> np.ndarray:
    """
    Preprocess BGR/Grayscale image to normalized float32 tensor (1, 3, H, W) in RGB.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    img = img.astype(np.float32) / 255.0
    tensor = np.transpose(img, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32)

def generate_tiles(
    img: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64
) -> List[Dict[str, Any]]:
    """
    Generates overlapping tiles for high-resolution imagery.
    """
    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    
    y_steps = max(1, int(np.ceil((h - overlap) / stride)))
    x_steps = max(1, int(np.ceil((w - overlap) / stride)))
    
    for y_idx in range(y_steps):
        for x_idx in range(x_steps):
            x1 = x_idx * stride
            y1 = y_idx * stride
            
            # Clamp right and bottom
            x2 = min(w, x1 + tile_size)
            y2 = min(h, y1 + tile_size)
            
            # If smaller than tile_size, shift back
            if x2 - x1 < tile_size and w >= tile_size:
                x1 = max(0, w - tile_size)
                x2 = w
            if y2 - y1 < tile_size and h >= tile_size:
                y1 = max(0, h - tile_size)
                y2 = h
                
            tile_img = img[y1:y2, x1:x2]
            tiles.append({
                "image": tile_img,
                "x_offset": x1,
                "y_offset": y1,
                "tile_w": x2 - x1,
                "tile_h": y2 - y1
            })
            
    return tiles
