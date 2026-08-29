import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import rasterio
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from backend.config import (
    UPLOADS_DIR,
    RESULTS_DIR,
    SAMPLES_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)
from backend.inference.engine import YOLOESIInferenceEngine
from backend.geospatial.metadata import extract_geospatial_metadata
from backend.geospatial.coordinates import pixel_to_geographic
from backend.utils.annotator import draw_annotations, generate_detection_only_view
from backend.utils.file_validator import validate_and_save_upload
from backend.reports.pdf_report import create_pdf_report
from backend.reports.csv_report import generate_csv_report
from backend.reports.json_report import generate_json_report
from backend.schemas.detection import (
    AnalysisResponse, AnalysisSummary, DetectionRecord, FileMetadata,
    GeospatialMetadata, ModelMetadata, BoundingBox, CenterPixel, Geolocation
)

router = APIRouter(tags=["Analysis"])

def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def load_image_to_numpy(image_path: str) -> Tuple[np.ndarray, int, int, int]:
    """
    Safely loads TIFF, GeoTIFF, JPG, PNG into uint8 BGR numpy array.
    """
    path = str(image_path)
    ext = Path(path).suffix.lower()

    if ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(path) as src:
                # Read first 3 bands or single band
                if src.count >= 3:
                    arr = src.read([1, 2, 3])
                    arr = np.transpose(arr, (1, 2, 0)) # H, W, 3
                else:
                    arr = src.read(1) # H, W
                
                # Normalize if 16-bit or float
                if arr.dtype != np.uint8:
                    min_v, max_v = arr.min(), arr.max()
                    if max_v > min_v:
                        arr = ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)

                if len(arr.shape) == 2:
                    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    
                h, w = bgr.shape[:2]
                channels = 3
                return bgr, w, h, channels
        except Exception:
            pass

    # Standard fallback via OpenCV / PIL
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            pil_img = Image.open(path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")

    if len(img.shape) == 2:
        h, w = img.shape
        channels = 1
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        h, w, channels = img.shape
        if channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            channels = 3

    return img, w, h, channels

def run_full_pipeline(
    image_path: str,
    orig_filename: str,
    file_size_bytes: int,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    use_tiling: Optional[bool] = None
) -> AnalysisResponse:
    t_start = time.perf_counter()
    analysis_id = uuid.uuid4().hex
    analysis_dir = RESULTS_DIR / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image
    img_bgr, width, height, channels = load_image_to_numpy(image_path)
    file_format = Path(orig_filename).suffix.upper().replace(".", "")

    # 2. Extract Geospatial Metadata
    geo_meta_raw = extract_geospatial_metadata(image_path, orig_filename=orig_filename)

    # 3. Run YOLO-ESI ONNX inference
    engine = YOLOESIInferenceEngine()
    detections_raw, timing = engine.predict(
        img_bgr,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        use_tiling=use_tiling
    )

    # 4. Resolve geographic coordinates for each detection
    detections_list: List[DetectionRecord] = []
    class_counts: Dict[str, int] = {}
    confidences = []

    for d in detections_raw:
        cid = d["class_id"]
        cname = d["class_name"]
        conf = d["confidence"]
        confidences.append(conf)
        class_counts[cname] = class_counts.get(cname, 0) + 1

        # Calculate geospatial coordinates using image metadata
        geo_dict = pixel_to_geographic(
            d["center_pixel"]["x"],
            d["center_pixel"]["y"],
            geo_meta_raw
        )
        geolocation = Geolocation(**geo_dict) if geo_dict else None

        detections_list.append(DetectionRecord(
            id=d["id"],
            class_id=cid,
            class_name=cname,
            confidence=conf,
            bbox=BoundingBox(**d["bbox"]),
            center_pixel=CenterPixel(**d["center_pixel"]),
            geolocation=geolocation
        ))

    # 5. Generate Visualizations
    annotated_bgr = draw_annotations(img_bgr, detections_raw)
    mask_bgr = generate_detection_only_view(img_bgr, detections_raw)

    # Save artifacts
    orig_save_path = analysis_dir / "original.png"
    annotated_save_path = analysis_dir / "annotated.png"
    mask_save_path = analysis_dir / "mask.png"
    pdf_save_path = analysis_dir / "report.pdf"
    csv_save_path = analysis_dir / "detections.csv"
    json_save_path = analysis_dir / "results.json"

    cv2.imwrite(str(orig_save_path), img_bgr)
    cv2.imwrite(str(annotated_save_path), annotated_bgr)
    cv2.imwrite(str(mask_save_path), mask_bgr)

    # 6. Build Summary Metrics
    total_dets = len(detections_list)
    debris_detected = total_dets > 0
    max_conf = max(confidences) if confidences else None
    avg_conf = (sum(confidences) / len(confidences)) if confidences else None

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    summary = AnalysisSummary(
        debris_detected=debris_detected,
        total_detections=total_dets,
        highest_confidence=round(max_conf, 4) if max_conf else None,
        average_confidence=round(avg_conf, 4) if avg_conf else None,
        class_counts=class_counts,
        inference_time_ms=timing["inference_time_ms"],
        total_time_ms=total_time_ms,
        status="SUCCESS",
        message="Analysis completed successfully." if debris_detected else "No objects exceeded the current confidence threshold."
    )

    file_metadata = FileMetadata(
        filename=orig_filename,
        format=file_format,
        width=width,
        height=height,
        file_size_bytes=file_size_bytes,
        file_size_human=format_file_size(file_size_bytes),
        channels=channels
    )

    geospatial_metadata = GeospatialMetadata(
        georeferenced=geo_meta_raw["georeferenced"],
        crs=geo_meta_raw.get("crs"),
        bounds=geo_meta_raw.get("bounds"),
        pixel_resolution=geo_meta_raw.get("pixel_resolution"),
        lat_lon_available=geo_meta_raw.get("lat_lon_available", False),
        coordinate_source=geo_meta_raw.get("coordinate_source"),
        status_message=geo_meta_raw.get("status_message"),
        camera_latitude=geo_meta_raw.get("camera_latitude"),
        camera_longitude=geo_meta_raw.get("camera_longitude"),
        camera_altitude=geo_meta_raw.get("camera_altitude"),
        capture_direction=geo_meta_raw.get("capture_direction"),
        footprint_geojson=geo_meta_raw.get("footprint_geojson")
    )

    model_metadata = ModelMetadata(
        **engine.get_metadata()
    )

    # Construct response dictionary for exports
    analysis_dict = {
        "analysis_id": analysis_id,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary.model_dump(),
        "detections": [d.model_dump() for d in detections_list],
        "file_metadata": file_metadata.model_dump(),
        "geospatial_metadata": geospatial_metadata.model_dump(),
        "model_metadata": model_metadata.model_dump()
    }

    # 7. Generate Reports
    csv_content = generate_csv_report([d.model_dump() for d in detections_list], geo_meta_raw)
    csv_save_path.write_text(csv_content)

    json_content = generate_json_report(analysis_dict)
    json_save_path.write_text(json_content)

    create_pdf_report(
        analysis_data=analysis_dict,
        annotated_image_path=str(annotated_save_path),
        output_path=str(pdf_save_path)
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        timestamp=analysis_dict["timestamp"],
        summary=summary,
        detections=detections_list,
        file_metadata=file_metadata,
        geospatial_metadata=geospatial_metadata,
        model_metadata=model_metadata,
        original_image_url=f"/api/export/{analysis_id}/original",
        annotated_image_url=f"/api/export/{analysis_id}/annotated",
        detection_mask_url=f"/api/export/{analysis_id}/mask",
        csv_export_url=f"/api/export/{analysis_id}/csv",
        json_export_url=f"/api/export/{analysis_id}/json",
        pdf_report_url=f"/api/export/{analysis_id}/pdf"
    )

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD),
    use_tiling: Optional[bool] = Form(None)
):
    saved_path, orig_name, file_size = validate_and_save_upload(file)
    try:
        response = run_full_pipeline(
            image_path=saved_path,
            orig_filename=orig_name,
            file_size_bytes=file_size,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            use_tiling=use_tiling
        )
        return response
    finally:
        # Cleanup uploaded raw temp file
        Path(saved_path).unlink(missing_ok=True)

class SampleRequest(BaseModel):
    sample_id: str
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    use_tiling: Optional[bool] = None

@router.post("/analyze-sample", response_model=AnalysisResponse)
async def analyze_sample_image(req: SampleRequest):
    sample_map = {
        "geotiff_debris": "sample_noaa_geotiff_debris.tif",
        "sss_marine_debris": "sample_sss_marine_debris.png",
        "seabed_background": "sample_seabed_background.png",
        "drone_aerial_geotagged": "sample_drone_aerial_geotagged.jpg"
    }

    if req.sample_id not in sample_map:
        raise HTTPException(status_code=404, detail=f"Unknown sample ID '{req.sample_id}'")

    sample_filename = sample_map[req.sample_id]
    sample_path = SAMPLES_DIR / sample_filename

    if not sample_path.exists():
        from backend.utils.samples_generator import ensure_sample_assets
        ensure_sample_assets()

    if not sample_path.exists():
        raise HTTPException(status_code=500, detail="Sample asset not found on server.")

    file_size = sample_path.stat().st_size
    return run_full_pipeline(
        image_path=str(sample_path),
        orig_filename=sample_filename,
        file_size_bytes=file_size,
        conf_threshold=req.confidence_threshold,
        iou_threshold=req.iou_threshold,
        use_tiling=req.use_tiling
    )
