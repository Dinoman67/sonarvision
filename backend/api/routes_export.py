import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import RESULTS_DIR

router = APIRouter(prefix="/export", tags=["Exports"])

def get_result_file(analysis_id: str, filename: str, media_type: str):
    analysis_dir = RESULTS_DIR / analysis_id
    file_path = analysis_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Export file '{filename}' for analysis '{analysis_id}' not found.")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )

@router.get("/{analysis_id}/pdf")
async def download_pdf_report(analysis_id: str):
    return get_result_file(analysis_id, "report.pdf", "application/pdf")

@router.get("/{analysis_id}/csv")
async def download_csv_report(analysis_id: str):
    return get_result_file(analysis_id, "detections.csv", "text/csv")

@router.get("/{analysis_id}/json")
async def download_json_report(analysis_id: str):
    return get_result_file(analysis_id, "results.json", "application/json")

@router.get("/{analysis_id}/annotated")
async def get_annotated_image(analysis_id: str):
    return get_result_file(analysis_id, "annotated.png", "image/png")

@router.get("/{analysis_id}/original")
async def get_original_image(analysis_id: str):
    return get_result_file(analysis_id, "original.png", "image/png")

@router.get("/{analysis_id}/mask")
async def get_detection_mask(analysis_id: str):
    return get_result_file(analysis_id, "mask.png", "image/png")
