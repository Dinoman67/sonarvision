import os
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException

from backend.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, UPLOADS_DIR

def validate_and_save_upload(file: UploadFile) -> Tuple[str, str, int]:
    """
    Validates uploaded file extension, size, and saves to secure temporary storage.
    Returns (saved_file_path, original_filename, file_size_bytes).
    """
    orig_name = file.filename or "uploaded_image"
    ext = Path(orig_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    unique_id = uuid.uuid4().hex
    safe_filename = f"{unique_id}_{Path(orig_name).name}"
    target_path = UPLOADS_DIR / safe_filename

    # Stream write and enforce file size limit
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = 0

    with open(target_path, "wb") as f:
        while chunk := file.file.read(65536):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
                )
            f.write(chunk)

    if total_bytes == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return str(target_path), orig_name, total_bytes
