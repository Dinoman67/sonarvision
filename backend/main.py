import os
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import HOST, PORT, SAMPLES_DIR, BASE_DIR
from backend.inference.engine import YOLOESIInferenceEngine
from backend.utils.samples_generator import ensure_sample_assets
from backend.api import health_router, metadata_router, analysis_router, export_router

FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once into singleton memory and verify assets
    print("==================================================================")
    print("  STARTING YOLO-ESI DEBRIS ANALYSIS WEB APPLICATION")
    print("==================================================================")
    ensure_sample_assets()
    engine = YOLOESIInferenceEngine()
    print(f"  Model: {engine.model_name} ({engine.model_path})")
    print(f"  SHA256: {engine.sha256_hash}")
    print(f"  Execution Provider: {engine.active_provider}")
    print("==================================================================")
    yield
    print("[Shutdown] Backend shutting down.")

app = FastAPI(
    title="YOLO-ESI Debris Intelligence Engine",
    description="Production Side-Scan Sonar & Geospatial Aerial Marine Debris Analysis Web API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router, prefix="/api")
app.include_router(metadata_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(export_router, prefix="/api")

# Mount static samples directory
app.mount("/static/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

# Mount frontend production dist if built
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        if full_path.startswith("api") or full_path.startswith("static"):
            return None
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend build not found"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
