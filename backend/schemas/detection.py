from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X pixel coordinate")
    y1: float = Field(..., description="Top-left Y pixel coordinate")
    x2: float = Field(..., description="Bottom-right X pixel coordinate")
    y2: float = Field(..., description="Bottom-right Y pixel coordinate")

class CenterPixel(BaseModel):
    x: float = Field(..., description="Center X pixel coordinate")
    y: float = Field(..., description="Center Y pixel coordinate")

class Geolocation(BaseModel):
    latitude: float = Field(..., description="WGS84 Latitude in decimal degrees")
    longitude: float = Field(..., description="WGS84 Longitude in decimal degrees")
    crs: str = Field(..., description="Coordinate reference system of source imagery")
    coordinate_source: str = Field(..., description="Method used to derive coordinates (GeoTIFF transform, EXIF GPS)")
    utm_x: Optional[float] = Field(None, description="Projected Easting coordinate if applicable")
    utm_y: Optional[float] = Field(None, description="Projected Northing coordinate if applicable")

class DetectionRecord(BaseModel):
    id: int = Field(..., description="Unique detection sequence ID")
    class_id: int = Field(..., description="Class integer ID")
    class_name: str = Field(..., description="Class name label")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    bbox: BoundingBox = Field(..., description="Detection bounding box in pixel space")
    center_pixel: CenterPixel = Field(..., description="Center point in pixel space")
    geolocation: Optional[Geolocation] = Field(None, description="Geographic coordinates if georeferenced")

class FileMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    file_size_bytes: int
    file_size_human: str
    channels: int

class GeospatialMetadata(BaseModel):
    georeferenced: bool
    crs: Optional[str] = None
    bounds: Optional[Dict[str, Any]] = None
    pixel_resolution: Optional[List[float]] = None
    lat_lon_available: bool = False
    coordinate_source: Optional[str] = None
    status_message: str
    camera_latitude: Optional[float] = None
    camera_longitude: Optional[float] = None
    camera_altitude: Optional[float] = None
    capture_direction: Optional[float] = None
    footprint_geojson: Optional[Dict[str, Any]] = None

class ModelMetadata(BaseModel):
    model_name: str = "YOLOv8-ESI"
    format: str = "ONNX"
    input_resolution: str = "256x256"
    execution_provider: str
    model_path: str
    sha256_hash: str
    classes: Dict[int, str]
    architecture: str = "YOLOv8n + Squeeze-and-Excitation (SE) Attention"
    attention_mechanism: str = "SE Channel Attention in C2f Feature Blocks"

class AnalysisSummary(BaseModel):
    debris_detected: bool
    total_detections: int
    highest_confidence: Optional[float] = None
    average_confidence: Optional[float] = None
    class_counts: Dict[str, int] = Field(default_factory=dict)
    inference_time_ms: float
    total_time_ms: float
    status: str
    message: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    summary: AnalysisSummary
    detections: List[DetectionRecord]
    file_metadata: FileMetadata
    geospatial_metadata: GeospatialMetadata
    model_metadata: ModelMetadata
    original_image_url: str
    annotated_image_url: str
    detection_mask_url: str
    csv_export_url: str
    json_export_url: str
    pdf_report_url: str
