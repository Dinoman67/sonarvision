from fastapi import APIRouter
from backend.inference.engine import YOLOESIInferenceEngine

router = APIRouter(tags=["Metadata"])

@router.get("/model-info")
async def get_model_info():
    engine = YOLOESIInferenceEngine()
    return engine.get_metadata()

@router.get("/samples")
async def get_available_samples():
    return [
        {
            "id": "geotiff_debris",
            "name": "NOAA H11833 GeoTIFF Crop (Georeferenced)",
            "type": "GeoTIFF (EPSG:26916)",
            "description": "Real 512x512 GeoTIFF crop from NOAA side-scan sonar survey containing georeferenced marine obstruction debris.",
            "has_geolocation": True,
            "filename": "sample_noaa_geotiff_debris.tif"
        },
        {
            "id": "sss_marine_debris",
            "name": "NOAA SSS Debris Target (Sonar PNG)",
            "type": "High-Contrast Sonar PNG",
            "description": "512x512 side-scan sonar image with acoustic shadow and high-reflectivity marine debris target.",
            "has_geolocation": False,
            "filename": "sample_sss_marine_debris.png"
        },
        {
            "id": "seabed_background",
            "name": "Clean Seabed Background (No Debris)",
            "type": "Seabed Acoustic Texture",
            "description": "High-resolution side-scan sonar recording of natural seabed ripple textures without debris targets.",
            "has_geolocation": False,
            "filename": "sample_seabed_background.png"
        },
        {
            "id": "drone_aerial_geotagged",
            "name": "Aerial Drone Survey (EXIF GPS)",
            "type": "Drone Aerial JPG",
            "description": "Coastal aerial drone survey image equipped with standard EXIF GPS positioning metadata.",
            "has_geolocation": True,
            "filename": "sample_drone_aerial_geotagged.jpg"
        }
    ]
