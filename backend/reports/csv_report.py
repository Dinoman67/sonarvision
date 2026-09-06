import csv
import io
from typing import List, Dict, Any

def generate_csv_report(
    detections: List[Dict[str, Any]],
    geospatial_meta: Dict[str, Any]
) -> str:
    """
    Generates structured CSV export for all detections.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "detection_id",
        "object_type",
        "class_name",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "latitude",
        "longitude",
        "crs",
        "coordinate_source"
    ])

    mapping = {
        "unknown_debris": "Marine Debris",
        "marine_debris": "Marine Debris",
        "airplane": "Submerged Aircraft",
        "mine": "Naval Mine",
        "wreck": "Shipwreck",
    }

    for det in detections:
        box = det.get("bbox", {})
        cp = det.get("center_pixel", {})
        geo = det.get("geolocation") or {}
        cname = det.get("class_name", "")
        obj_type = det.get("object_type") or mapping.get(str(cname).lower(), str(cname).replace("_", " ").title())

        writer.writerow([
            det.get("id"),
            obj_type,
            det.get("class_name"),
            det.get("class_id"),
            f"{det.get('confidence', 0.0):.4f}",
            box.get("x1"),
            box.get("y1"),
            box.get("x2"),
            box.get("y2"),
            cp.get("x"),
            cp.get("y"),
            geo.get("latitude") if geo.get("latitude") is not None else "",
            geo.get("longitude") if geo.get("longitude") is not None else "",
            geo.get("crs") if geo.get("crs") is not None else (geospatial_meta.get("crs") if geospatial_meta.get("georeferenced") else ""),
            geo.get("coordinate_source") if geo.get("coordinate_source") is not None else (geospatial_meta.get("coordinate_source") if geospatial_meta.get("georeferenced") else "")
        ])

    return output.getvalue()
