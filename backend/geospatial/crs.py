from typing import Optional, Any
import rasterio.crs
import pyproj

def get_crs_name(crs_obj: Any) -> Optional[str]:
    """
    Returns a normalized, human-readable CRS string (e.g. 'EPSG:4326', 'EPSG:26916').
    """
    if crs_obj is None:
        return None

    # Already string
    if isinstance(crs_obj, str):
        crs_str = crs_obj.strip()
        if crs_str.upper().startswith("EPSG:"):
            return crs_str.upper()
        try:
            parsed = pyproj.CRS.from_user_input(crs_str)
            epsg = parsed.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
            return parsed.name or crs_str
        except Exception:
            return crs_str

    # Integer EPSG code
    if isinstance(crs_obj, int):
        return f"EPSG:{crs_obj}"

    # rasterio.crs.CRS or pyproj.CRS
    try:
        if hasattr(crs_obj, "to_epsg"):
            epsg = crs_obj.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
        if hasattr(crs_obj, "to_string"):
            s = crs_obj.to_string()
            if s and not s.startswith("+"):
                return s
        if hasattr(crs_obj, "name") and crs_obj.name:
            return crs_obj.name
        return str(crs_obj)
    except Exception:
        return str(crs_obj)

def parse_crs(crs_input: Any) -> Optional[rasterio.crs.CRS]:
    """
    Parses various CRS representations into a rasterio CRS object.
    """
    if crs_input is None:
        return None
    if isinstance(crs_input, rasterio.crs.CRS):
        return crs_input
    try:
        if isinstance(crs_input, pyproj.CRS):
            epsg = crs_input.to_epsg()
            if epsg:
                return rasterio.crs.CRS.from_epsg(epsg)
            return rasterio.crs.CRS.from_wkt(crs_input.to_wkt())
        return rasterio.crs.CRS.from_user_input(crs_input)
    except Exception:
        try:
            return rasterio.crs.CRS.from_string(str(crs_input))
        except Exception:
            return None
