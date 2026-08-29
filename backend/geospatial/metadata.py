import os
import re
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from pyproj import Transformer
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from backend.geospatial.crs import get_crs_name, parse_crs

# Authoritative NOAA H11833 parent raster metadata
NOAA_PARENT_TIFF_METADATA = {
    "H11833_1of2.tif": {
        "crs": "EPSG:26916",
        "raw_crs": CRS.from_epsg(26916),
        "transform": Affine(0.5, 0.0, 261030.75, 0.0, -0.5, 3213746.75),
        "width": 33928,
        "height": 32328,
        "pixel_size": (0.5, 0.5),
    },
    "H11833_2of2.tif": {
        "crs": "EPSG:26916",
        "raw_crs": CRS.from_epsg(26916),
        "transform": Affine(0.5, 0.0, 261065.25, 0.0, -0.5, 3213661.75),
        "width": 33948,
        "height": 31996,
        "pixel_size": (0.5, 0.5),
    }
}

# Cache for NOAA survey patch geolocation records
_NOAA_PATCH_CACHE: Optional[Dict[str, Dict[str, Any]]] = None

def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _safe_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def _load_noaa_patch_manifest() -> Dict[str, Dict[str, Any]]:
    global _NOAA_PATCH_CACHE
    if _NOAA_PATCH_CACHE is not None:
        return _NOAA_PATCH_CACHE

    manifest_map: Dict[str, Dict[str, Any]] = {}
    base_dirs = [
        Path("/home/ashish/sonar-vision/datasets/noaa-debris"),
        Path(__file__).resolve().parent.parent.parent / "datasets" / "noaa-debris",
        Path("datasets/noaa-debris"),
    ]

    for base in base_dirs:
        if not base.exists():
            continue

        # 1. E3 crop metadata (e3, e3_fixed, e3_v2)
        for rel_csv in [
            "e3/metadata/crop_metadata.csv",
            "e3_fixed/metadata/crop_metadata.csv",
            "e3_v2/crop_metadata.csv",
            "e3_v2/metadata/crop_metadata.csv",
        ]:
            csv_file = base / rel_csv
            if csv_file.exists():
                try:
                    with open(csv_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            cid = r.get("crop_id", "").strip()
                            fn = r.get("image_filename", "").strip()
                            src_tiff = r.get("source_tiff", "H11833_1of2.tif").strip()
                            cx = _safe_int(r.get("crop_x"))
                            cy = _safe_int(r.get("crop_y"))
                            cw = _safe_int(r.get("crop_width")) or 512
                            ch = _safe_int(r.get("crop_height")) or 512
                            data = {
                                "crop_id": cid,
                                "image_filename": fn,
                                "source_tiff": src_tiff,
                                "crop_x": cx,
                                "crop_y": cy,
                                "crop_width": cw,
                                "crop_height": ch,
                                "utm_x": _safe_float(r.get("utm_x")),
                                "utm_y": _safe_float(r.get("utm_y")),
                                "target_id": r.get("target_id"),
                                "target_latitude": _safe_float(r.get("target_latitude")),
                                "target_longitude": _safe_float(r.get("target_longitude")),
                                "is_positive": r.get("is_positive") == "true",
                            }
                            if fn:
                                manifest_map[fn.lower()] = data
                                manifest_map[Path(fn).stem.lower()] = data
                            if cid:
                                manifest_map[cid.lower()] = data
                except Exception:
                    pass

        # 2. G7 crop manifest
        g7_csv = base / "g7" / "crop_manifest.csv"
        if g7_csv.exists():
            try:
                with open(g7_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        fn = r.get("image_filename", "").strip()
                        src_tiff = r.get("source_tiff", "H11833_1of2.tif").strip()
                        px = _safe_int(r.get("pixel_x"))
                        py = _safe_int(r.get("pixel_y"))
                        data = {
                            "crop_id": Path(fn).stem,
                            "image_filename": fn,
                            "source_tiff": src_tiff,
                            "crop_x": px,
                            "crop_y": py,
                            "crop_width": 512,
                            "crop_height": 512,
                            "utm_x": _safe_float(r.get("utm_x")),
                            "utm_y": _safe_float(r.get("utm_y")),
                        }
                        if fn:
                            manifest_map[fn.lower()] = data
                            manifest_map[Path(fn).stem.lower()] = data
            except Exception:
                pass

        # 3. E4 crop metadata
        e4_csv = base / "e4" / "crop_metadata.csv"
        if e4_csv.exists():
            try:
                with open(e4_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        cid = r.get("crop_id", "").strip()
                        src_tiff = r.get("source_tiff", "H11833_1of2.tif").strip()
                        tid = r.get("target_id", "").strip()
                        target_data = None
                        if tid:
                            for k, v in manifest_map.items():
                                if v.get("target_id") == tid and v.get("source_tiff") == src_tiff:
                                    target_data = v
                                    break
                        data = {
                            "crop_id": cid,
                            "image_filename": f"{cid}.png",
                            "source_tiff": src_tiff,
                            "target_id": tid,
                            "utm_x": target_data.get("utm_x") if target_data else None,
                            "utm_y": target_data.get("utm_y") if target_data else None,
                            "crop_x": target_data.get("crop_x") if target_data else None,
                            "crop_y": target_data.get("crop_y") if target_data else None,
                            "crop_width": 512,
                            "crop_height": 512,
                        }
                        if cid:
                            manifest_map[cid.lower()] = data
                            manifest_map[f"{cid.lower()}.png"] = data
            except Exception:
                pass

        # 4. H8 Test manifest (TEST_00000.png...)
        h8_test_csv = base / "h8_test" / "test_manifest.csv"
        if h8_test_csv.exists():
            try:
                with open(h8_test_csv, "r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        img = r.get("image", "").strip()
                        src_tiff = r.get("source_tiff", "H11833_1of2.tif").strip()
                        px = _safe_int(r.get("pixel_x"))
                        py = _safe_int(r.get("pixel_y"))
                        data = {
                            "crop_id": Path(img).stem,
                            "image_filename": img,
                            "source_tiff": src_tiff,
                            "crop_x": px,
                            "crop_y": py,
                            "crop_width": 512,
                            "crop_height": 512,
                        }
                        if img:
                            manifest_map[img.lower()] = data
                            manifest_map[Path(img).stem.lower()] = data
            except Exception:
                pass

        # 5. image_geolocation.csv
        geo_csv = base / "metadata" / "image_geolocation.csv"
        if geo_csv.exists():
            try:
                with open(geo_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fn = row.get("image_filename", "").strip()
                        pid = row.get("patch_id", "").strip()
                        if fn.lower() not in manifest_map:
                            data = {
                                "crop_id": pid,
                                "image_filename": fn,
                                "source_tiff": row.get("source_tiff", "H11833_1of2.tif"),
                                "utm_x": _safe_float(row.get("utm_x")),
                                "utm_y": _safe_float(row.get("utm_y")),
                                "target_latitude": _safe_float(row.get("latitude")),
                                "target_longitude": _safe_float(row.get("longitude")),
                            }
                            if fn:
                                manifest_map[fn.lower()] = data
                                manifest_map[Path(fn).stem.lower()] = data
                            if pid:
                                manifest_map[pid.lower()] = data
            except Exception:
                pass

        # 6. H8 Unseen Test manifest (maps UNSEEN_xxxx -> original_stem in manifest_map)
        h8_unseen_csv = base / "h8_unseen_test" / "test_manifest.csv"
        if h8_unseen_csv.exists():
            try:
                with open(h8_unseen_csv, "r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        img = r.get("image", "").strip()
                        orig_stem = r.get("original_stem", "").strip()
                        if orig_stem and orig_stem.lower() in manifest_map:
                            orig_data = manifest_map[orig_stem.lower()].copy()
                            orig_data["alias_of"] = orig_stem
                            orig_data["image_filename"] = img
                            manifest_map[img.lower()] = orig_data
                            manifest_map[Path(img).stem.lower()] = orig_data
            except Exception:
                pass

        break

    _NOAA_PATCH_CACHE = manifest_map
    return _NOAA_PATCH_CACHE

def dms_to_decimal(dms_val: Any, ref: Optional[str] = None) -> Optional[float]:
    """
    Converts various EXIF DMS formats (tuples of IFDRational, floats, ints, or decimal degrees)
    into WGS84 decimal degrees.
    """
    if dms_val is None:
        return None

    try:
        if isinstance(dms_val, (int, float)):
            dec = float(dms_val)
        elif isinstance(dms_val, (tuple, list)):
            if len(dms_val) == 1:
                dec = float(dms_val[0])
            elif len(dms_val) == 2:
                dec = float(dms_val[0]) + float(dms_val[1]) / 60.0
            elif len(dms_val) >= 3:
                d = float(dms_val[0])
                m = float(dms_val[1])
                s = float(dms_val[2])
                dec = d + m / 60.0 + s / 3600.0
            else:
                return None
        else:
            return None

        if ref:
            ref_str = str(ref).strip().upper()
            if ref_str in ["S", "W"] and dec > 0:
                dec = -dec
            elif ref_str in ["N", "E"] and dec < 0:
                dec = -dec

        return dec
    except Exception:
        return None

def extract_exif_gps(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts camera/image EXIF GPS coordinates (latitude, longitude, altitude, capture direction).
    Distinguishes camera coordinates from object-level detection coordinates.
    """
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None

            # Look up GPS IFD (tag 0x8825 / 34853)
            gps_info = exif.get_ifd(0x8825)
            if not gps_info and hasattr(img, "_getexif"):
                raw_exif = img._getexif()
                if raw_exif:
                    gps_info = raw_exif.get(34853) or raw_exif.get(0x8825)

            if not gps_info:
                return None

            gps_data: Dict[str, Any] = {}
            for tag_id, val in gps_info.items():
                tag_name = GPSTAGS.get(tag_id, str(tag_id))
                gps_data[tag_name] = val
                gps_data[tag_id] = val

            lat_ref = gps_data.get("GPSLatitudeRef") or gps_data.get(1, "N")
            lat_raw = gps_data.get("GPSLatitude") or gps_data.get(2)
            lon_ref = gps_data.get("GPSLongitudeRef") or gps_data.get(3, "W")
            lon_raw = gps_data.get("GPSLongitude") or gps_data.get(4)

            alt_raw = gps_data.get("GPSAltitude") or gps_data.get(6)
            alt_ref = gps_data.get("GPSAltitudeRef") or gps_data.get(5, 0)
            dir_raw = gps_data.get("GPSImgDirection") or gps_data.get(17)

            lat = dms_to_decimal(lat_raw, lat_ref)
            lon = dms_to_decimal(lon_raw, lon_ref)

            if lat is not None and lon is not None:
                altitude = None
                if alt_raw is not None:
                    try:
                        altitude = float(alt_raw)
                        if alt_ref == 1:
                            altitude = -altitude
                    except Exception:
                        pass

                direction = None
                if dir_raw is not None:
                    try:
                        direction = float(dir_raw)
                    except Exception:
                        pass

                return {
                    "camera_latitude": round(lat, 7),
                    "camera_longitude": round(lon, 7),
                    "camera_altitude": round(altitude, 2) if altitude is not None else None,
                    "capture_direction": round(direction, 2) if direction is not None else None,
                    "crs": "EPSG:4326",
                    "source": "EXIF GPS"
                }
    except Exception:
        pass
    return None

def find_sidecar_file(image_path: Path, orig_name: Optional[str], extensions: List[str]) -> Optional[Path]:
    """
    Looks for matching sidecar files in the same directory using current path and original name.
    """
    search_dirs = [image_path.parent]
    stems = [image_path.stem]
    if orig_name:
        stems.append(Path(orig_name).stem)

    for d in search_dirs:
        for s in stems:
            for ext in extensions:
                candidate = d / f"{s}{ext}"
                if candidate.exists():
                    return candidate
                # Also check uppercase extension
                candidate_upper = d / f"{s}{ext.upper()}"
                if candidate_upper.exists():
                    return candidate_upper
    return None

def parse_world_file(world_file_path: Path) -> Optional[Affine]:
    """
    Parses standard 6-line ESRI world files (.tfw, .jgw, .pgw, .wld) into rasterio Affine transform.
    Line 1: A (pixel size in x direction)
    Line 2: D (rotation about y axis)
    Line 3: B (rotation about x axis)
    Line 4: E (pixel size in y direction)
    Line 5: C (x coordinate of upper-left pixel center)
    Line 6: F (y coordinate of upper-left pixel center)
    """
    try:
        lines = [line.strip() for line in world_file_path.read_text().splitlines() if line.strip()]
        if len(lines) >= 6:
            a = float(lines[0])
            d = float(lines[1])
            b = float(lines[2])
            e = float(lines[3])
            c_center = float(lines[4])
            f_center = float(lines[5])
            # Corner (0,0) offset from pixel center (0.5, 0.5):
            c_corner = c_center - (a / 2.0) - (b / 2.0)
            f_corner = f_center - (d / 2.0) - (e / 2.0)
            return Affine(a, b, c_corner, d, e, f_corner)
    except Exception:
        pass
    return None

def parse_prj_file(prj_file_path: Path) -> Optional[CRS]:
    """
    Parses .prj projection file into rasterio CRS.
    """
    try:
        text = prj_file_path.read_text().strip()
        if text:
            return CRS.from_user_input(text)
    except Exception:
        pass
    return None

def parse_aux_xml(xml_path: Path) -> Tuple[Optional[Affine], Optional[CRS]]:
    """
    Parses GDAL PAM XML auxiliary file (.aux.xml) for GeoTransform and SRS.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        transform = None
        crs = None

        gt_elem = root.find(".//GeoTransform")
        if gt_elem is not None and gt_elem.text:
            parts = [float(x.strip()) for x in gt_elem.text.split(",")]
            if len(parts) >= 6:
                # GDAL GeoTransform: c, a, b, f, d, e
                transform = Affine(parts[1], parts[2], parts[0], parts[4], parts[5], parts[3])

        srs_elem = root.find(".//SRS")
        if srs_elem is not None and srs_elem.text:
            crs = CRS.from_user_input(srs_elem.text)

        return transform, crs
    except Exception:
        return None, None

def build_wgs84_bounds_and_footprint(
    transform: Affine,
    width: int,
    height: int,
    crs: CRS
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    """
    Calculates exact WGS84 bounding box and GeoJSON polygon footprint from Affine transform & CRS.
    """
    # 4 image corners in continuous pixel space: (col, row)
    corners_pixel = [
        (0.0, 0.0),            # Top-Left
        (float(width), 0.0),   # Top-Right
        (float(width), float(height)), # Bottom-Right
        (0.0, float(height)),  # Bottom-Left
    ]

    proj_corners = [transform * (col, row) for col, row in corners_pixel]

    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        wgs84_corners = [transformer.transform(x, y) for x, y in proj_corners] # [(lon, lat), ...]
    except Exception:
        # If CRS is already geographic (EPSG:4326) or conversion fails
        wgs84_corners = [(x, y) for x, y in proj_corners]

    lons = [c[0] for c in wgs84_corners]
    lats = [c[1] for c in wgs84_corners]

    proj_xs = [c[0] for c in proj_corners]
    proj_ys = [c[1] for c in proj_corners]

    bounds_dict = {
        "left": round(min(proj_xs), 3),
        "bottom": round(min(proj_ys), 3),
        "right": round(max(proj_xs), 3),
        "top": round(max(proj_ys), 3),
        "wgs84_min_lon": round(min(lons), 7),
        "wgs84_min_lat": round(min(lats), 7),
        "wgs84_max_lon": round(max(lons), 7),
        "wgs84_max_lat": round(max(lats), 7),
    }

    footprint_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [round(wgs84_corners[0][0], 7), round(wgs84_corners[0][1], 7)],
            [round(wgs84_corners[1][0], 7), round(wgs84_corners[1][1], 7)],
            [round(wgs84_corners[2][0], 7), round(wgs84_corners[2][1], 7)],
            [round(wgs84_corners[3][0], 7), round(wgs84_corners[3][1], 7)],
            [round(wgs84_corners[0][0], 7), round(wgs84_corners[0][1], 7)],
        ]]
    }

    return bounds_dict, footprint_geojson

def extract_geospatial_metadata(
    image_path: str,
    orig_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts geospatial metadata using the authoritative source hierarchy:
    1. Georeferenced GeoTIFF (rasterio/GDAL CRS, affine transform, ModelPixelScale/Tiepoints)
    2. World file sidecars (.tfw, .jgw, .pgw, .wld) with companion .prj
    3. Auxiliary PAM XML / PRJ georeferencing files (.aux.xml, .prj)
    4. NOAA survey crop parent raster reconstruction
    5. EXIF GPS camera positioning tags (Camera location only)
    6. Non-georeferenced fallback (explicitly reported without fake coordinates)
    """
    path_obj = Path(image_path)
    ext = path_obj.suffix.lower()
    search_name = orig_filename or path_obj.name

    # =========================================================================
    # SOURCE 1: Georeferenced GeoTIFF (.tif / .tiff)
    # =========================================================================
    if ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(str(path_obj)) as src:
                crs = src.crs
                transform = src.transform
                width, height = src.width, src.height

                # If CRS missing, search for sidecar .prj
                if crs is None:
                    prj_path = find_sidecar_file(path_obj, orig_filename, [".prj"])
                    if prj_path:
                        crs = parse_prj_file(prj_path)

                # If transform is identity, search for sidecar world file or GCPs
                if (transform is None or transform.is_identity) and src.gcps and len(src.gcps[0]) >= 3:
                    try:
                        transform = rasterio.transform.from_gcps(src.gcps[0])
                        if crs is None and src.gcps[1]:
                            crs = src.gcps[1]
                    except Exception:
                        pass

                if transform is None or transform.is_identity:
                    tfw_path = find_sidecar_file(path_obj, orig_filename, [".tfw", ".tifw", ".wld"])
                    if tfw_path:
                        transform = parse_world_file(tfw_path)

                if transform is not None and not transform.is_identity:
                    # If CRS is still None, check if coordinates are in degree range [-180..180, -90..90]
                    if crs is None:
                        c_x, c_y = transform * (width / 2.0, height / 2.0)
                        if -180.0 <= c_x <= 180.0 and -90.0 <= c_y <= 90.0:
                            crs = CRS.from_epsg(4326)
                        else:
                            crs = CRS.from_epsg(26916) # Default UTM 16N for NOAA survey context

                    crs_str = get_crs_name(crs)
                    bounds_dict, footprint = build_wgs84_bounds_and_footprint(transform, width, height, crs)
                    res_x = abs(transform.a)
                    res_y = abs(transform.e)

                    return {
                        "georeferenced": True,
                        "crs": crs_str,
                        "raw_crs": crs,
                        "transform": transform,
                        "bounds": bounds_dict,
                        "pixel_resolution": [round(res_x, 6), round(res_y, 6)],
                        "lat_lon_available": True,
                        "coordinate_source": "GeoTIFF Affine Transform",
                        "status_message": f"Georeferenced GeoTIFF detected ({crs_str})",
                        "camera_latitude": None,
                        "camera_longitude": None,
                        "camera_altitude": None,
                        "capture_direction": None,
                        "footprint_geojson": footprint
                    }
        except Exception:
            pass

    # =========================================================================
    # SOURCE 2: World File Sidecars (.tfw, .jgw, .pgw, .wld)
    # =========================================================================
    world_exts = [".wld"]
    if ext in [".tif", ".tiff"]:
        world_exts.extend([".tfw", ".tifw"])
    elif ext in [".jpg", ".jpeg"]:
        world_exts.extend([".jgw", ".jpegw", ".jpgw"])
    elif ext in [".png"]:
        world_exts.extend([".pgw", ".pngw"])

    world_file = find_sidecar_file(path_obj, orig_filename, world_exts)
    if world_file:
        transform = parse_world_file(world_file)
        if transform is not None:
            # Check PRJ
            prj_file = find_sidecar_file(path_obj, orig_filename, [".prj"])
            crs = parse_prj_file(prj_file) if prj_file else None

            # Read image dimensions
            try:
                with Image.open(str(path_obj)) as im:
                    w, h = im.size
            except Exception:
                w, h = 512, 512

            if crs is None:
                c_x, c_y = transform * (w / 2.0, h / 2.0)
                if -180.0 <= c_x <= 180.0 and -90.0 <= c_y <= 90.0:
                    crs = CRS.from_epsg(4326)
                else:
                    crs = CRS.from_epsg(26916)

            crs_str = get_crs_name(crs)
            bounds_dict, footprint = build_wgs84_bounds_and_footprint(transform, w, h, crs)

            return {
                "georeferenced": True,
                "crs": crs_str,
                "raw_crs": crs,
                "transform": transform,
                "bounds": bounds_dict,
                "pixel_resolution": [round(abs(transform.a), 6), round(abs(transform.e), 6)],
                "lat_lon_available": True,
                "coordinate_source": f"World File Affine Transform ({world_file.suffix})",
                "status_message": f"World file georeferencing detected ({world_file.name}, {crs_str})",
                "camera_latitude": None,
                "camera_longitude": None,
                "camera_altitude": None,
                "capture_direction": None,
                "footprint_geojson": footprint
            }

    # =========================================================================
    # SOURCE 3: Auxiliary Georeferencing Files (.aux.xml, .PAM.xml)
    # =========================================================================
    aux_xml = find_sidecar_file(path_obj, orig_filename, [".aux.xml", ".PAM.xml"])
    if aux_xml:
        transform, crs = parse_aux_xml(aux_xml)
        if transform is not None:
            try:
                with Image.open(str(path_obj)) as im:
                    w, h = im.size
            except Exception:
                w, h = 512, 512

            if crs is None:
                crs = CRS.from_epsg(4326)

            crs_str = get_crs_name(crs)
            bounds_dict, footprint = build_wgs84_bounds_and_footprint(transform, w, h, crs)

            return {
                "georeferenced": True,
                "crs": crs_str,
                "raw_crs": crs,
                "transform": transform,
                "bounds": bounds_dict,
                "pixel_resolution": [round(abs(transform.a), 6), round(abs(transform.e), 6)],
                "lat_lon_available": True,
                "coordinate_source": "GDAL PAM XML Auxiliary Georeferencing",
                "status_message": f"Auxiliary XML georeferencing detected ({aux_xml.name}, {crs_str})",
                "camera_latitude": None,
                "camera_longitude": None,
                "camera_altitude": None,
                "capture_direction": None,
                "footprint_geojson": footprint
            }

    # =========================================================================
    # SOURCE 4: Derived NOAA Survey Crop Reconstruction
    # =========================================================================
    noaa_patches = _load_noaa_patch_manifest()
    lookup_keys = [
        search_name.lower(),
        Path(search_name).stem.lower(),
        path_obj.name.lower(),
        path_obj.stem.lower()
    ]
    # Also test for patch ID regex like H11833_000001
    match_pid = re.search(r'(H11833_\d+)', search_name, re.IGNORECASE)
    if match_pid:
        lookup_keys.append(match_pid.group(1).lower())

    patch_entry = None
    for k in lookup_keys:
        if k in noaa_patches:
            patch_entry = noaa_patches[k]
            break

    if patch_entry:
        try:
            with Image.open(str(path_obj)) as im:
                w, h = im.size
        except Exception:
            w, h = 512, 512

        source_tiff = patch_entry.get("source_tiff", "H11833_1of2.tif")
        parent_meta = NOAA_PARENT_TIFF_METADATA.get(source_tiff, NOAA_PARENT_TIFF_METADATA["H11833_1of2.tif"])
        parent_trans = parent_meta["transform"]
        crs = parent_meta["raw_crs"]
        crs_str = parent_meta["crs"]

        crop_x = patch_entry.get("crop_x")
        crop_y = patch_entry.get("crop_y")
        crop_w = patch_entry.get("crop_width") or w
        crop_h = patch_entry.get("crop_height") or h

        if crop_x is not None and crop_y is not None:
            # Exact parent raster window transform
            win = Window(crop_x, crop_y, crop_w, crop_h)
            win_transform = rasterio.windows.transform(win, parent_trans)
            if crop_w != w or crop_h != h:
                scale_x = crop_w / w
                scale_y = crop_h / h
                transform = Affine(
                    win_transform.a * scale_x,
                    win_transform.b,
                    win_transform.c,
                    win_transform.d,
                    win_transform.e * scale_y,
                    win_transform.f
                )
            else:
                transform = win_transform
        elif patch_entry.get("utm_x") is not None and patch_entry.get("utm_y") is not None:
            cx_utm = patch_entry["utm_x"]
            cy_utm = patch_entry["utm_y"]
            res_x, res_y = 0.5, 0.5
            left_utm = cx_utm - (w / 2.0) * res_x
            top_utm = cy_utm + (h / 2.0) * res_y
            transform = Affine(res_x, 0.0, left_utm, 0.0, -res_y, top_utm)
        else:
            transform = None

        if transform is not None:
            bounds_dict, footprint = build_wgs84_bounds_and_footprint(transform, w, h, crs)
            res_x = abs(transform.a)
            res_y = abs(transform.e)

            return {
                "georeferenced": True,
                "crs": crs_str,
                "raw_crs": crs,
                "transform": transform,
                "bounds": bounds_dict,
                "pixel_resolution": [round(res_x, 6), round(res_y, 6)],
                "lat_lon_available": True,
                "coordinate_source": f"NOAA Survey Parent Raster Reconstruction ({source_tiff})",
                "status_message": f"NOAA Survey Crop reconstructed from parent raster ({crs_str})",
                "camera_latitude": None,
                "camera_longitude": None,
                "camera_altitude": None,
                "capture_direction": None,
                "footprint_geojson": footprint
            }

    # =========================================================================
    # SOURCE 5: EXIF GPS (Camera positioning only)
    # =========================================================================
    exif_gps = extract_exif_gps(str(path_obj))
    if exif_gps:
        cam_lat = exif_gps["camera_latitude"]
        cam_lon = exif_gps["camera_longitude"]

        return {
            "georeferenced": True,
            "crs": exif_gps["crs"],
            "raw_crs": CRS.from_epsg(4326),
            "transform": None,
            "bounds": None,
            "pixel_resolution": None,
            "lat_lon_available": True,
            "coordinate_source": "EXIF GPS",
            "status_message": "EXIF GPS Camera Positioning tags detected (Camera location only)",
            "camera_latitude": cam_lat,
            "camera_longitude": cam_lon,
            "camera_altitude": exif_gps.get("camera_altitude"),
            "capture_direction": exif_gps.get("capture_direction"),
            "footprint_geojson": {
                "type": "Point",
                "coordinates": [cam_lon, cam_lat]
            }
        }

    # =========================================================================
    # SOURCE 6: Non-georeferenced
    # =========================================================================
    return {
        "georeferenced": False,
        "crs": None,
        "raw_crs": None,
        "transform": None,
        "bounds": None,
        "pixel_resolution": None,
        "lat_lon_available": False,
        "coordinate_source": None,
        "status_message": "No valid GeoTIFF transform, CRS, EXIF GPS, or world-file information was found.",
        "camera_latitude": None,
        "camera_longitude": None,
        "camera_altitude": None,
        "capture_direction": None,
        "footprint_geojson": None
    }

