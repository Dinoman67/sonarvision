#!/usr/bin/env python3
"""
scripts/shadow_analysis.py

Extract and measure shadow properties for all 17 verified debris targets AND
background bright spots (potential false positives) from NOAA H11833 SSS GeoTIFFs.

SSS shadow physics:
- Debris rises above seafloor → strong acoustic return (bright) + acoustic shadow (dark)
- Shadow is cast in the across-track direction (away from sonar nadir)
- Shadow length ∝ object height; shadow darkness ∝ object height + grazing angle
- Natural seabed variations can produce bright returns but typically lack the
  structured bright+shadow pair characteristic of elevated debris

Outputs:
- shadow_report.json: per-target and per-background-spot shadow measurements
- shadow_comparison.csv: tabular summary for analysis
- visual comparisons saved to results/shadow_analysis/
"""

import os
import sys
import json
import csv
import re
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image, ImageDraw, ImageFont

# ── Verified targets (same as prepare_noaa_e3.py) ───────────────────────────
VERIFIED_TARGETS = [
    {"target_id": "TGT001", "name": "DtoN_1_Pipeline_Exposed",
     "lat_str": "28° 59' 03.3504\" N", "lon_str": "089° 18' 09.3672\" W",
     "length_m": 12.0, "width_m": 6.0, "height_m": 2.1, "cluster": "NE_Cluster"},
    {"target_id": "TGT002", "name": "DtoN_2_Obstruction_Debris",
     "lat_str": "28° 58' 58.6344\" N", "lon_str": "089° 18' 07.3908\" W",
     "length_m": 14.0, "width_m": 10.0, "height_m": 1.4, "cluster": "NE_Cluster"},
    {"target_id": "TGT003", "name": "DtoN_5_Damaged_Wellhead_Ruins",
     "lat_str": "29° 01' 35.0112\" N", "lon_str": "089° 17' 25.5372\" W",
     "length_m": 12.0, "width_m": 12.0, "height_m": None, "cluster": "NE_Cluster"},
    {"target_id": "TGT004", "name": "DtoN_3_1_Elevated_Pipeline",
     "lat_str": "28° 54' 58.5288\" N", "lon_str": "089° 21' 56.8800\" W",
     "length_m": 16.0, "width_m": 6.0, "height_m": None, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT005", "name": "DtoN_3_2_Elevated_Pipeline",
     "lat_str": "28° 54' 48.1320\" N", "lon_str": "089° 22' 05.1348\" W",
     "length_m": 16.0, "width_m": 6.0, "height_m": None, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT006", "name": "Rig_Caisson_Pipeline_Feature",
     "lat_str": "28° 54' 54.197\" N", "lon_str": "089° 22' 16.974\" W",
     "length_m": 16.0, "width_m": 8.0, "height_m": None, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT007", "name": "DtoN_3_3_Obstruction_Debris",
     "lat_str": "28° 54' 10.6380\" N", "lon_str": "089° 25' 30.9252\" W",
     "length_m": 14.0, "width_m": 8.0, "height_m": 2.1, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT008", "name": "DtoN_4_Ruined_Training_Wall",
     "lat_str": "28° 54' 42.4764\" N", "lon_str": "089° 25' 24.0312\" W",
     "length_m": 16.0, "width_m": 10.0, "height_m": 2.7, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT009", "name": "Training_Wall_Ruins_Center",
     "lat_str": "28° 54' 41.62\" N", "lon_str": "089° 25' 22.30\" W",
     "length_m": 22.0, "width_m": 10.0, "height_m": None, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT010", "name": "Obstruction_22ft",
     "lat_str": "28° 54' 38.79\" N", "lon_str": "089° 25' 17.40\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 6.7, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT011", "name": "Obstruction_17ft",
     "lat_str": "28° 54' 59.10\" N", "lon_str": "089° 25' 34.45\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 5.2, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT012", "name": "Obstruction_18ft",
     "lat_str": "28° 54' 59.78\" N", "lon_str": "089° 25' 32.53\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 5.5, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT013", "name": "Obstruction_14ft",
     "lat_str": "28° 54' 32.396\" N", "lon_str": "089° 25' 59.387\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 4.3, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT014", "name": "Obstruction_10ft",
     "lat_str": "28° 54' 30.07\" N", "lon_str": "089° 26' 03.29\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 3.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT015", "name": "Obstruction_Charted_Retained",
     "lat_str": "28° 54' 18.14\" N", "lon_str": "089° 25' 45.09\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": None, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT016", "name": "Pipe_PA_Debris",
     "lat_str": "28° 53' 35.68\" N", "lon_str": "089° 25' 44.09\" W",
     "length_m": 14.0, "width_m": 6.0, "height_m": None, "cluster": "SW_Pass_South_Cluster"},
    {"target_id": "TGT017", "name": "AWOIS_11622_Wreck_Debris",
     "lat_str": "28° 53' 19.464\" N", "lon_str": "089° 26' 14.276\" W",
     "length_m": 12.0, "width_m": 10.0, "height_m": None, "cluster": "SW_Pass_South_Cluster"},
]

TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]

# Search radius around target center to find shadow/bright regions (pixels)
SEARCH_RADIUS = 200  # ~100m at 0.5m/px
# Neighborhood radius for local statistics
LOCAL_RADIUS = 50  # ~25m

PIXEL_SIZE_M = 0.5  # meters per pixel


def dms_to_dd(dms_str: str) -> float:
    parts = re.split(r'[°\'\"/\-\s]+', dms_str.strip())
    parts = [p for p in parts if p]
    d = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    s = float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + s / 3600.0
    if 'S' in dms_str.upper() or 'W' in dms_str.upper() or dms_str.startswith('-'):
        dd = -dd
    return dd


def safe_read(ds, col, row, radius, crop_size=None):
    """Read a region around (col, row), clamping to raster bounds. Returns (data, offset)."""
    c, r = int(round(col)), int(round(row))
    if crop_size:
        half = crop_size // 2
        x1, y1 = c - half, r - half
        x2, y2 = c + half, r + half
    else:
        x1, y1 = c - radius, r - radius
        x2, y2 = c + radius, r + radius

    # Clamp to raster bounds
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(ds.width, x2)
    y2c = min(ds.height, y2)

    if x1c >= x2c or y1c >= y2c:
        return None, None, (x1, y1)

    data = ds.read(1, window=Window(x1c, y1c, x2c - x1c, y2c - y1c))
    return data, (x1c, y1c), (x1, y1)


def local_background(data, center_local, radius=LOCAL_RADIUS):
    """Compute local background mean/std in a ring around center (excluding center region)."""
    h, w = data.shape
    cx, cy = int(round(center_local[0])), int(round(center_local[1]))
    # Create ring mask: exclude center ± radius/3, include radius/2 to radius
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    ring_mask = (dist >= radius * 0.5) & (dist <= radius * 1.5)
    # Also exclude zero/nodata pixels
    valid = (data > 0) & ring_mask
    if valid.sum() < 20:
        return None, None
    return float(data[valid].mean()), float(data[valid].std())


def find_shadow_region(data, bright_center_local, scan_direction='across_track'):
    """
    Search for shadow (dark region) near a bright return.
    SSS shadows appear adjacent to bright returns in the across-track direction.
    We search in a cone behind the bright spot.
    """
    h, w = data.shape
    cx, cy = bright_center_local

    best_shadow_score = -1
    best_shadow_center = None
    best_shadow_mask = None

    # Search in angular sectors around the bright return
    for angle_deg in range(0, 360, 15):
        angle_rad = np.radians(angle_deg)
        # Check a ray from bright center outward
        for dist in range(20, min(SEARCH_RADIUS, h // 2, w // 2), 5):
            sx = int(cx + dist * np.cos(angle_rad))
            sy = int(cy + dist * np.sin(angle_rad))
            if sx < 5 or sy < 5 or sx >= w - 5 or sy >= h - 5:
                continue

            # Sample a small region at this position
            patch_r = 8
            y1 = max(0, sy - patch_r)
            y2 = min(h, sy + patch_r)
            x1 = max(0, sx - patch_r)
            x2 = min(w, sx + patch_r)
            patch = data[y1:y2, x1:x2]
            valid_patch = patch[patch > 0]
            if len(valid_patch) < 10:
                continue

            patch_mean = float(valid_patch.mean())
            local_bg, _ = local_background(data, (sx, sy), LOCAL_RADIUS)
            if local_bg is None or local_bg < 5:
                continue

            # Shadow score: how much darker than local background
            shadow_depth = (local_bg - patch_mean) / max(local_bg, 1)

            # Shadow must be substantially darker than background
            if shadow_depth > 0.15 and patch_mean < local_bg * 0.8:
                score = shadow_depth * dist  # prefer further, darker shadows
                if score > best_shadow_score:
                    best_shadow_score = score
                    best_shadow_center = (sx, sy)
                    # Create shadow mask around this point
                    best_shadow_mask = (data[y1:y2, x1:x2] > 0) & (data[y1:y2, x1:x2] < local_bg * 0.7)

    return best_shadow_center, best_shadow_score


def measure_shadow_properties(data, bright_local, shadow_local, local_bg_mean):
    """Measure detailed shadow properties relative to bright return."""
    if shadow_local is None or local_bg_mean is None or local_bg_mean < 5:
        return {}

    h, w = data.shape
    bx, by = bright_local
    sx, sy = shadow_local

    # Shadow vector from bright to shadow center
    shadow_vec_x = sx - bx
    shadow_vec_y = sy - by
    shadow_dist_px = np.sqrt(shadow_vec_x**2 + shadow_vec_y**2)
    shadow_dist_m = shadow_dist_px * PIXEL_SIZE_M
    shadow_angle = np.degrees(np.arctan2(shadow_vec_y, shadow_vec_x))

    # Measure shadow region properties
    patch_r = 15
    sy_i, sx_i = int(round(sy)), int(round(sx))
    by_i, bx_i = int(round(by)), int(round(bx))

    y1 = max(0, sy_i - patch_r)
    y2 = min(h, sy_i + patch_r)
    x1 = max(0, sx_i - patch_r)
    x2 = min(w, sx_i + patch_r)
    shadow_patch = data[y1:y2, x1:x2]
    valid_shadow = shadow_patch[(shadow_patch > 0)]

    if len(valid_shadow) == 0:
        return {}

    shadow_mean = float(valid_shadow.mean())
    shadow_min = float(valid_shadow.min())
    shadow_depth = (local_bg_mean - shadow_mean) / max(local_bg_mean, 1)

    # Measure bright region properties
    y1b = max(0, by_i - patch_r)
    y2b = min(h, by_i + patch_r)
    x1b = max(0, bx_i - patch_r)
    x2b = min(w, bx_i + patch_r)
    bright_patch = data[y1b:y2b, x1b:x2b]
    valid_bright = bright_patch[(bright_patch > 0)]

    bright_mean = float(valid_bright.mean()) if len(valid_bright) > 0 else 0
    bright_max = float(valid_bright.max()) if len(valid_bright) > 0 else 0

    return {
        "shadow_dist_px": round(shadow_dist_px, 1),
        "shadow_dist_m": round(shadow_dist_m, 1),
        "shadow_angle_deg": round(shadow_angle, 1),
        "shadow_mean_intensity": round(shadow_mean, 1),
        "shadow_min_intensity": round(shadow_min, 1),
        "shadow_depth": round(shadow_depth, 4),  # (bg - shadow) / bg
        "bright_mean_intensity": round(bright_mean, 1),
        "bright_max_intensity": round(bright_max, 1),
        "bright_shadow_contrast": round(bright_mean - shadow_mean, 1),
        "local_bg_mean": round(local_bg_mean, 1),
        "contrast_ratio": round(bright_mean / max(shadow_mean, 1), 2),
    }


def find_bright_spots_in_region(data, local_bg_mean, local_bg_std, num_spots=5):
    """Find top bright spots in a data region that could be potential false positives."""
    if local_bg_mean is None or local_bg_std is None or local_bg_std < 2:
        return []

    # Threshold for "bright" = mean + 2*std
    bright_thresh = local_bg_mean + 2.0 * local_bg_std
    bright_mask = (data > bright_thresh) & (data > 30)  # also must be absolute bright

    if bright_mask.sum() < 5:
        return []

    # Label connected components (simple approach: use local maxima)
    from scipy import ndimage
    # Dilate to merge nearby bright pixels
    dilated = ndimage.binary_dilation(bright_mask, iterations=3)
    labeled, num_features = ndimage.label(dilated)

    spots = []
    for i in range(1, num_features + 1):
        component = labeled == i
        # Find centroid
        ys, xs = np.where(component & bright_mask)
        if len(xs) < 3:
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        # Mean brightness in the component
        comp_data = data[component]
        valid_comp = comp_data[comp_data > 0]
        if len(valid_comp) == 0:
            continue
        mean_bright = float(valid_comp.mean())
        spots.append({
            "cx": cx, "cy": cy,
            "mean_intensity": round(mean_bright, 1),
            "area_px": int(component.sum()),
            "excess_brightness": round((mean_bright - local_bg_mean) / max(local_bg_std, 1), 2),
        })

    # Sort by excess brightness, return top N
    spots.sort(key=lambda s: s["excess_brightness"], reverse=True)
    return spots[:num_spots]


def process_target(t, ds_list, transformer_utm, out_dir):
    """Process a single target: extract shadow measurements from both TIFFs."""
    lat = dms_to_dd(t["lat_str"])
    lon = dms_to_dd(t["lon_str"])
    if lon > 0:
        lon = -lon
    utm_x, utm_y = transformer_utm.transform(lon, lat)

    bw_px = t["length_m"] / PIXEL_SIZE_M
    bh_px = t["width_m"] / PIXEL_SIZE_M

    results = {
        "target_id": t["target_id"],
        "name": t["name"],
        "cluster": t["cluster"],
        "height_m": t["height_m"],
        "bbox_length_m": t["length_m"],
        "bbox_width_m": t["width_m"],
        "measurements": [],
    }

    for tiff_name, ds in ds_list:
        inv = ~ds.transform
        tgt_col, tgt_row = inv * (utm_x, utm_y)

        # Read extended region around target
        data, data_offset, _ = safe_read(ds, tgt_col, tgt_row, SEARCH_RADIUS)
        if data is None:
            continue

        # Local coordinates of target in the data window
        local_c = tgt_col - data_offset[0]
        local_r = tgt_row - data_offset[1]

        # Check validity
        valid_mask = data > 0
        if valid_mask.sum() < 100:
            continue

        local_bg_mean, local_bg_std = local_background(data, (local_c, local_r), LOCAL_RADIUS)
        if local_bg_mean is None:
            continue

        # Find the brightest region near target (should be the debris return)
        # Search within the target bounding box area
        x1_bb = max(0, int(local_c - bw_px))
        x2_bb = min(data.shape[1], int(local_c + bw_px))
        y1_bb = max(0, int(local_r - bh_px))
        y2_bb = min(data.shape[0], int(local_r + bh_px))

        target_region = data[y1_bb:y2_bb, x1_bb:x2_bb]
        valid_target = target_region[target_region > 0]

        bright_spot = None
        shadow_result = None
        shadow_props = {}

        if len(valid_target) > 0:
            # Find local maximum in target region (brightest point = debris return)
            bright_mask = target_region == target_region[target_region > 0].max() if valid_target.sum() > 0 else np.zeros_like(target_region, dtype=bool)
            # Use a more robust approach: find center of mass of top 10% brightest pixels
            thresh_val = np.percentile(valid_target, 90)
            top_bright = target_region >= thresh_val
            if top_bright.sum() > 0:
                by_local, bx_local = np.where(top_bright)
                bright_cx = float(np.mean(bx_local)) + x1_bb
                bright_cy = float(np.mean(by_local)) + y1_bb

                # Find shadow
                shadow_center, shadow_score = find_shadow_region(data, (bright_cx, bright_cy))

                if shadow_center is not None and shadow_score > 0:
                    shadow_props = measure_shadow_properties(
                        data, (bright_cx, bright_cy), shadow_center, local_bg_mean
                    )

                bright_spot = {
                    "cx": round(bright_cx + data_offset[0], 2),  # absolute raster coords
                    "cy": round(bright_cy + data_offset[1], 2),
                    "mean_intensity": round(float(valid_target.mean()), 1),
                    "max_intensity": round(float(valid_target.max()), 1),
                    "bright_excess": round(float(valid_target.mean() - local_bg_mean), 1),
                }

                # Also find shadow depth in the darkest area near target
                # Search in all directions from bright spot
                shadow_search_data = data.copy()
                # Mask the bright region to avoid it
                y1s = max(0, int(bright_cy) - 5)
                y2s = min(data.shape[0], int(bright_cy) + 5)
                x1s = max(0, int(bright_cx) - 5)
                x2s = min(data.shape[1], int(bright_cx) + 5)
                # Don't mask it, just search for darkest connected region

        # Extract a 512x512 visualization patch (if fits)
        vis_patch = None
        vis_size = 512
        wc = int(tgt_col - vis_size // 2)
        wr = int(tgt_row - vis_size // 2)
        wc = max(0, min(ds.width - vis_size, wc))
        wr = max(0, min(ds.height - vis_size, wr))

        if wc >= 0 and wr >= 0 and wc + vis_size <= ds.width and wr + vis_size <= ds.height:
            vis_data = ds.read(1, window=Window(wc, wr, vis_size, vis_size))
            if np.count_nonzero(vis_data) / vis_data.size > 0.3:
                vis_patch = vis_data
                # Save visualization
                vis_img = Image.fromarray(vis_data).convert("RGB")
                draw = ImageDraw.Draw(vis_img)

                # Draw target bounding box (original)
                tc_x = tgt_col - wc
                tc_y = tgt_row - wr
                tx1 = tc_x - bw_px / 2
                ty1 = tc_y - bh_px / 2
                tx2 = tc_x + bw_px / 2
                ty2 = tc_y + bh_px / 2
                draw.rectangle([tx1, ty1, tx2, ty2], outline="#00FF00", width=2)

                # Draw bright spot
                if bright_spot is not None:
                    bxs = bright_spot["cx"] - wc
                    bys = bright_spot["cy"] - wr
                    draw.ellipse([bxs - 5, bys - 5, bxs + 5, bys + 5], outline="#FF0000", width=2)

                # Draw shadow region
                if shadow_props:
                    # Draw shadow center marker
                    sx = shadow_props.get("shadow_dist_px", 0) * np.cos(np.radians(shadow_props.get("shadow_angle_deg", 0))) + tc_x
                    sy = shadow_props.get("shadow_dist_px", 0) * np.sin(np.radians(shadow_props.get("shadow_angle_deg", 0))) + tc_y
                    draw.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], outline="#4444FF", width=2)
                    # Draw line from bright to shadow
                    draw.line([(tc_x, tc_y), (sx, sy)], fill="#4444FF", width=1)

                # Banner
                banner = f"{t['target_id']} | shadow_depth={shadow_props.get('shadow_depth', 'N/A')} contrast={shadow_props.get('bright_shadow_contrast', 'N/A')}"
                draw.rectangle([0, 0, vis_size, 18], fill="#222222")
                draw.text((4, 2), banner, fill="#00FF00")

                vis_path = os.path.join(out_dir, f"shadow_{t['target_id']}_{tiff_name.replace('.tif', '')}.png")
                vis_img.save(vis_path, format="PNG")

        measurement = {
            "source_tiff": tiff_name,
            "target_pixel": (round(tgt_col, 2), round(tgt_row, 2)),
            "local_bg_mean": round(local_bg_mean, 1) if local_bg_mean else None,
            "local_bg_std": round(local_bg_std, 1) if local_bg_std else None,
            "bright_spot": bright_spot,
            "shadow": shadow_props,
        }
        results["measurements"].append(measurement)

    return results


def process_background_spots(ds_list, transformer_utm, out_dir, num_regions=8):
    """
    Find bright spots in background regions (far from all verified targets)
    that could serve as false positive comparisons.
    """
    # Define background regions far from any debris target
    # Each region is specified as (split, tiff_name, utm_x, utm_y, description)
    bg_regions = [
        ("train", "H11833_1of2.tif", 262500, 3199000, "SW_train_bg_1"),
        ("train", "H11833_1of2.tif", 264000, 3201000, "SW_train_bg_2"),
        ("train", "H11833_1of2.tif", 265000, 3198500, "SW_train_bg_3"),
        ("train", "H11833_2of2.tif", 263000, 3200500, "SW_train_bg_4"),
        ("train", "H11833_2of2.tif", 264500, 3199500, "SW_train_bg_5"),
        ("val", "H11833_1of2.tif", 276000, 3210000, "NE_val_bg_1"),
        ("val", "H11833_2of2.tif", 277000, 3209000, "NE_val_bg_2"),
        ("test", "H11833_1of2.tif", 269500, 3201000, "CE_test_bg_1"),
    ]

    bg_results = []

    # Collect verified target UTM coords for distance check
    verified_utm = []
    for t in VERIFIED_TARGETS:
        lat = dms_to_dd(t["lat_str"])
        lon = dms_to_dd(t["lon_str"])
        if lon > 0:
            lon = -lon
        ux, uy = transformer_utm.transform(lon, lat)
        verified_utm.append((ux, uy))

    for split, tiff_name, utm_x, utm_y, desc in bg_regions:
        ds = next((d for name, d in ds_list if name == tiff_name), None)
        if ds is None:
            continue

        # Check distance from all verified targets
        min_dist = min(np.hypot(utm_x - vx, utm_y - vy) for vx, vy in verified_utm)
        if min_dist < 500:
            continue

        inv = ~ds.transform
        col, row = inv * (utm_x, utm_y)

        data, data_offset, _ = safe_read(ds, col, row, SEARCH_RADIUS)
        if data is None:
            continue

        local_c = col - data_offset[0]
        local_r = row - data_offset[1]

        valid_mask = data > 0
        if valid_mask.sum() < 100:
            continue

        local_bg_mean, local_bg_std = local_background(data, (local_c, local_r), LOCAL_RADIUS)
        if local_bg_mean is None or local_bg_std is None:
            continue

        # Find bright spots
        bright_spots = find_bright_spots_in_region(data, local_bg_mean, local_bg_std, num_spots=3)

        for idx, spot in enumerate(bright_spots):
            # For each bright spot, try to find shadow
            shadow_center, shadow_score = find_shadow_region(data, (spot["cx"], spot["cy"]))

            shadow_props = {}
            if shadow_center is not None and shadow_score > 0:
                shadow_props = measure_shadow_properties(
                    data, (spot["cx"], spot["cy"]), shadow_center, local_bg_mean
                )

            bg_record = {
                "region": desc,
                "split": split,
                "source_tiff": tiff_name,
                "local_bg_mean": round(local_bg_mean, 1),
                "local_bg_std": round(local_bg_std, 1),
                "bright_spot": spot,
                "shadow": shadow_props,
                "has_structured_shadow": bool(shadow_props and shadow_props.get("shadow_depth", 0) > 0.15),
            }
            bg_results.append(bg_record)

            # Save visualization for top spots
            if idx == 0:
                vis_size = 512
                wc = int(col - vis_size // 2)
                wr = int(row - vis_size // 2)
                wc = max(0, min(ds.width - vis_size, wc))
                wr = max(0, min(ds.height - vis_size, wr))
                if wc >= 0 and wr >= 0 and wc + vis_size <= ds.width and wr + vis_size <= ds.height:
                    vis_data = ds.read(1, window=Window(wc, wr, vis_size, vis_size))
                    if np.count_nonzero(vis_data) / vis_data.size > 0.3:
                        vis_img = Image.fromarray(vis_data).convert("RGB")
                        draw = ImageDraw.Draw(vis_img)
                        # Draw bright spot
                        bxs = spot["cx"] - wc
                        bys = spot["cy"] - wr
                        draw.ellipse([bxs - 5, bys - 5, bxs + 5, bys + 5], outline="#FF0000", width=2)

                        # Draw shadow
                        if shadow_props and "shadow_angle_deg" in shadow_props:
                            sx = shadow_props["shadow_dist_px"] * np.cos(np.radians(shadow_props["shadow_angle_deg"])) + bxs
                            sy = shadow_props["shadow_dist_px"] * np.sin(np.radians(shadow_props["shadow_angle_deg"])) + bys
                            draw.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], outline="#4444FF", width=2)

                        banner = f"BG: {desc} | excess={spot['excess_brightness']:.1f}σ"
                        draw.rectangle([0, 0, vis_size, 18], fill="#222222")
                        draw.text((4, 2), banner, fill="#FF8800")

                        # Save vis
                        vis_path = os.path.join(out_dir, f"shadow_BG_{desc.replace(' ', '_')}.png")
                        vis_img.save(vis_path, format="PNG")

    return bg_results


def write_comparison_csv(target_results, bg_results, out_dir):
    """Write a comparison CSV with shadow metrics for debris vs background."""
    csv_path = os.path.join(out_dir, "shadow_comparison.csv")

    rows = []
    for t_res in target_results:
        for m in t_res["measurements"]:
            shadow = m.get("shadow", {})
            bright = m.get("bright_spot", {})
            rows.append({
                "category": "debris",
                "id": t_res["target_id"],
                "name": t_res["name"],
                "source_tiff": m["source_tiff"],
                "local_bg_mean": m.get("local_bg_mean"),
                "bright_mean": bright.get("mean_intensity") if bright else None,
                "bright_excess": bright.get("bright_excess") if bright else None,
                "shadow_depth": shadow.get("shadow_depth"),
                "shadow_dist_m": shadow.get("shadow_dist_m"),
                "shadow_angle_deg": shadow.get("shadow_angle_deg"),
                "bright_shadow_contrast": shadow.get("bright_shadow_contrast"),
                "contrast_ratio": shadow.get("contrast_ratio"),
                "height_m": t_res["height_m"],
            })

    for bg in bg_results:
        shadow = bg.get("shadow", {})
        bright = bg.get("bright_spot", {})
        rows.append({
            "category": "background",
            "id": bg["region"],
            "name": bg["region"],
            "source_tiff": bg["source_tiff"],
            "local_bg_mean": bg["local_bg_mean"],
            "bright_mean": bright.get("mean_intensity"),
            "bright_excess": bright.get("excess_brightness"),
            "shadow_depth": shadow.get("shadow_depth"),
            "shadow_dist_m": shadow.get("shadow_dist_m"),
            "shadow_angle_deg": shadow.get("shadow_angle_deg"),
            "bright_shadow_contrast": shadow.get("bright_shadow_contrast"),
            "contrast_ratio": shadow.get("contrast_ratio"),
            "height_m": None,
        })

    fieldnames = ["category", "id", "name", "source_tiff", "local_bg_mean",
                  "bright_mean", "bright_excess", "shadow_depth", "shadow_dist_m",
                  "shadow_angle_deg", "bright_shadow_contrast", "contrast_ratio", "height_m"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] Saved shadow_comparison.csv ({len(rows)} records)")
    return csv_path


def print_summary(target_results, bg_results):
    """Print a summary comparison of debris vs background shadow properties."""
    debris_shadows = []
    debris_bright_excesses = []
    bg_shadows = []
    bg_bright_excesses = []

    for t in target_results:
        for m in t["measurements"]:
            s = m.get("shadow", {})
            b = m.get("bright_spot", {})
            if s.get("shadow_depth") is not None:
                debris_shadows.append(s["shadow_depth"])
            if b and b.get("bright_excess") is not None:
                debris_bright_excesses.append(b["bright_excess"])

    for bg in bg_results:
        s = bg.get("shadow", {})
        b = bg.get("bright_spot", {})
        if s.get("shadow_depth") is not None:
            bg_shadows.append(s["shadow_depth"])
        if b and b.get("bright_excess") is not None:
            bg_bright_excesses.append(b["bright_excess"])

    print("\n" + "=" * 70)
    print("SHADOW ANALYSIS SUMMARY: Debris vs Background")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Debris':>12} {'Background':>12} {'Δ':>10}")
    print("-" * 70)

    if debris_shadows and bg_shadows:
        d_mean = np.mean(debris_shadows)
        b_mean = np.mean(bg_shadows)
        print(f"{'Shadow depth (mean)':<35} {d_mean:>12.3f} {b_mean:>12.3f} {d_mean - b_mean:>+10.3f}")
        print(f"{'Shadow depth (median)':<35} {np.median(debris_shadows):>12.3f} {np.median(bg_shadows):>12.3f}")
        print(f"{'Shadow depth (std)':<35} {np.std(debris_shadows):>12.3f} {np.std(bg_shadows):>12.3f}")
        print(f"{'Shadow found count':<35} {len(debris_shadows):>12d} {len(bg_shadows):>12d}")
    else:
        print(f"{'Shadow depth (mean)':<35} {'N/A':>12} {'N/A':>12}")

    if debris_bright_excesses and bg_bright_excesses:
        d_mean = np.mean(debris_bright_excesses)
        b_mean = np.mean(bg_bright_excesses)
        print(f"\n{'Bright excess (mean σ)':<35} {d_mean:>12.2f} {b_mean:>12.2f} {d_mean - b_mean:>+10.2f}")
        print(f"{'Bright excess (median σ)':<35} {np.median(debris_bright_excesses):>12.2f} {np.median(bg_bright_excesses):>12.2f}")

    # Shadow distance
    debris_dists = []
    bg_dists = []
    for t in target_results:
        for m in t["measurements"]:
            s = m.get("shadow", {})
            if s.get("shadow_dist_m") is not None:
                debris_dists.append(s["shadow_dist_m"])
    for bg in bg_results:
        s = bg.get("shadow", {})
        if s.get("shadow_dist_m") is not None:
            bg_dists.append(s["shadow_dist_m"])

    if debris_dists and bg_dists:
        print(f"\n{'Shadow distance (mean m)':<35} {np.mean(debris_dists):>12.1f} {np.mean(bg_dists):>12.1f}")
        print(f"{'Shadow distance (median m)':<35} {np.median(debris_dists):>12.1f} {np.median(bg_dists):>12.1f}")

    # Verdict
    print(f"\n{'=' * 70}")
    if debris_shadows and bg_shadows:
        d_mean = np.mean(debris_shadows)
        b_mean = np.mean(bg_shadows)
        # T-test
        from scipy import stats
        if len(debris_shadows) >= 3 and len(bg_shadows) >= 3:
            t_stat, p_val = stats.mannwhitneyu(debris_shadows, bg_shadows, alternative='two-sided')
            print(f"Mann-Whitney U test: U={t_stat:.1f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("→ SIGNIFICANT difference: shadow signal IS measurable")
            else:
                print("→ No significant difference found (p >= 0.05)")
    else:
        print("Insufficient shadow detections for statistical test.")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="SSS Shadow Analysis for Marine Debris Detection")
    default_base = str(Path(__file__).resolve().parent.parent)
    parser.add_argument("--base-dir", type=str, default=default_base)
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = os.path.join(base_dir, "results", "shadow_analysis")
    os.makedirs(out_dir, exist_ok=True)

    transformer_utm = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    # Open GeoTIFFs
    ds_list = []
    for rel_path in TIFF_FILES:
        full_p = os.path.join(base_dir, rel_path)
        ds = rasterio.open(full_p)
        ds_list.append((os.path.basename(rel_path), ds))

    print(f"Processing {len(VERIFIED_TARGETS)} verified debris targets...")
    all_target_results = []
    for t in VERIFIED_TARGETS:
        result = process_target(t, ds_list, transformer_utm, out_dir)
        all_target_results.append(result)
        # Count measurements with shadow
        shadow_count = sum(1 for m in result["measurements"] if m.get("shadow"))
        print(f"  {t['target_id']} ({t['name']}): {len(result['measurements'])} TIFFs, "
              f"{shadow_count} with shadow data")

    print(f"\nProcessing background bright spots...")
    bg_results = process_background_spots(ds_list, transformer_utm, out_dir, num_regions=8)
    print(f"  Found {len(bg_results)} background bright spots with shadow analysis")

    # Write outputs
    # 1. Full JSON report
    report_path = os.path.join(out_dir, "shadow_report.json")
    with open(report_path, "w") as f:
        json.dump({"targets": all_target_results, "background_spots": bg_results},
                  f, indent=2, default=str)
    print(f"[✓] Saved shadow_report.json")

    # 2. Comparison CSV
    csv_path = write_comparison_csv(all_target_results, bg_results, out_dir)

    # 3. Summary
    try:
        print_summary(all_target_results, bg_results)
    except Exception as e:
        print(f"Summary error: {e}")

    # Close
    for _, ds in ds_list:
        ds.close()

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
