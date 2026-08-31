#!/usr/bin/env python3
"""
scripts/prepare_noaa_sss.py

Converts NOAA H11833 Side-Scan Sonar (SSS) GeoTIFF mosaics into a clean, reproducible
YOLO object-detection dataset with:
- Train (~70%) / val (~20%) / test (~10%) splits with geographic clustering to prevent data leakage
- YOLO normalized bounding box labels
- Separate image geolocation metadata (image_geolocation.csv)
- Annotation metadata (annotations.csv)
- Split manifest (split_manifest.csv)
- TIFF metadata (tiff_metadata.json)
- data.yaml

Source TIFF files are accessed READ-ONLY and never modified.
"""

import os
import sys
import json
import csv
import shutil
import argparse
import re
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image

# Default configuration
DEFAULT_PATCH_SIZE = 1024
DEFAULT_OVERLAP = 128
CLASS_NAME = "marine_debris"
CLASS_ID = 0

# GeoTIFF relative paths
TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif"
]

def dms_to_dd(dms_str: str) -> float:
    """Parse DMS string to decimal degrees."""
    parts = re.split(r'[°\'\"/\-\s]+', dms_str.strip())
    parts = [p for p in parts if p]
    d = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    s = float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + s / 3600.0
    if 'S' in dms_str.upper() or 'W' in dms_str.upper() or dms_str.startswith('-'):
        dd = -dd
    return dd

# Verified marine debris targets from NOAA Hydrographic Survey H11833 Descriptive Report
# Evaluated and confirmed in Gulf of Mexico Marine Debris Project (GOMMDP)
VERIFIED_TARGETS = [
    {
        "name": "DtoN_1_Pipeline_Exposed",
        "lat_str": "28° 59' 03.3504\" N",
        "lon_str": "089° 18' 09.3672\" W",
        "desc": "10m long exposed pipeline section separated from seafloor, 2.1m elevation",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN1",
        "length_m": 12.0,
        "width_m": 6.0,
        "cluster": "NE_Cluster"
    },
    {
        "name": "DtoN_2_Obstruction_Debris",
        "lat_str": "28° 58' 58.6344\" N",
        "lon_str": "089° 18' 07.3908\" W",
        "desc": "11m x 8.5m obstruction debris rising 1.4m above seafloor",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN2",
        "length_m": 14.0,
        "width_m": 10.0,
        "cluster": "NE_Cluster"
    },
    {
        "name": "DtoN_5_Damaged_Wellhead_Ruins",
        "lat_str": "29° 01' 35.0112\" N",
        "lon_str": "089° 17' 25.5372\" W",
        "desc": "Damaged wellhead and platform ruins debris structure",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN5",
        "length_m": 12.0,
        "width_m": 12.0,
        "cluster": "NE_Cluster"
    },
    {
        "name": "DtoN_3_1_Elevated_Pipeline",
        "lat_str": "28° 54' 58.5288\" N",
        "lon_str": "089° 21' 56.8800\" W",
        "desc": "Elevated pipeline section debris",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN3.1",
        "length_m": 16.0,
        "width_m": 6.0,
        "cluster": "Central_East_Cluster"
    },
    {
        "name": "DtoN_3_2_Elevated_Pipeline",
        "lat_str": "28° 54' 48.1320\" N",
        "lon_str": "089° 22' 05.1348\" W",
        "desc": "Elevated pipeline section debris",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN3.2",
        "length_m": 16.0,
        "width_m": 6.0,
        "cluster": "Central_East_Cluster"
    },
    {
        "name": "Rig_Caisson_Pipeline_Feature",
        "lat_str": "28° 54' 54.197\" N",
        "lon_str": "089° 22' 16.974\" W",
        "desc": "Linear mudline debris feature near AWOIS 11805 / DtoN 3",
        "confidence": "medium",
        "source": "NOAA_H11833_AWOIS11805",
        "length_m": 16.0,
        "width_m": 8.0,
        "cluster": "Central_East_Cluster"
    },
    {
        "name": "DtoN_3_3_Obstruction_Debris",
        "lat_str": "28° 54' 10.6380\" N",
        "lon_str": "089° 25' 30.9252\" W",
        "desc": "35-ft obstruction / elevated pipeline section rising 2.1m",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN3.3",
        "length_m": 14.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "DtoN_4_Ruined_Training_Wall",
        "lat_str": "28° 54' 42.4764\" N",
        "lon_str": "089° 25' 24.0312\" W",
        "desc": "9-ft obstruction / ruined submerged structure debris",
        "confidence": "high",
        "source": "NOAA_H11833_DtoN4",
        "length_m": 16.0,
        "width_m": 10.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Training_Wall_Ruins_Center",
        "lat_str": "28° 54' 41.62\" N",
        "lon_str": "089° 25' 22.30\" W",
        "desc": "Training wall in ruins debris structure",
        "confidence": "high",
        "source": "NOAA_H11833_TrainingWallRuins",
        "length_m": 22.0,
        "width_m": 10.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Obstruction_22ft",
        "lat_str": "28° 54' 38.79\" N",
        "lon_str": "089° 25' 17.40\" W",
        "desc": "22ft obstruction debris surveyed in Southwest Pass",
        "confidence": "high",
        "source": "NOAA_H11833_OBSTRN22",
        "length_m": 12.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Obstruction_17ft",
        "lat_str": "28° 54' 59.10\" N",
        "lon_str": "089° 25' 34.45\" W",
        "desc": "17ft obstruction debris surveyed in Southwest Pass",
        "confidence": "high",
        "source": "NOAA_H11833_OBSTRN17",
        "length_m": 12.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Obstruction_18ft",
        "lat_str": "28° 54' 59.78\" N",
        "lon_str": "089° 25' 32.53\" W",
        "desc": "18ft obstruction debris surveyed in Southwest Pass",
        "confidence": "high",
        "source": "NOAA_H11833_OBSTRN18",
        "length_m": 12.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Obstruction_14ft",
        "lat_str": "28° 54' 32.396\" N",
        "lon_str": "089° 25' 59.387\" W",
        "desc": "Obstruction rising 1.6m above seafloor in SW pass",
        "confidence": "high",
        "source": "NOAA_H11833_OBSTRN14",
        "length_m": 12.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Obstruction_10ft",
        "lat_str": "28° 54' 30.07\" N",
        "lon_str": "089° 26' 03.29\" W",
        "desc": "10ft obstruction debris in SW pass",
        "confidence": "high",
        "source": "NOAA_H11833_OBSTRN10",
        "length_m": 12.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Obstruction_Charted_Retained",
        "lat_str": "28° 54' 18.14\" N",
        "lon_str": "089° 25' 45.09\" W",
        "desc": "Obstruction debris in SW pass",
        "confidence": "high",
        "source": "NOAA_H11833_OBSTRN_Charted",
        "length_m": 12.0,
        "width_m": 8.0,
        "cluster": "SW_Pass_Main_Cluster"
    },
    {
        "name": "Pipe_PA_Debris",
        "lat_str": "28° 53' 35.68\" N",
        "lon_str": "089° 25' 44.09\" W",
        "desc": "Submerged pipe debris investigated with SSS and MBES",
        "confidence": "high",
        "source": "NOAA_H11833_PipePA",
        "length_m": 14.0,
        "width_m": 6.0,
        "cluster": "SW_Pass_South_Cluster"
    },
    {
        "name": "AWOIS_11622_Wreck_Debris",
        "lat_str": "28° 53' 19.464\" N",
        "lon_str": "089° 26' 14.276\" W",
        "desc": "Small angular debris contact from wreck PA investigated with MBES",
        "confidence": "high",
        "source": "NOAA_H11833_AWOIS11622",
        "length_m": 12.0,
        "width_m": 10.0,
        "cluster": "SW_Pass_South_Cluster"
    }
]

# Cluster to Split mapping: ensures completely unseen, isolated spatial test set and val set
# Zero spatial leakage between splits
CLUSTER_SPLIT_MAP = {
    "Central_East_Cluster": "test",      # Completely held-out unseen region (~10%)
    "NE_Cluster": "val",                 # Validation region (~20%)
    "SW_Pass_Main_Cluster": "train",     # Training region (~70%)
    "SW_Pass_South_Cluster": "train"     # Training region
}

# Offsets per split to achieve balanced ~70% train, ~20% val, ~10% test
SPLIT_OFFSETS = {
    "train": [
        (0, 0),
        (-128, -128),
        (128, 128),
        (-128, 128),
        (128, -128),
        (-256, 0),
        (256, 0),
        (0, -256),
        (0, 256)
    ],
    "val": [
        (0, 0),
        (-128, -128),
        (128, 128),
        (-128, 128),
        (128, -128)
    ],
    "test": [
        (0, 0),
        (-128, -128),
        (128, 128)
    ]
}

def export_tiff_metadata(base_dir: str):
    """Inspect and save GeoTIFF metadata to tiff_metadata.json."""
    meta_list = []
    for rel_path in TIFF_FILES:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"TIFF file not found: {full_path}")
        with rasterio.open(full_path) as src:
            meta_list.append({
                "filename": os.path.basename(rel_path),
                "relative_path": rel_path,
                "width": src.width,
                "height": src.height,
                "crs": src.crs.to_string() if src.crs else None,
                "transform": list(src.transform),
                "pixel_size": [abs(src.transform[0]), abs(src.transform[4])],
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top
                },
                "band_count": src.count,
                "data_type": src.dtypes[0]
            })
    
    meta_dir = os.path.join(base_dir, "datasets/noaa-debris/metadata")
    os.makedirs(meta_dir, exist_ok=True)
    out_file = os.path.join(meta_dir, "tiff_metadata.json")
    with open(out_file, "w") as f:
        json.dump(meta_list, f, indent=2)
    print(f"[✓] Saved TIFF metadata to {out_file}")


def prepare_dataset(base_dir: str, patch_size: int = DEFAULT_PATCH_SIZE, overlap: int = DEFAULT_OVERLAP):
    """Extract patches, compute YOLO labels, georeferencing, and split allocations."""
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)

    # Clean existing yolo and metadata folders (keeping raw untouched)
    yolo_dir = os.path.join(base_dir, "datasets/noaa-debris/yolo")
    meta_dir = os.path.join(base_dir, "datasets/noaa-debris/metadata")

    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Compute UTM and decimal coordinates for verified targets
    targets_with_coords = []
    for t in VERIFIED_TARGETS:
        lat = dms_to_dd(t["lat_str"])
        lon = dms_to_dd(t["lon_str"])
        if lon > 0:
            lon = -lon
        utm_x, utm_y = transformer_to_utm.transform(lon, lat)
        t_copy = dict(t)
        t_copy["lat"] = lat
        t_copy["lon"] = lon
        t_copy["utm_x"] = utm_x
        t_copy["utm_y"] = utm_y
        targets_with_coords.append(t_copy)

    # Open TIFF datasets read-only
    ds_list = []
    for rel_path in TIFF_FILES:
        full_p = os.path.join(base_dir, rel_path)
        ds = rasterio.open(full_p)
        ds_list.append((os.path.basename(rel_path), ds))

    patches_meta = []
    annotations_list = []
    split_manifest = []

    patch_counter = 0
    ann_counter = 0

    print(f"Extracting target-centered patches across 100% and 200% SSS coverage...")
    
    for t in targets_with_coords:
        split = CLUSTER_SPLIT_MAP[t["cluster"]]
        offsets = SPLIT_OFFSETS[split]
        tx, ty = t["utm_x"], t["utm_y"]

        for tiff_name, ds in ds_list:
            t_row, t_col = ds.index(tx, ty)
            
            # Extract patches with designated split offsets
            for off_x, off_y in offsets:
                win_col = int(t_col - patch_size // 2 + off_x)
                win_row = int(t_row - patch_size // 2 + off_y)

                # Ensure within dataset bounds
                if win_col < 0 or win_row < 0 or win_col + patch_size > ds.width or win_row + patch_size > ds.height:
                    continue

                window = Window(win_col, win_row, patch_size, patch_size)
                data = ds.read(1, window=window)

                # Check if data contains valid sonar signal (not empty / nodata)
                nonzeros = np.count_nonzero(data)
                valid_ratio = nonzeros / data.size
                if valid_ratio < 0.70 or data.mean() < 15:
                    continue

                patch_counter += 1
                patch_id = f"H11833_{patch_counter:06d}"
                img_filename = f"{patch_id}.png"

                # Calculate true geographic coordinates of patch center
                patch_center_col = win_col + patch_size / 2.0
                patch_center_row = win_row + patch_size / 2.0
                patch_utm_x, patch_utm_y = ds.xy(patch_center_row, patch_center_col)
                patch_lon, patch_lat = transformer_to_wgs84.transform(patch_utm_x, patch_utm_y)

                # Save image
                img = Image.fromarray(data)
                img_path = os.path.join(yolo_dir, "images", split, img_filename)
                img.save(img_path, format="PNG")

                # Find all targets that fall within this patch window
                patch_yolo_labels = []
                for candidate_t in targets_with_coords:
                    cand_row, cand_col = ds.index(candidate_t["utm_x"], candidate_t["utm_y"])
                    rel_px = cand_col - win_col
                    rel_py = cand_row - win_row

                    if 0 <= rel_px <= patch_size and 0 <= rel_py <= patch_size:
                        x_center = rel_px / patch_size
                        y_center = rel_py / patch_size
                        # Convert target length/width (meters) to normalized patch coords (0.5m/px)
                        box_w = (candidate_t["length_m"] / 0.5) / patch_size
                        box_h = (candidate_t["width_m"] / 0.5) / patch_size

                        # Clamp values
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        box_w = max(0.005, min(1.0, box_w))
                        box_h = max(0.005, min(1.0, box_h))

                        patch_yolo_labels.append((CLASS_ID, x_center, y_center, box_w, box_h))

                        # Record annotation metadata
                        ann_counter += 1
                        ann_id = f"ANN_H11833_{ann_counter:06d}"
                        annotations_list.append({
                            "annotation_id": ann_id,
                            "patch_id": patch_id,
                            "class": CLASS_NAME,
                            "confidence": candidate_t["confidence"],
                            "verified": "True",
                            "source": candidate_t["source"]
                        })

                # Write YOLO label file
                label_path = os.path.join(yolo_dir, "labels", split, f"{patch_id}.txt")
                with open(label_path, "w") as lf:
                    for lab in patch_yolo_labels:
                        lf.write(f"{lab[0]} {lab[1]:.6f} {lab[2]:.6f} {lab[3]:.6f} {lab[4]:.6f}\n")

                # Record geolocation metadata
                patches_meta.append({
                    "patch_id": patch_id,
                    "image_filename": img_filename,
                    "source_tiff": tiff_name,
                    "utm_x": f"{patch_utm_x:.2f}",
                    "utm_y": f"{patch_utm_y:.2f}",
                    "latitude": f"{patch_lat:.8f}",
                    "longitude": f"{patch_lon:.8f}",
                    "split": split
                })

                split_manifest.append({
                    "patch_id": patch_id,
                    "split": split
                })

    print(f"[✓] Extracted positive target patches. Total positive annotations: {ann_counter}")

    # Extract baseline background / negative patches from diverse seabed textures in each split cluster area
    print(f"Extracting baseline background seabed patches...")
    bg_centers = [
        # SW Pass Main (train)
        ("train", "H11833_1of2.tif", 263000, 3201000, [-128, 128], [-128, 128]),
        ("train", "H11833_1of2.tif", 263500, 3200000, [0], [0]),
        ("train", "H11833_2of2.tif", 263200, 3200800, [-128, 128], [-128, 128]),
        ("train", "H11833_2of2.tif", 263800, 3200200, [0], [0]),
        # SW Pass South (train)
        ("train", "H11833_1of2.tif", 262800, 3198300, [-128, 128], [0]),
        ("train", "H11833_2of2.tif", 262500, 3198500, [0], [-128, 128]),
        # NE Region (val)
        ("val", "H11833_1of2.tif", 276000, 3209000, [-128, 128], [0]),
        ("val", "H11833_2of2.tif", 276500, 3208500, [0], [-128, 128]),
        # Central East (test)
        ("test", "H11833_1of2.tif", 269800, 3201500, [0], [0]),
        ("test", "H11833_2of2.tif", 269500, 3200500, [0], [0])
    ]

    for split, tiff_name, bg_x, bg_y, offs_x, offs_y in bg_centers:
        ds = [d for name, d in ds_list if name == tiff_name][0]
        bg_row, bg_col = ds.index(bg_x, bg_y)

        for off_x in offs_x:
            for off_y in offs_y:
                win_col = int(bg_col - patch_size // 2 + off_x)
                win_row = int(bg_row - patch_size // 2 + off_y)

                if win_col < 0 or win_row < 0 or win_col + patch_size > ds.width or win_row + patch_size > ds.height:
                    continue

                # Ensure no verified target is within this background patch
                is_clean = True
                for t in targets_with_coords:
                    t_r, t_c = ds.index(t["utm_x"], t["utm_y"])
                    if win_col - 100 <= t_c <= win_col + patch_size + 100 and win_row - 100 <= t_r <= win_row + patch_size + 100:
                        is_clean = False
                        break
                if not is_clean:
                    continue

                window = Window(win_col, win_row, patch_size, patch_size)
                data = ds.read(1, window=window)

                nonzeros = np.count_nonzero(data)
                valid_ratio = nonzeros / data.size
                if valid_ratio < 0.85 or data.mean() < 20 or data.std() < 5:
                    continue

                patch_counter += 1
                patch_id = f"H11833_{patch_counter:06d}"
                img_filename = f"{patch_id}.png"

                patch_center_col = win_col + patch_size / 2.0
                patch_center_row = win_row + patch_size / 2.0
                patch_utm_x, patch_utm_y = ds.xy(patch_center_row, patch_center_col)
                patch_lon, patch_lat = transformer_to_wgs84.transform(patch_utm_x, patch_utm_y)

                # Save background image
                img = Image.fromarray(data)
                img_path = os.path.join(yolo_dir, "images", split, img_filename)
                img.save(img_path, format="PNG")

                # Empty YOLO label file for negative background patch
                label_path = os.path.join(yolo_dir, "labels", split, f"{patch_id}.txt")
                with open(label_path, "w") as lf:
                    pass

                # Geolocation metadata
                patches_meta.append({
                    "patch_id": patch_id,
                    "image_filename": img_filename,
                    "source_tiff": tiff_name,
                    "utm_x": f"{patch_utm_x:.2f}",
                    "utm_y": f"{patch_utm_y:.2f}",
                    "latitude": f"{patch_lat:.8f}",
                    "longitude": f"{patch_lon:.8f}",
                    "split": split
                })

                split_manifest.append({
                    "patch_id": patch_id,
                    "split": split
                })

    # Extract 150 additional carefully selected negative SSS patches
    print(f"Extracting 150 additional diverse negative SSS background patches (105 train / 30 val / 15 test)...")
    existing_positions = [(float(p["utm_x"]), float(p["utm_y"])) for p in patches_meta]
    regions_cfg = {
        "train": {"x_range": (261200, 266500), "y_range": (3197600, 3204000), "count_t1": 55, "count_t2": 50},
        "val":   {"x_range": (274500, 277800), "y_range": (3206000, 3213500), "count_t1": 15, "count_t2": 15},
        "test":  {"x_range": (268000, 271500), "y_range": (3199500, 3203500), "count_t1": 8,  "count_t2": 7}
    }

    for split, cfg in regions_cfg.items():
        for tiff_rel, req_count in [("H11833_1of2.tif", cfg["count_t1"]), ("H11833_2of2.tif", cfg["count_t2"])]:
            ds = [d for name, d in ds_list if name == tiff_rel][0]
            r_a, c_a = ds.index(cfg["x_range"][0], cfg["y_range"][0])
            r_b, c_b = ds.index(cfg["x_range"][1], cfg["y_range"][1])
            c_min = max(0, min(c_a, c_b))
            c_max = min(ds.width - patch_size, max(c_a, c_b))
            r_min = max(0, min(r_a, r_b))
            r_max = min(ds.height - patch_size, max(r_a, r_b))

            candidates = []
            for r in range(r_min, r_max, 128):
                for c in range(c_min, c_max, 128):
                    cx_px = c + patch_size / 2.0
                    cy_px = r + patch_size / 2.0
                    center_x, center_y = ds.xy(cy_px, cx_px)

                    # Safety margin: minimum distance to any verified target >= 450m
                    dists_tgt = [np.hypot(center_x - t["utm_x"], center_y - t["utm_y"]) for t in targets_with_coords]
                    if min(dists_tgt) < 450.0:
                        continue

                    # Minimum distance to any already extracted patch >= 200m
                    dists_exist = [np.hypot(center_x - ex[0], center_y - ex[1]) for ex in existing_positions]
                    if min(dists_exist) < 200.0:
                        continue

                    win = Window(c, r, patch_size, patch_size)
                    data = ds.read(1, window=win)
                    valid_ratio = np.count_nonzero(data) / data.size
                    if valid_ratio < 0.85:
                        continue

                    mean_val = float(data.mean())
                    std_val = float(data.std())
                    if mean_val < 18 or mean_val > 175 or std_val < 6.0:
                        continue

                    score = std_val * (valid_ratio ** 2)
                    candidates.append({
                        "tiff_name": tiff_rel,
                        "win_col": c,
                        "win_row": r,
                        "center_utm": (center_x, center_y),
                        "score": score,
                        "data": data
                    })

            candidates.sort(key=lambda x: x["score"], reverse=True)
            min_sep = 200.0
            selected = []
            for cand in candidates:
                if len(selected) >= req_count:
                    break
                if all(np.hypot(cand["center_utm"][0] - s["center_utm"][0], cand["center_utm"][1] - s["center_utm"][1]) >= min_sep for s in selected):
                    selected.append(cand)

            for item in selected:
                patch_counter += 1
                patch_id = f"H11833_{patch_counter:06d}"
                img_filename = f"{patch_id}.png"

                patch_center_col = item["win_col"] + patch_size / 2.0
                patch_center_row = item["win_row"] + patch_size / 2.0
                patch_utm_x, patch_utm_y = ds.xy(patch_center_row, patch_center_col)
                patch_lon, patch_lat = transformer_to_wgs84.transform(patch_utm_x, patch_utm_y)

                # Save background image
                img = Image.fromarray(item["data"])
                img_path = os.path.join(yolo_dir, "images", split, img_filename)
                img.save(img_path, format="PNG")

                # Empty YOLO label file
                label_path = os.path.join(yolo_dir, "labels", split, f"{patch_id}.txt")
                with open(label_path, "w") as lf:
                    pass

                # Geolocation metadata
                patches_meta.append({
                    "patch_id": patch_id,
                    "image_filename": img_filename,
                    "source_tiff": item["tiff_name"],
                    "utm_x": f"{patch_utm_x:.2f}",
                    "utm_y": f"{patch_utm_y:.2f}",
                    "latitude": f"{patch_lat:.8f}",
                    "longitude": f"{patch_lon:.8f}",
                    "split": split
                })

                split_manifest.append({
                    "patch_id": patch_id,
                    "split": split
                })

                existing_positions.append((patch_utm_x, patch_utm_y))

    # Close GeoTIFFs
    for _, ds in ds_list:
        ds.close()

    # Write metadata CSVs
    # 1. image_geolocation.csv
    geo_csv_path = os.path.join(meta_dir, "image_geolocation.csv")
    with open(geo_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_id", "image_filename", "source_tiff", "utm_x", "utm_y", "latitude", "longitude"])
        writer.writeheader()
        for p in patches_meta:
            writer.writerow({
                "patch_id": p["patch_id"],
                "image_filename": p["image_filename"],
                "source_tiff": p["source_tiff"],
                "utm_x": p["utm_x"],
                "utm_y": p["utm_y"],
                "latitude": p["latitude"],
                "longitude": p["longitude"]
            })
    print(f"[✓] Saved image_geolocation.csv ({len(patches_meta)} records)")

    # 2. annotations.csv
    ann_csv_path = os.path.join(meta_dir, "annotations.csv")
    with open(ann_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["annotation_id", "patch_id", "class", "confidence", "verified", "source"])
        writer.writeheader()
        for a in annotations_list:
            writer.writerow(a)
    print(f"[✓] Saved annotations.csv ({len(annotations_list)} records)")

    # 3. split_manifest.csv
    split_csv_path = os.path.join(meta_dir, "split_manifest.csv")
    with open(split_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_id", "split"])
        writer.writeheader()
        for s in split_manifest:
            writer.writerow(s)
    print(f"[✓] Saved split_manifest.csv ({len(split_manifest)} records)")

    # 4. data.yaml
    data_yaml_path = os.path.join(yolo_dir, "data.yaml")
    yaml_content = f"""# YOLOv8 Dataset Configuration - NOAA H11833 Side-Scan Sonar Marine Debris
path: {yolo_dir}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[✓] Saved data.yaml")

    # Print summary
    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_manifest:
        counts[s["split"]] += 1

    print("\n" + "=" * 50)
    print(f"NOAA SSS Data Preparation Complete:")
    print(f"Total patches generated: {len(patches_meta)}")
    print(f"  Train: {counts['train']} ({counts['train']/len(patches_meta)*100:.1f}%)")
    print(f"  Val:   {counts['val']} ({counts['val']/len(patches_meta)*100:.1f}%)")
    print(f"  Test:  {counts['test']} ({counts['test']/len(patches_meta)*100:.1f}%)")
    print(f"Total annotations:     {len(annotations_list)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Prepare NOAA SSS YOLO Object Detection Dataset")
    default_base = str(Path(__file__).resolve().parent.parent)
    parser.add_argument("--base-dir", type=str, default=default_base, help="Workspace base directory")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size in pixels")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap in pixels")
    args = parser.parse_args()

    export_tiff_metadata(args.base_dir)
    prepare_dataset(args.base_dir, patch_size=args.patch_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
