#!/usr/bin/env python3
"""
scripts/prepare_noaa_e3.py

Prepares the E3 target-centered YOLO dataset (512x512 crops) from NOAA H11833 SSS GeoTIFFs:
- Strict contact-level split: contacts are never split across train/val/test
- Target-centered 512x512 crops derived from verified NOAA contact coordinates
- Exact YOLO bounding box calculation from target coordinate and crop window geometry
- Genuine background / negative crops with >= 450m safety margin from all verified contacts
- Complete metadata generation: crop_metadata.csv, split_manifest.csv, data.yaml
- Visual QA generator drawing YOLO boxes on crops
- Pilot mode for initial verification before full dataset generation

Source GeoTIFFs are accessed READ-ONLY.
"""

import os
import sys
import csv
import json
import shutil
import argparse
import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image, ImageDraw, ImageFont

# Verified NOAA H11833 Targets
VERIFIED_TARGETS = [
    {
        "target_id": "TGT001",
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
        "target_id": "TGT002",
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
        "target_id": "TGT003",
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
        "target_id": "TGT004",
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
        "target_id": "TGT005",
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
        "target_id": "TGT006",
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
        "target_id": "TGT007",
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
        "target_id": "TGT008",
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
        "target_id": "TGT009",
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
        "target_id": "TGT010",
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
        "target_id": "TGT011",
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
        "target_id": "TGT012",
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
        "target_id": "TGT013",
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
        "target_id": "TGT014",
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
        "target_id": "TGT015",
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
        "target_id": "TGT016",
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
        "target_id": "TGT017",
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

# Strict Contact-level Split Mapping (zero geographic leakage)
CLUSTER_SPLIT_MAP = {
    "SW_Pass_Main_Cluster": "train",    # 9 contacts
    "SW_Pass_South_Cluster": "train",   # 2 contacts
    "NE_Cluster": "val",                # 3 contacts
    "Central_East_Cluster": "test"      # 3 contacts (completely held out)
}

TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif"
]

CROP_SIZE = 512
CLASS_ID = 0
CLASS_NAME = "marine_debris"


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


def get_verified_targets_with_utm():
    """Returns verified target list enriched with UTM coordinates."""
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)
    targets = []
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
        t_copy["split"] = CLUSTER_SPLIT_MAP[t["cluster"]]
        targets.append(t_copy)
    return targets


def compute_shifted_crop_window(t: dict, ds: rasterio.DatasetReader, off_x: int, off_y: int, all_targets: list, crop_size: int = CROP_SIZE):
    """
    Computes a 512x512 crop window centered around target t with offset (off_x, off_y).
    If the target (or any nearby target in the same split) would be clipped by the crop boundary,
    the crop window is shifted so that complete target bounding box(es) are fully inside [0, 512].
    
    Returns (win_col, win_row).
    """
    inv_transform = ~ds.transform
    c_col, c_row = inv_transform * (t["utm_x"], t["utm_y"])
    t_bw = t["length_m"] / 0.5
    t_bh = t["width_m"] / 0.5
    t_x1 = c_col - t_bw / 2.0
    t_x2 = c_col + t_bw / 2.0
    t_y1 = c_row - t_bh / 2.0
    t_y2 = c_row + t_bh / 2.0

    # 1. Initial crop window centered at target + offset
    win_col = int(round(c_col - crop_size / 2.0 + off_x))
    win_row = int(round(c_row - crop_size / 2.0 + off_y))

    # 2. Shift to fully contain primary target t
    if win_col > t_x1:
        win_col = int(np.floor(t_x1))
    if win_col + crop_size < t_x2:
        win_col = int(np.ceil(t_x2 - crop_size))
    if win_row > t_y1:
        win_row = int(np.floor(t_y1))
    if win_row + crop_size < t_y2:
        win_row = int(np.ceil(t_y2 - crop_size))

    # 3. Check and adjust for secondary targets in the same split
    split_targets = [cand for cand in all_targets if cand["split"] == t["split"]]

    for _ in range(5):
        changed = False
        for cand in split_targets:
            cand_col, cand_row = inv_transform * (cand["utm_x"], cand["utm_y"])
            cand_bw = cand["length_m"] / 0.5
            cand_bh = cand["width_m"] / 0.5
            cand_x1 = cand_col - cand_bw / 2.0
            cand_x2 = cand_col + cand_bw / 2.0
            cand_y1 = cand_row - cand_bh / 2.0
            cand_y2 = cand_row + cand_bh / 2.0

            # Overlaps current crop window?
            overlaps = (cand_x2 > win_col and cand_x1 < win_col + crop_size and
                        cand_y2 > win_row and cand_y1 < win_row + crop_size)
            if not overlaps:
                continue

            # Fully inside current crop window?
            fully_inside = (cand_x1 >= win_col and cand_x2 <= win_col + crop_size and
                            cand_y1 >= win_row and cand_y2 <= win_row + crop_size)
            if fully_inside:
                continue

            # Partially clipped: try shifting to encompass both targets
            new_min_x = min(t_x1, cand_x1)
            new_max_x = max(t_x2, cand_x2)
            new_min_y = min(t_y1, cand_y1)
            new_max_y = max(t_y2, cand_y2)

            if (new_max_x - new_min_x <= crop_size) and (new_max_y - new_min_y <= crop_size):
                if win_col > new_min_x:
                    win_col = int(np.floor(new_min_x))
                    changed = True
                if win_col + crop_size < new_max_x:
                    win_col = int(np.ceil(new_max_x - crop_size))
                    changed = True
                if win_row > new_min_y:
                    win_row = int(np.floor(new_min_y))
                    changed = True
                if win_row + crop_size < new_max_y:
                    win_row = int(np.ceil(new_max_y - crop_size))
                    changed = True
            else:
                # Cannot fit both in crop_size: shift away from cand to exclude it completely
                if cand_x1 < win_col and cand_x2 > win_col:
                    candidate_win_col = int(np.ceil(cand_x2))
                    if candidate_win_col <= t_x1 and candidate_win_col + crop_size >= t_x2:
                        win_col = candidate_win_col
                        changed = True
                elif cand_x2 > win_col + crop_size and cand_x1 < win_col + crop_size:
                    candidate_win_col = int(np.floor(cand_x1 - crop_size))
                    if candidate_win_col <= t_x1 and candidate_win_col + crop_size >= t_x2:
                        win_col = candidate_win_col
                        changed = True

                if cand_y1 < win_row and cand_y2 > win_row:
                    candidate_win_row = int(np.ceil(cand_y2))
                    if candidate_win_row <= t_y1 and candidate_win_row + crop_size >= t_y2:
                        win_row = candidate_win_row
                        changed = True
                elif cand_y2 > win_row + crop_size and cand_y1 < win_row + crop_size:
                    candidate_win_row = int(np.floor(cand_y1 - crop_size))
                    if candidate_win_row <= t_y1 and candidate_win_row + crop_size >= t_y2:
                        win_row = candidate_win_row
                        changed = True

        if not changed:
            break

    # 4. Clamp crop window to GeoTIFF raster boundaries
    win_col = max(0, min(ds.width - crop_size, win_col))
    win_row = max(0, min(ds.height - crop_size, win_row))

    return win_col, win_row


def has_valid_sonar(ds: rasterio.DatasetReader, col: float, row: float,
                     kernel_radius: int = 15, min_valid_ratio: float = 0.05) -> bool:
    """
    Check if a neighborhood around (col, row) contains valid sonar signal.
    Uses a (2*kernel_radius+1)² kernel centered at the target pixel.
    Accepts the target when at least min_valid_ratio of the kernel pixels are nonzero.
    """
    c = int(round(col))
    r = int(round(row))
    x1 = max(0, c - kernel_radius)
    y1 = max(0, r - kernel_radius)
    x2 = min(ds.width, c + kernel_radius + 1)
    y2 = min(ds.height, r + kernel_radius + 1)
    if x1 >= x2 or y1 >= y2:
        return False
    data = ds.read(1, window=Window(x1, y1, x2 - x1, y2 - y1))
    total = data.size
    if total == 0:
        return False
    nz_ratio = np.count_nonzero(data) / total
    return nz_ratio >= min_valid_ratio


def has_valid_crop_window(ds: rasterio.DatasetReader, win_col: int, win_row: int,
                          crop_size: int = CROP_SIZE, min_valid_ratio: float = 0.40,
                          min_mean: float = 20.0) -> bool:
    """
    Check if a full crop window contains valid sonar data.
    Used as a fallback when the target center pixel is in a nodata gap.
    """
    if win_col < 0 or win_row < 0 or win_col + crop_size > ds.width or win_row + crop_size > ds.height:
        return False
    data = ds.read(1, window=Window(win_col, win_row, crop_size, crop_size))
    total = data.size
    if total == 0:
        return False
    nz_ratio = np.count_nonzero(data) / total
    mean_val = float(np.mean(data))
    return nz_ratio >= min_valid_ratio and mean_val >= min_mean


def generate_e3_dataset(base_dir: str, pilot_only: bool = True):
    """
    Extracts 512x512 target-centered crops and background crops for E3.
    If pilot_only=True, generates 20-30 positive crops and 10-20 negative crops,
    plus visual QA images.
    """
    base_path = Path(base_dir)
    e3_dir = base_path / "datasets" / "noaa-debris" / "e3"
    qa_dir = e3_dir / "qa"
    meta_dir = e3_dir / "metadata"

    # Reset/create clean directories
    if e3_dir.exists():
        shutil.rmtree(e3_dir)

    for split in ["train", "val", "test"]:
        (e3_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (e3_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    targets = get_verified_targets_with_utm()

    # Open GeoTIFFs read-only
    ds_list = []
    for rel_path in TIFF_FILES:
        full_p = base_path / rel_path
        if not full_p.exists():
            raise FileNotFoundError(f"Source TIFF not found: {full_p}")
        ds = rasterio.open(str(full_p))
        ds_list.append((full_p.name, ds))

    crop_metadata_records = []
    split_manifest_records = []
    numerical_qa_records = []
    target_crop_counters = {t["target_id"]: 0 for t in targets}
    bg_counter = 0

    print("=" * 60)
    print(f"Generating E3 dataset (Pilot Mode: {pilot_only})...")
    print("=" * 60)

    # 1. POSITIVE CROPS GENERATION
    positive_crop_plan = []

    if pilot_only:
        # Pilot: 24 positive crops across splits (14 train, 6 val, 4 test)
        # Train contacts (11 targets in SW Pass)
        train_targets_pilot = ["TGT007", "TGT010", "TGT012", "TGT014", "TGT016", "TGT017", "TGT008"]
        for tid in train_targets_pilot:
            t = next(x for x in targets if x["target_id"] == tid)
            # Find the best TIFF for this target.
            # First pass: prefer a TIFF where the center pixel has valid sonar.
            # Fallback: if no TIFF has a valid center (e.g. TGT015 nodata gap),
            #           use the TIFF whose full 512×512 crop window has the best
            #           valid-ratio × mean product.
            best_tname = None
            best_score = -1
            fallback_tname = None
            fallback_score = -1
            for tname, ds in ds_list:
                row, col = ds.index(t["utm_x"], t["utm_y"])
                if not (0 <= col < ds.width and 0 <= row < ds.height):
                    continue
                center_ok = has_valid_sonar(ds, col, row)
                if center_ok:
                    best_tname = tname
                    break  # center-valid TIFF is preferred, stop searching
                # Crop-window fallback: score = valid_ratio × mean
                test_wc = int(round(col - CROP_SIZE / 2.0))
                test_wr = int(round(row - CROP_SIZE / 2.0))
                if has_valid_crop_window(ds, test_wc, test_wr):
                    cw_data = ds.read(1, window=Window(test_wc, test_wr, CROP_SIZE, CROP_SIZE))
                    cw_vr = np.count_nonzero(cw_data) / cw_data.size
                    cw_mean = float(np.mean(cw_data))
                    score = cw_vr * cw_mean
                    if score > fallback_score:
                        fallback_score = score
                        fallback_tname = tname
            if best_tname is None:
                best_tname = fallback_tname
            if best_tname is not None:
                positive_crop_plan.append((t, best_tname, (0, 0)))
                if tid in ["TGT007", "TGT010", "TGT014", "TGT016", "TGT017", "TGT012", "TGT008"] and len([x for x in positive_crop_plan if x[0]["split"] == "train"]) < 14:
                    positive_crop_plan.append((t, best_tname, (-32, 32)))
        # Keep exactly 14 train positive crops
        positive_crop_plan = [p for p in positive_crop_plan if p[0]["split"] != "train"] + [p for p in positive_crop_plan if p[0]["split"] == "train"][:14]

        # Val targets (3 targets in NE Cluster) -> 6 crops (2 per target)
        for tid in ["TGT001", "TGT002", "TGT003"]:
            t = next(x for x in targets if x["target_id"] == tid)
            positive_crop_plan.append((t, "H11833_1of2.tif", (0, 0)))
            positive_crop_plan.append((t, "H11833_2of2.tif", (24, -24)))

        # Test targets (3 targets in Central East) -> 4 crops
        for tid in ["TGT004", "TGT006"]:
            t = next(x for x in targets if x["target_id"] == tid)
            positive_crop_plan.append((t, "H11833_1of2.tif", (0, 0)))
            positive_crop_plan.append((t, "H11833_2of2.tif", (32, 32)))

    else:
        # Full generation
        for t in targets:
            offsets = [(0, 0), (-32, -32), (32, 32), (-32, 32), (32, -32), (0, -48), (0, 48), (-48, 0), (48, 0)]
            for tname, ds in ds_list:
                row, col = ds.index(t["utm_x"], t["utm_y"])
                if 0 <= col < ds.width and 0 <= row < ds.height:
                    center_ok = has_valid_sonar(ds, col, row)
                    if not center_ok:
                        test_wc = int(round(col - CROP_SIZE / 2.0))
                        test_wr = int(round(row - CROP_SIZE / 2.0))
                        center_ok = has_valid_crop_window(ds, test_wc, test_wr)
                    if center_ok:
                        for off in offsets:
                            positive_crop_plan.append((t, tname, off))

    print(f"Extracting {len(positive_crop_plan)} positive target-centered crops...")

    for t, tiff_name, (off_x, off_y) in positive_crop_plan:
        ds = next(d for name, d in ds_list if name == tiff_name)
        split = t["split"]
        inv_transform = ~ds.transform
        tgt_col, tgt_row = inv_transform * (t["utm_x"], t["utm_y"])

        # Compute shifted crop window ensuring target bounding boxes are fully inside crop
        win_col, win_row = compute_shifted_crop_window(t, ds, off_x, off_y, targets, CROP_SIZE)

        # Primary target dimensions and bbox in crop frame
        t_bw = t["length_m"] / 0.5
        t_bh = t["width_m"] / 0.5
        t_x1 = tgt_col - t_bw / 2.0 - win_col
        t_x2 = tgt_col + t_bw / 2.0 - win_col
        t_y1 = tgt_row - t_bh / 2.0 - win_row
        t_y2 = tgt_row + t_bh / 2.0 - win_row

        # Verify primary target is completely inside [0, CROP_SIZE]
        if not (0.0 <= t_x1 and 0.0 <= t_y1 and t_x2 <= CROP_SIZE and t_y2 <= CROP_SIZE):
            print(f"[REJECT] Primary target {t['target_id']} cannot be fully contained: [{t_x1:.2f}, {t_y1:.2f}, {t_x2:.2f}, {t_y2:.2f}]")
            continue

        # Calculate exact YOLO bounding boxes for all targets that fall within this crop window
        yolo_labels = []
        has_clipped_secondary = False
        split_targets = [cand for cand in targets if cand["split"] == split]

        for cand_t in split_targets:
            c_col, c_row = inv_transform * (cand_t["utm_x"], cand_t["utm_y"])
            c_bw = cand_t["length_m"] / 0.5
            c_bh = cand_t["width_m"] / 0.5
            c_x1 = c_col - c_bw / 2.0 - win_col
            c_x2 = c_col + c_bw / 2.0 - win_col
            c_y1 = c_row - c_bh / 2.0 - win_row
            c_y2 = c_row + c_bh / 2.0 - win_row

            # Check if cand_t intersects the crop window
            if c_x2 > 0 and c_x1 < CROP_SIZE and c_y2 > 0 and c_y1 < CROP_SIZE:
                # Verify cand_t is completely inside [0, CROP_SIZE]
                if not (0.0 <= c_x1 and 0.0 <= c_y1 and c_x2 <= CROP_SIZE and c_y2 <= CROP_SIZE):
                    has_clipped_secondary = True
                    print(f"[REJECT] Secondary target {cand_t['target_id']} clipped in crop for {t['target_id']}: [{c_x1:.2f}, {c_y1:.2f}, {c_x2:.2f}, {c_y2:.2f}]")
                    break

                xc = (c_col - win_col) / float(CROP_SIZE)
                yc = (c_row - win_row) / float(CROP_SIZE)
                bw = c_bw / float(CROP_SIZE)
                bh = c_bh / float(CROP_SIZE)

                yolo_labels.append((CLASS_ID, xc, yc, bw, bh, cand_t, c_col, c_row, c_x1, c_y1, c_x2, c_y2))

        if has_clipped_secondary or len(yolo_labels) == 0:
            continue

        target_crop_counters[t["target_id"]] += 1
        seq = target_crop_counters[t["target_id"]]
        crop_id = f"E3_H11833_{t['target_id']}_{seq:04d}"
        img_filename = f"{crop_id}.png"

        window = Window(win_col, win_row, CROP_SIZE, CROP_SIZE)
        data = ds.read(1, window=window)

        # Save image
        img = Image.fromarray(data)
        img_path = e3_dir / "images" / split / img_filename
        img.save(img_path, format="PNG")

        # Write YOLO label file
        label_path = e3_dir / "labels" / split / f"{crop_id}.txt"
        with open(label_path, "w") as lf:
            for lab in yolo_labels:
                lf.write(f"{lab[0]} {lab[1]:.6f} {lab[2]:.6f} {lab[3]:.6f} {lab[4]:.6f}\n")

        # Metadata record
        crop_metadata_records.append({
            "crop_id": crop_id,
            "image_filename": img_filename,
            "split": split,
            "source_tiff": tiff_name,
            "target_id": t["target_id"],
            "target_latitude": f"{t['lat']:.8f}",
            "target_longitude": f"{t['lon']:.8f}",
            "utm_x": f"{t['utm_x']:.2f}",
            "utm_y": f"{t['utm_y']:.2f}",
            "crop_x": win_col,
            "crop_y": win_row,
            "crop_width": CROP_SIZE,
            "crop_height": CROP_SIZE,
            "is_positive": "true"
        })

        split_manifest_records.append({
            "crop_id": crop_id,
            "split": split,
            "target_id": t["target_id"],
            "cluster": t["cluster"],
            "is_positive": "true"
        })

        # Draw QA visualization image
        draw_qa_image(img, yolo_labels, crop_id, qa_dir, is_positive=True, split=split)

        # Record for numerical QA
        for lab in yolo_labels:
            numerical_qa_records.append({
                "crop_id": crop_id,
                "target_id": lab[5]["target_id"],
                "source_tiff": tiff_name,
                "target_pixel_x": lab[6],
                "target_pixel_y": lab[7],
                "crop_x": win_col,
                "crop_y": win_row,
                "bbox_x_min": lab[8],
                "bbox_y_min": lab[9],
                "bbox_x_max": lab[10],
                "bbox_y_max": lab[11]
            })

    # 2. NEGATIVE / BACKGROUND CROPS GENERATION
    print("Extracting background / negative crops...")
    bg_regions = {
        "train": {
            "x_range": (262000, 265000), "y_range": (3198000, 3203500),
            "count": 8 if pilot_only else 60
        },
        "val": {
            "x_range": (275000, 277500), "y_range": (3207000, 3213000),
            "count": 2 if pilot_only else 20
        },
        "test": {
            "x_range": (268500, 271000), "y_range": (3200000, 3203000),
            "count": 2 if pilot_only else 10
        }
    }

    for split, rcfg in bg_regions.items():
        req_count = rcfg["count"]
        count_per_tiff = (req_count + 1) // 2
        
        for tname, ds in ds_list:
            r_a, c_a = ds.index(rcfg["x_range"][0], rcfg["y_range"][0])
            r_b, c_b = ds.index(rcfg["x_range"][1], rcfg["y_range"][1])
            c_min = max(0, min(c_a, c_b))
            c_max = min(ds.width - CROP_SIZE, max(c_a, c_b))
            r_min = max(0, min(r_a, r_b))
            r_max = min(ds.height - CROP_SIZE, max(r_a, r_b))

            extracted = 0
            for r in range(r_min, r_max, 400):
                if extracted >= count_per_tiff or len([r for r in crop_metadata_records if r["split"] == split and r["is_positive"] == "false"]) >= req_count:
                    break
                for c in range(c_min, c_max, 400):
                    if extracted >= count_per_tiff or len([r for r in crop_metadata_records if r["split"] == split and r["is_positive"] == "false"]) >= req_count:
                        break
                    
                    cx_px = c + CROP_SIZE / 2.0
                    cy_px = r + CROP_SIZE / 2.0
                    center_utm_x, center_utm_y = ds.xy(cy_px, cx_px)

                    # Strict safety margin: >= 450m from ANY verified target
                    dists_to_targets = [np.hypot(center_utm_x - tg["utm_x"], center_utm_y - tg["utm_y"]) for tg in targets]
                    if min(dists_to_targets) < 450.0:
                        continue

                    win = Window(c, r, CROP_SIZE, CROP_SIZE)
                    data = ds.read(1, window=win)
                    nz = np.count_nonzero(data) / data.size
                    mean_val = float(np.mean(data))
                    std_val = float(np.std(data))

                    if nz < 0.90 or mean_val < 20.0 or mean_val > 180.0 or std_val < 6.0:
                        continue

                    bg_counter += 1
                    crop_id = f"E3_H11833_BG_{bg_counter:04d}"
                    img_filename = f"{crop_id}.png"

                    # Save negative image
                    img = Image.fromarray(data)
                    img_path = e3_dir / "images" / split / img_filename
                    img.save(img_path, format="PNG")

                    # Empty label file
                    label_path = e3_dir / "labels" / split / f"{crop_id}.txt"
                    with open(label_path, "w") as lf:
                        pass

                    # Metadata
                    crop_metadata_records.append({
                        "crop_id": crop_id,
                        "image_filename": img_filename,
                        "split": split,
                        "source_tiff": tname,
                        "target_id": "",
                        "target_latitude": "",
                        "target_longitude": "",
                        "utm_x": f"{center_utm_x:.2f}",
                        "utm_y": f"{center_utm_y:.2f}",
                        "crop_x": c,
                        "crop_y": r,
                        "crop_width": CROP_SIZE,
                        "crop_height": CROP_SIZE,
                        "is_positive": "false"
                    })

                    split_manifest_records.append({
                        "crop_id": crop_id,
                        "split": split,
                        "target_id": "",
                        "cluster": f"{split.capitalize()}_Seabed_Background",
                        "is_positive": "false"
                    })

                    # Draw QA image
                    draw_qa_image(img, [], crop_id, qa_dir, is_positive=False, split=split)
                    extracted += 1

    # Close TIFF datasets
    for _, ds in ds_list:
        ds.close()

    # 3. SAVE METADATA CSVs
    crop_meta_path = meta_dir / "crop_metadata.csv"
    with open(crop_meta_path, "w", newline="") as f:
        fieldnames = [
            "crop_id", "image_filename", "split", "source_tiff",
            "target_id", "target_latitude", "target_longitude",
            "utm_x", "utm_y", "crop_x", "crop_y", "crop_width", "crop_height",
            "is_positive"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in crop_metadata_records:
            writer.writerow(rec)
    print(f"[✓] Saved {crop_meta_path} ({len(crop_metadata_records)} records)")

    split_manifest_path = meta_dir / "split_manifest.csv"
    with open(split_manifest_path, "w", newline="") as f:
        fieldnames = ["crop_id", "split", "target_id", "cluster", "is_positive"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in split_manifest_records:
            writer.writerow(rec)
    print(f"[✓] Saved {split_manifest_path} ({len(split_manifest_records)} records)")

    # 4. SAVE data.yaml
    data_yaml_path = e3_dir / "data.yaml"
    yaml_content = f"""# YOLOv8 Dataset Configuration - NOAA H11833 SSS Marine Debris E3 (512x512)
path: {e3_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[✓] Saved {data_yaml_path}")

    # Summary
    pos_count = len([r for r in crop_metadata_records if r["is_positive"] == "true"])
    neg_count = len([r for r in crop_metadata_records if r["is_positive"] == "false"])
    print("\n" + "=" * 60)
    print(f"E3 Generation Complete:")
    print(f"  Positive crops: {pos_count}")
    print(f"  Negative crops: {neg_count}")
    print(f"  Total crops:    {len(crop_metadata_records)}")
    print(f"  QA images:      {len(list(qa_dir.glob('*.png')))}")
    print("=" * 60)

    # Numerical QA Report
    print("\n" + "=" * 145)
    print("NUMERICAL QA REPORT (Positive Crops Bounding Box Bounds):")
    print("=" * 145)
    h_crop = "crop_id"
    h_tgt = "target_id"
    h_tiff = "source_tiff"
    h_px = "target_pixel_x"
    h_py = "target_pixel_y"
    h_cx = "crop_x"
    h_cy = "crop_y"
    h_xmin = "bbox_x_min"
    h_ymin = "bbox_y_min"
    h_xmax = "bbox_x_max"
    h_ymax = "bbox_y_max"
    print(f"| {h_crop:<23} | {h_tgt:<9} | {h_tiff:<15} | {h_px:<14} | {h_py:<14} | {h_cx:<6} | {h_cy:<6} | {h_xmin:<10} | {h_ymin:<10} | {h_xmax:<10} | {h_ymax:<10} |")
    print("=" * 145)

    clipped_count = 0
    for rec in numerical_qa_records:
        cid = rec["crop_id"]
        tid = rec["target_id"]
        tiff = rec["source_tiff"]
        tpx = rec["target_pixel_x"]
        tpy = rec["target_pixel_y"]
        cx = rec["crop_x"]
        cy = rec["crop_y"]
        xmin = rec["bbox_x_min"]
        ymin = rec["bbox_y_min"]
        xmax = rec["bbox_x_max"]
        ymax = rec["bbox_y_max"]
        is_clipped = (xmin < 0 or ymin < 0 or xmax > CROP_SIZE or ymax > CROP_SIZE)
        if is_clipped:
            clipped_count += 1
        print(f"| {cid:<23} | {tid:<9} | {tiff:<15} | {tpx:<14.2f} | {tpy:<14.2f} | {cx:<6} | {cy:<6} | {xmin:<10.2f} | {ymin:<10.2f} | {xmax:<10.2f} | {ymax:<10.2f} |")
    print("=" * 145)
    print(f"Numerical QA Check: {len(numerical_qa_records)} bounding boxes checked. Clipped bboxes: {clipped_count}")
    print("=" * 145 + "\n")

    return pos_count, neg_count


def draw_qa_image(img: Image.Image, yolo_labels: list, crop_id: str, qa_dir: Path, is_positive: bool, split: str):
    """Generates visual QA image with bounding boxes, target info, and crop tags."""
    rgb_img = img.convert("RGB")
    draw = ImageDraw.Draw(rgb_img)

    if is_positive:
        for lab in yolo_labels:
            cid, xc, yc, bw, bh, t = lab[:6]
            px_xc = xc * CROP_SIZE
            px_yc = yc * CROP_SIZE
            px_w = bw * CROP_SIZE
            px_h = bh * CROP_SIZE

            x1 = int(round(px_xc - px_w / 2.0))
            y1 = int(round(px_yc - px_h / 2.0))
            x2 = int(round(px_xc + px_w / 2.0))
            y2 = int(round(px_yc + px_h / 2.0))

            # Draw green bounding box
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)

            # Draw center point crosshair
            draw.line([(px_xc - 4, px_yc), (px_xc + 4, px_yc)], fill="#FF0000", width=1)
            draw.line([(px_xc, px_yc - 4), (px_xc, px_yc + 4)], fill="#FF0000", width=1)

            # Label box
            label_text = f"{t['target_id']}: {CLASS_NAME} ({t['length_m']:.0f}x{t['width_m']:.0f}m)"
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label_text) * 7 + 6, y1], fill="#00AA00")
            draw.text((x1 + 3, max(0, y1 - 16)), label_text, fill="#FFFFFF")

        banner = f"[{split.upper()}] {crop_id} (Target-Centered 512x512)"
        draw.rectangle([0, 0, 512, 18], fill="#222222")
        draw.text((6, 2), banner, fill="#00FF00")
    else:
        banner = f"[{split.upper()}] {crop_id} - Clean Seabed Background"
        draw.rectangle([0, 0, 512, 18], fill="#222222")
        draw.text((6, 2), banner, fill="#00CCFF")

    out_qa_path = qa_dir / f"QA_{crop_id}.png"
    rgb_img.save(out_qa_path, format="PNG")


def main():
    parser = argparse.ArgumentParser(description="Prepare NOAA SSS E3 YOLO Dataset")
    parser.add_argument("--base-dir", type=str, default="/home/ashish/sonar-vision", help="Workspace base directory")
    parser.add_argument("--pilot", action="store_true", default=True, help="Generate pilot dataset")
    parser.add_argument("--full", action="store_true", help="Generate full E3 dataset")
    args = parser.parse_args()

    pilot_mode = not args.full
    generate_e3_dataset(args.base_dir, pilot_only=pilot_mode)


if __name__ == "__main__":
    main()
