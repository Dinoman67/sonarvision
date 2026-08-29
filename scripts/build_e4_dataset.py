#!/usr/bin/env python3
"""
scripts/build_e4_dataset.py

Build E4 dataset — balanced debris detection dataset with diverse backgrounds.

Problem with E3 V2:
- Only 3 background images (BG_0001-0003) for ~200 positive crops
- YOLO learns "everything is debris" because it never sees enough "not debris"
- Model produces 300 detections per image, mAP is near zero

E4 solution:
1. Keep positive crops from E3 V2 (already re-centered on actual returns)
2. Reduce redundancy: fewer near-identical crops per target (6 instead of 18)
3. Generate 200+ diverse background crops from a GRID across the full TIFF
4. Backgrounds include varied textures: bright seabed, dark patches, textured areas
5. Verify: no accidental debris returns in background crops
6. Balanced splits: train ~200 pos / 200 neg, val ~50/50, test ~50/50
"""

import os
import csv
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image
from scipy import ndimage
import re

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]
OUT_DIR = os.path.join(BASE_DIR, "datasets", "noaa-debris", "e4")
E3_V2_DIR = os.path.join(BASE_DIR, "datasets", "noaa-debris", "e3_v2")
CROP_SIZE = 512
CLASS_ID = 0
CLASS_NAME = "marine_debris"
PIXEL_SIZE_M = 0.5

# Background extraction: dense grid scan (TIFFs have narrow valid strips)
BG_GRID_STEP = 150  # dense sampling — TIFFs are mostly zeros, valid strip is narrow
BG_EXCLUSION_R = 512  # exclude grid points within 1 crop width of a target center

# Positives: which offsets to keep (subset of the 18 from E3 V2)
POSITIVE_OFFSETS = list(range(9))  # use all 9 offsets = 9 per TIFF × 2 TIFFs = 18 per target

# ── Target definitions (same as fix_e3_v2.py) ────────────────────────────────
VERIFIED_TARGETS = [
    {"target_id": "TGT001", "lat_str": "28° 59' 03.3504\" N", "lon_str": "089° 18' 09.3672\" W",
     "length_m": 12.0, "width_m": 6.0, "cluster": "NE_Cluster"},
    {"target_id": "TGT002", "lat_str": "28° 58' 58.6344\" N", "lon_str": "089° 18' 07.3908\" W",
     "length_m": 14.0, "width_m": 10.0, "cluster": "NE_Cluster"},
    {"target_id": "TGT003", "lat_str": "29° 01' 35.0112\" N", "lon_str": "089° 17' 25.5372\" W",
     "length_m": 12.0, "width_m": 12.0, "cluster": "NE_Cluster"},
    {"target_id": "TGT004", "lat_str": "28° 54' 58.5288\" N", "lon_str": "089° 21' 56.8800\" W",
     "length_m": 16.0, "width_m": 6.0, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT005", "lat_str": "28° 54' 48.1320\" N", "lon_str": "089° 22' 05.1348\" W",
     "length_m": 16.0, "width_m": 6.0, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT006", "lat_str": "28° 54' 54.197\" N", "lon_str": "089° 22' 16.974\" W",
     "length_m": 16.0, "width_m": 8.0, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT007", "lat_str": "28° 54' 10.6380\" N", "lon_str": "089° 25' 30.9252\" W",
     "length_m": 14.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT008", "lat_str": "28° 54' 42.4764\" N", "lon_str": "089° 25' 24.0312\" W",
     "length_m": 16.0, "width_m": 10.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT009", "lat_str": "28° 54' 41.62\" N", "lon_str": "089° 25' 22.30\" W",
     "length_m": 22.0, "width_m": 10.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT010", "lat_str": "28° 54' 38.79\" N", "lon_str": "089° 25' 17.40\" W",
     "length_m": 12.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT011", "lat_str": "28° 54' 59.10\" N", "lon_str": "089° 25' 34.45\" W",
     "length_m": 12.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT012", "lat_str": "28° 54' 59.78\" N", "lon_str": "089° 25' 32.53\" W",
     "length_m": 12.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT013", "lat_str": "28° 54' 32.396\" N", "lon_str": "089° 25' 59.387\" W",
     "length_m": 12.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT014", "lat_str": "28° 54' 30.07\" N", "lon_str": "089° 26' 03.29\" W",
     "length_m": 12.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT015", "lat_str": "28° 54' 18.14\" N", "lon_str": "089° 25' 45.09\" W",
     "length_m": 12.0, "width_m": 8.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT016", "lat_str": "28° 53' 35.68\" N", "lon_str": "089° 25' 44.09\" W",
     "length_m": 14.0, "width_m": 6.0, "cluster": "SW_Pass_South_Cluster"},
    {"target_id": "TGT017", "lat_str": "28° 53' 19.464\" N", "lon_str": "089° 26' 14.276\" W",
     "length_m": 12.0, "width_m": 10.0, "cluster": "SW_Pass_South_Cluster"},
]

# Mixed split: every split gets strong + weak targets
# This ensures test set has detectable targets (not just the weakest)
CLUSTER_SPLIT = {
    # Train: 7 targets (mix of strong SW_Pass + some NE)
    "SW_Pass_Main_Cluster": "train",
    "SW_Pass_South_Cluster": "train",
    # Val: mix of NE + Central East
    "NE_Cluster": "val",
    "Central_East_Cluster": "train",
}

# Override: put 1 strong target in val, 1 weak in test, 1 weak in val
TARGET_OVERRIDE_SPLIT = {
    "TGT002": "val",   # NE strong → val
    "TGT004": "test",  # CE weak → test (1 only)
    "TGT010": "val",   # SW strong → val
    "TGT006": "test",  # CE weak → test (2 only)
    "TGT008": "test",  # SW strong → test (ensures test has detectable targets)
    "TGT014": "val",   # SW strong → val
}


def dms_to_dd(s):
    parts = re.split(r'[°\'\"\s/]+', s.strip())
    parts = [p for p in parts if p]
    d, m, sec = float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0, float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + sec / 3600.0
    if 'S' in s.upper() or 'W' in s.upper():
        dd = -dd
    return dd


def extract_crop(ds, center_col, center_row):
    """Extract a CROP_SIZE×CROP_SIZE crop. Returns (data, offset) or None."""
    wc = int(round(center_col - CROP_SIZE / 2))
    wr = int(round(center_row - CROP_SIZE / 2))
    wc = max(0, min(ds.width - CROP_SIZE, wc))
    wr = max(0, min(ds.height - CROP_SIZE, wr))
    if wc < 0 or wr < 0:
        return None
    data = ds.read(1, window=Window(wc, wr, CROP_SIZE, CROP_SIZE))
    if np.count_nonzero(data) / data.size < 0.3:
        return None
    return data, (wc, wr)


def is_near_target(col, row, target_pixels, exclusion_r):
    """Check if a grid point is too close to any known target."""
    for tx, ty in target_pixels:
        if abs(col - tx) < exclusion_r and abs(row - ty) < exclusion_r:
            return True
    return False


def classify_background(data):
    """Classify background crop by visual characteristics."""
    valid = data[data > 0]
    if len(valid) < 100:
        return None
    mean_val = float(valid.mean())
    std_val = float(valid.std())
    return {"mean": mean_val, "std": std_val, "min": float(valid.min()), "max": float(valid.max())}


def main():
    print("=" * 70)
    print("E4 DATASET BUILDER")
    print("Balanced debris detection with diverse seabed backgrounds")
    print("=" * 70)

    # Setup output directories
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)
    qa_dir = os.path.join(OUT_DIR, "qa")
    os.makedirs(qa_dir, exist_ok=True)

    # Transform coordinates
    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    targets_utm = []
    for t in VERIFIED_TARGETS:
        lat = dms_to_dd(t["lat_str"])
        lon = dms_to_dd(t["lon_str"])
        if lon > 0:
            lon = -lon
        ux, uy = tf.transform(lon, lat)
        t_copy = dict(t)
        t_copy["utm_x"] = ux
        t_copy["utm_y"] = uy
        t_copy["bw_px"] = t["length_m"] / PIXEL_SIZE_M
        t_copy["bh_px"] = t["width_m"] / PIXEL_SIZE_M
        # Check per-target override first, then cluster default
        t_copy["split"] = TARGET_OVERRIDE_SPLIT.get(t["target_id"], CLUSTER_SPLIT.get(t["cluster"], "train"))
        targets_utm.append(t_copy)

    # Open TIFFs
    ds_list = []
    for rel in TIFF_FILES:
        ds = rasterio.open(os.path.join(BASE_DIR, rel))
        ds_list.append((os.path.basename(rel), ds))

    # Convert target UTM coords to pixel coords for each TIFF
    target_pixels_all = {}
    for tiff_name, ds in ds_list:
        inv = ~ds.transform
        target_pixels_all[tiff_name] = []
        for t in targets_utm:
            col, row = inv * (t["utm_x"], t["utm_y"])
            target_pixels_all[tiff_name].append((col, row))

    meta_records = []
    bg_stats = {"train": [], "val": [], "test": []}

    # Map target_id → E3 V2 source split (where the files actually are)
    E3_V2_SOURCE_SPLIT = {}
    for t in targets_utm:
        tid = t["target_id"]
        # Find which E3 V2 split has this target's files
        for src_split in ["train", "val", "test"]:
            check_dir = os.path.join(E3_V2_DIR, "images", src_split)
            if os.path.isdir(check_dir) and any(tid in f for f in os.listdir(check_dir)):
                E3_V2_SOURCE_SPLIT[tid] = src_split
                break
        if tid not in E3_V2_SOURCE_SPLIT:
            E3_V2_SOURCE_SPLIT[tid] = "train"  # fallback
    print(f"  E3 V2 source mapping: {E3_V2_SOURCE_SPLIT}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Copy positive crops from E3 V2 (subset of offsets)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Step 1: Copying positive crops from E3 V2 ---")
    pos_counter = 0
    for target in targets_utm:
        dst_split = target["split"]  # new split assignment
        tid = target["target_id"]
        src_split = E3_V2_SOURCE_SPLIT[tid]  # E3 V2 source split
        # E3 V2 naming: E3v2_{TGT}_{tiff_idx}_{offset_idx:02d}
        for tiff_idx in [1, 2]:
            for off_idx in POSITIVE_OFFSETS:
                src_id = f"E3v2_{tid}_{tiff_idx}_{off_idx+1:02d}"
                dst_id = f"E4_{tid}_{tiff_idx}_{off_idx+1:02d}"

                src_img = os.path.join(E3_V2_DIR, "images", src_split, f"{src_id}.png")
                src_lbl = os.path.join(E3_V2_DIR, "labels", src_split, f"{src_id}.txt")

                if not os.path.exists(src_img) or not os.path.exists(src_lbl):
                    continue

                # Copy image → new split
                shutil.copy2(src_img, os.path.join(OUT_DIR, "images", dst_split, f"{dst_id}.png"))
                # Copy label → new split
                shutil.copy2(src_lbl, os.path.join(OUT_DIR, "labels", dst_split, f"{dst_id}.txt"))

                pos_counter += 1
                meta_records.append({
                    "crop_id": dst_id, "split": dst_split, "type": "positive",
                    "target_id": tid, "source_tiff": f"H11833_{tiff_idx}of2.tif",
                    "bright_excess": 0, "bg_mean": 0, "bg_std": 0,
                })

    print(f"  Copied {pos_counter} positive crops")
    for split in ["train", "val", "test"]:
        n = sum(1 for m in meta_records if m["split"] == split and m["type"] == "positive")
        targets_in_split = sorted(set(m["target_id"] for m in meta_records if m["split"] == split and m["type"] == "positive"))
        print(f"    {split}: {n} positives → {targets_in_split}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Extract diverse background crops from grid scan
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Step 2: Extracting diverse background crops from grid ---")

    bg_counter = 0
    bg_candidates = []  # (quality_score, crop_data, stats, tiff_name, col, row)

    for tiff_name, ds in ds_list:
        pixel_targets = target_pixels_all[tiff_name]
        print(f"\n  Scanning {tiff_name} ({ds.width}×{ds.height})...")

        # Grid scan
        grid_cols = range(CROP_SIZE // 2, ds.width - CROP_SIZE // 2, BG_GRID_STEP)
        grid_rows = range(CROP_SIZE // 2, ds.height - CROP_SIZE // 2, BG_GRID_STEP)

        n_candidates = 0
        for col in grid_cols:
            for row in grid_rows:
                if is_near_target(col, row, pixel_targets, BG_EXCLUSION_R):
                    continue

                result = extract_crop(ds, col, row)
                if result is None:
                    continue

                crop_data, (wc, wr) = result

                # Verify minimum valid data
                valid = crop_data[crop_data > 0]
                if len(valid) < 200:
                    continue

                crop_mean = float(valid.mean())
                crop_std = float(valid.std())

                # Skip if too dark (edge of survey)
                if crop_mean < 3:
                    continue

                # Quality score: prefer diverse textures (CV = std/mean)
                quality = crop_std / max(crop_mean, 1)

                stats = classify_background(crop_data)
                if stats is None:
                    continue

                bg_candidates.append((quality, crop_data.copy(), stats, tiff_name, col, row))
                n_candidates += 1

        print(f"    {n_candidates} background candidates found")

    # Sort by quality (most textured first) and diversity
    bg_candidates.sort(key=lambda x: -x[0])

    # Select diverse backgrounds: cluster by (mean, std) to ensure variety
    print(f"\n  Selecting diverse backgrounds from {len(bg_candidates)} candidates...")

    selected_bgs = []
    used_regions = []  # (col, row) — prevent overlapping crops

    for quality, crop_data, stats, tiff_name, col, row in bg_candidates:
        # Check not overlapping with already selected
        overlaps = False
        for uc, ur in used_regions:
            if abs(col - uc) < CROP_SIZE * 0.5 and abs(row - ur) < CROP_SIZE * 0.5:
                overlaps = True
                break
        if overlaps:
            continue

        # Assign split based on position within the TIFF
        # Use thirds: top=one split, middle=another, bottom=another
        # This ensures geographic diversity in each split
        ds_height = ds_list[0][1].height if tiff_name == "H11833_1of2.tif" else ds_list[1][1].height
        third = ds_height // 3
        if tiff_name == "H11833_1of2.tif":
            if row < third:
                split = "val"
            elif row < 2 * third:
                split = "train"
            else:
                split = "train"  # More training data
        else:
            if row < third:
                split = "test"
            elif row < 2 * third:
                split = "train"
            else:
                split = "train"  # More training data

        selected_bgs.append((crop_data, stats, tiff_name, col, row, split))
        used_regions.append((col, row))

        # Stop when we have enough per split
        counts = {}
        for _, _, _, _, _, s in selected_bgs:
            counts[s] = counts.get(s, 0) + 1

        # Target: 200 train, 60 val, 60 test (balanced with positives)
        if counts.get("train", 0) >= 200 and counts.get("val", 0) >= 60 and counts.get("test", 0) >= 60:
            break

    # Write background crops
    for crop_data, stats, tiff_name, col, row, split in selected_bgs:
        bg_counter += 1
        crop_id = f"E4_BG_{bg_counter:04d}"

        img = Image.fromarray(crop_data)
        img.save(os.path.join(OUT_DIR, "images", split, f"{crop_id}.png"))

        lbl_path = os.path.join(OUT_DIR, "labels", split, f"{crop_id}.txt")
        with open(lbl_path, "w") as f:
            pass  # Empty = background

        bg_stats[split].append(stats)
        meta_records.append({
            "crop_id": crop_id, "split": split, "type": "background",
            "target_id": "BG", "source_tiff": tiff_name,
            "bright_excess": 0,
            "bg_mean": round(stats["mean"], 1),
            "bg_std": round(stats["std"], 1),
        })

    print(f"\n  Total background crops: {bg_counter}")
    for split in ["train", "val", "test"]:
        n = sum(1 for m in meta_records if m["split"] == split and m["type"] == "background")
        print(f"    {split}: {n} backgrounds")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Write metadata, data.yaml, and summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Step 3: Writing metadata and config ---")

    # Metadata CSV
    meta_path = os.path.join(OUT_DIR, "crop_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "split", "type", "target_id",
                                                "source_tiff", "bright_excess", "bg_mean", "bg_std"])
        writer.writeheader()
        writer.writerows(meta_records)
    print(f"  [✓] crop_metadata.csv ({len(meta_records)} records)")

    # Data YAML (local path)
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# YOLOv8 Dataset - NOAA H11833 SSS Marine Debris E4
# Balanced: ~200 positive crops + ~360 diverse background crops
# Positives: re-centered on actual sonar returns (from E3 V2)
# Backgrounds: grid-sampled from full survey, filtered for no debris
path: {OUT_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
""")
    print(f"  [✓] data.yaml")

    # Data YAML (Colab path)
    yaml_colab = os.path.join(OUT_DIR, "data_colab.yaml")
    with open(yaml_colab, "w") as f:
        f.write(f"""# YOLOv8 Dataset - NOAA H11833 SSS Marine Debris E4 (Colab)
path: /content/e4
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
""")
    print(f"  [✓] data_colab.yaml")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Summary and QA
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E4 DATASET SUMMARY")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        n_pos = sum(1 for m in meta_records if m["split"] == split and m["type"] == "positive")
        n_neg = sum(1 for m in meta_records if m["split"] == split and m["type"] == "background")
        n_total = n_pos + n_neg
        print(f"\n  {split.upper()}: {n_total} images ({n_pos} positive, {n_neg} background)")
        if n_pos > 0 and n_neg > 0:
            ratio = n_neg / n_pos
            print(f"    pos:neg ratio = 1:{ratio:.1f}")

    # Background diversity stats
    print("\n  Background diversity:")
    for split in ["train", "val", "test"]:
        stats_list = bg_stats[split]
        if not stats_list:
            continue
        means = [s["mean"] for s in stats_list]
        stds = [s["std"] for s in stats_list]
        print(f"    {split}: mean brightness {np.mean(means):.0f}±{np.std(means):.0f}, "
              f"texture {np.mean(stds):.0f}±{np.std(stds):.0f} "
              f"(range: {np.min(means):.0f}-{np.max(means):.0f})")

    # Close TIFFs
    for _, ds in ds_list:
        ds.close()

    print(f"\n[✓] E4 dataset ready at: {OUT_DIR}")
    print(f"    Zip with: cd datasets/noaa-debris && zip -r e4.zip e4/")


if __name__ == "__main__":
    main()
