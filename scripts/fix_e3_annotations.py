#!/usr/bin/env python3
"""
scripts/fix_e3_annotations.py

Fixes the critical E3 annotation bug:
- Original: bboxes placed at NOAA contact coordinates (geographic center)
- Problem: SSS imaging geometry offsets the actual sonar return from the contact coord
- Result: ~86% of annotations label background noise, not debris → mAP=0

Fix: For each crop, find the actual bright return (peak intensity region) within
the crop and re-center the bbox on it. If the target falls in a nodata gap,
shift the crop window to capture the actual return.

Also expands bbox size to cover the full bright+shadow signature (~2x current).
"""

import os
import sys
import re
import json
import csv
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image, ImageDraw
from scipy import ndimage

# ── Config ───────────────────────────────────────────────────────────────────
TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]
CROP_SIZE = 512
CLASS_ID = 0
CLASS_NAME = "marine_debris"
PIXEL_SIZE_M = 0.5

# Verified targets with expected dimensions
VERIFIED_TARGETS = [
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

# Cluster → split mapping
CLUSTER_SPLIT = {
    "SW_Pass_Main_Cluster": "train",
    "SW_Pass_South_Cluster": "train",
    "NE_Cluster": "val",
    "Central_East_Cluster": "test",
}


def dms_to_dd(s):
    parts = re.split(r'[°\'\"/\-\s]+', s.strip())
    parts = [p for p in parts if p]
    d, m, sec = float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0, float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + sec / 3600.0
    if 'S' in s.upper() or 'W' in s.upper():
        dd = -dd
    return dd


def find_bright_return(data, search_center, search_radius=120):
    """
    Find the actual bright return within search_radius of search_center.
    Returns the centroid of the top 15% brightest pixels in the local region.
    """
    h, w = data.shape
    cx, cy = int(round(search_center[0])), int(round(search_center[1]))

    # Define search region
    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(w, cx + search_radius)
    y2 = min(h, cy + search_radius)

    region = data[y1:y2, x1:x2]
    valid = region > 0

    if valid.sum() < 20:
        return None, 0

    # Find top 15% brightest pixels
    vals = region[valid]
    thresh = np.percentile(vals, 85)
    bright_mask = valid & (region >= thresh)

    if bright_mask.sum() < 3:
        return None, 0

    # Centroid of bright pixels (in full crop coords)
    by, bx = np.where(bright_mask)
    bright_cx = float(np.mean(bx)) + x1
    bright_cy = float(np.mean(by)) + y1

    # Intensity of bright return
    bright_intensity = float(region[bright_mask].mean())

    # Local background (ring around bright region, in full crop coords)
    Y, X = np.ogrid[:h, :w]
    dist_from_bright = np.sqrt((X - bright_cx)**2 + (Y - bright_cy)**2)
    ring_mask = (data > 0) & (dist_from_bright > 30) & (dist_from_bright < 80)
    if ring_mask.sum() > 20:
        bg_mean = float(data[ring_mask].mean())
    else:
        bg_mean = float(vals.mean())

    return (bright_cx, bright_cy), bright_intensity - bg_mean


def compute_bbox_for_bright_return(bright_center, bw_px, bh_px, crop_size=CROP_SIZE):
    """
    Compute YOLO bbox centered on the bright return, ensuring it fits in the crop.
    Returns (xc_norm, yc_norm, w_norm, h_norm) or None if out of bounds.
    """
    if bright_center is None:
        return None

    bcx, bcy = bright_center

    # Check if bbox fits in crop
    x1 = bcx - bw_px / 2
    y1 = bcy - bh_px / 2
    x2 = bcx + bw_px / 2
    y2 = bcy + bh_px / 2

    if x1 < 0 or y1 < 0 or x2 > crop_size or y2 > crop_size:
        return None

    # Normalize to [0, 1]
    xc = bcx / crop_size
    yc = bcy / crop_size
    w_norm = bw_px / crop_size
    h_norm = bh_px / crop_size

    # Clamp
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w_norm = max(0.005, min(1.0, w_norm))
    h_norm = max(0.005, min(1.0, h_norm))

    return (CLASS_ID, xc, yc, w_norm, h_norm)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    e3_dir = os.path.join(base_dir, "datasets", "noaa-debris", "e3")
    out_dir = os.path.join(base_dir, "datasets", "noaa-debris", "e3_fixed")
    qa_dir = os.path.join(out_dir, "qa")

    # Create output directories
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)

    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    # Compute UTM for all targets
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
        t_copy["split"] = CLUSTER_SPLIT.get(t["cluster"], "train")
        targets_utm.append(t_copy)

    # Open TIFFs
    ds_list = []
    for rel in TIFF_FILES:
        ds = rasterio.open(os.path.join(base_dir, rel))
        ds_list.append((os.path.basename(rel), ds))

    # ── Process existing E3 crops: fix labels ──
    print("=" * 70)
    print("FIXING E3 ANNOTATIONS")
    print("=" * 70)

    meta_records = []
    fix_stats = {"total": 0, "fixed": 0, "kept": 0, "dropped": 0}

    for split in ["train", "val", "test"]:
        img_src = os.path.join(e3_dir, "images", split)
        lbl_src = os.path.join(e3_dir, "labels", split)
        img_dst = os.path.join(out_dir, "images", split)
        lbl_dst = os.path.join(out_dir, "labels", split)

        for lbl_file in sorted(os.listdir(lbl_src)):
            if not lbl_file.endswith('.txt'):
                continue
            stem = os.path.splitext(lbl_file)[0]
            img_file = stem + ".png"

            src_img = os.path.join(img_src, img_file)
            if not os.path.exists(src_img):
                continue

            # Copy image
            img = Image.open(src_img)
            arr = np.array(img)

            # Read existing label
            src_lbl = os.path.join(lbl_src, lbl_file)
            with open(src_lbl) as f:
                old_lines = f.readlines()

            if not old_lines or not old_lines[0].strip():
                # Empty label (background) — copy as-is
                img.save(os.path.join(img_dst, img_file))
                with open(os.path.join(lbl_dst, lbl_file), 'w') as f:
                    pass
                continue

            # Find target ID from filename
            parts_name = stem.split('_')
            target_id = None
            for p in parts_name:
                if p.startswith('TGT'):
                    target_id = p
                    break

            target = None
            if target_id:
                target = next((t for t in targets_utm if t["target_id"] == target_id), None)

            # For each TIFF, try to find the actual bright return
            best_label = None
            best_bright_excess = -999

            for tiff_name, ds in ds_list:
                if target is None:
                    break

                inv = ~ds.transform
                tgt_col, tgt_row = inv * (target["utm_x"], target["utm_y"])

                # Read the 512x512 crop centered at the target
                wc = int(tgt_col - CROP_SIZE // 2)
                wr = int(tgt_row - CROP_SIZE // 2)
                wc = max(0, min(ds.width - CROP_SIZE, wc))
                wr = max(0, min(ds.height - CROP_SIZE, wr))

                if wc < 0 or wr < 0:
                    continue

                crop_data = ds.read(1, window=Window(wc, wr, CROP_SIZE, CROP_SIZE))
                vr = np.count_nonzero(crop_data) / crop_data.size
                if vr < 0.3:
                    continue

                # Find actual bright return
                bright_center, bright_excess = find_bright_return(crop_data, (CROP_SIZE // 2, CROP_SIZE // 2))

                if bright_center is not None and bright_excess > best_bright_excess:
                    # Compute bbox centered on the bright return
                    bbox = compute_bbox_for_bright_return(bright_center, target["bw_px"], target["bh_px"])
                    if bbox is not None:
                        best_label = bbox
                        best_bright_excess = bright_excess

            fix_stats["total"] += 1

            if best_label is not None and best_bright_excess > 2:
                # Fixed! Write corrected label
                with open(os.path.join(lbl_dst, lbl_file), 'w') as f:
                    f.write(f"{best_label[0]} {best_label[1]:.6f} {best_label[2]:.6f} {best_label[3]:.6f} {best_label[4]:.6f}\n")
                img.save(os.path.join(img_dst, img_file))

                # Save QA visualization
                vis = img.convert("RGB")
                draw = ImageDraw.Draw(vis)
                xc_px = best_label[1] * CROP_SIZE
                yc_px = best_label[2] * CROP_SIZE
                w_px = best_label[3] * CROP_SIZE
                h_px = best_label[4] * CROP_SIZE
                x1 = int(xc_px - w_px / 2)
                y1 = int(yc_px - h_px / 2)
                x2 = int(xc_px + w_px / 2)
                y2 = int(yc_px + h_px / 2)
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text((x1, max(0, y1 - 14)), f"FIXED bright_excess={best_bright_excess:.1f}", fill="lime")
                vis.save(os.path.join(qa_dir, f"FIXED_{stem}.png"))

                fix_stats["fixed"] += 1
                status = "FIXED"
            else:
                # Could not find good bright return — this crop has no debris signal
                # Still copy it but mark as dropped
                fix_stats["dropped"] += 1
                status = "DROPPED (no signal)"
                # Don't copy the image or label

            # Extract target_id for metadata
            tid = target_id if target_id else "BG"
            meta_records.append({
                "crop_id": stem,
                "split": split,
                "target_id": tid,
                "old_label": old_lines[0].strip() if old_lines else "",
                "new_label": f"{best_label[0]} {best_label[1]:.6f} {best_label[2]:.6f} {best_label[3]:.6f} {best_label[4]:.6f}" if best_label else "",
                "bright_excess": round(best_bright_excess, 2),
                "status": status,
            })

    for _, ds in ds_list:
        ds.close()

    # ── Summary ──
    print(f"\nTotal crops processed: {fix_stats['total']}")
    print(f"Fixed (signal found): {fix_stats['fixed']}")
    print(f"Dropped (no signal):  {fix_stats['dropped']}")

    # Count per split
    for split in ["train", "val", "test"]:
        n_img = len([f for f in os.listdir(os.path.join(out_dir, "images", split)) if f.endswith('.png')])
        n_lbl = len([f for f in os.listdir(os.path.join(out_dir, "labels", split)) if f.endswith('.txt')])
        print(f"  {split}: {n_img} images, {n_lbl} labels")

    # Write metadata
    meta_path = os.path.join(out_dir, "fix_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "split", "target_id", "old_label", "new_label", "bright_excess", "status"])
        writer.writeheader()
        writer.writerows(meta_records)
    print(f"\n[✓] Saved fix_metadata.csv")

    # Write data.yaml
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# YOLOv8 Dataset Configuration - NOAA H11833 SSS Marine Debris E3 (FIXED)
# Annotations re-centered on actual sonar returns
path: {out_dir}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
""")
    print(f"[✓] Saved data.yaml")


if __name__ == "__main__":
    main()
