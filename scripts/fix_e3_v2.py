#!/usr/bin/env python3
"""
scripts/fix_e3_v2.py

Root cause fix: Re-center CROPS on actual sonar returns, not NOAA contact coords.

The E3 bug:
- Crops are centered at NOAA contact coordinates (lat/lon)
- The actual SSS acoustic return is offset 50-150px due to imaging geometry
- Result: bbox labels background noise → mAP = 0

This script:
1. Opens raw TIFFs
2. For each target, searches a wide area for the actual bright return
3. Re-centers the 512x512 crop on the bright return
4. Places the bbox at the bright return location
5. Maintains same split assignments and dataset structure
"""

import os
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
# Search radius around contact coord to find actual return (pixels)
WIDE_SEARCH_R = 300

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


def find_actual_return_in_tiff(ds, contact_col, contact_row, search_r, bw_px, bh_px):
    """
    Search a wide area around the contact coordinate in the raw TIFF to find
    the actual bright return. Returns (return_col, return_row, bright_excess).
    """
    h, y_size = ds.height, ds.width

    # Read search region
    x1 = max(0, int(contact_col) - search_r)
    y1 = max(0, int(contact_row) - search_r)
    x2 = min(ds.width, int(contact_col) + search_r)
    y2 = min(ds.height, int(contact_row) + search_r)

    if x2 - x1 < 50 or y2 - y1 < 50:
        return None, 0

    data = ds.read(1, window=Window(x1, y1, x2 - x1, y2 - y1))

    # Find valid data
    valid = data > 0
    if valid.sum() < 100:
        return None, 0

    # Compute local background in the search region (outside target area)
    # Ring mask: not too close to center (where target might be), not too far
    region_h, region_w = data.shape
    rcx = (x2 - x1) / 2.0
    rcy = (y2 - y1) / 2.0
    Y, X = np.ogrid[:region_h, :region_w]
    dist_from_center = np.sqrt((X - rcx)**2 + (Y - rcy)**2)
    bg_mask = valid & (dist_from_center > bw_px) & (dist_from_center < search_r * 0.7)
    if bg_mask.sum() < 50:
        return None, 0

    bg_mean = float(data[bg_mask].mean())
    bg_std = float(data[bg_mask].std())
    if bg_std < 2:
        return None, 0

    # Find bright return: threshold at mean + 1.5*std
    bright_thresh = bg_mean + 1.5 * bg_std
    bright_mask = valid & (data > bright_thresh)

    if bright_mask.sum() < 5:
        return None, 0

    # Find the largest bright cluster (likely the debris return)
    labeled, nf = ndimage.label(bright_mask)
    if nf == 0:
        return None, 0

    sizes = ndimage.sum(bright_mask, labeled, range(1, nf + 1))
    largest_label = np.argmax(sizes) + 1
    component = labeled == largest_label

    # Centroid of the bright cluster (in region-local coords)
    cy, cx = np.where(component)
    if len(cx) < 3:
        return None, 0

    bright_cx_local = float(np.mean(cx))
    bright_cy_local = float(np.mean(cy))

    # Convert to absolute TIFF coords
    bright_cx_abs = bright_cx_local + x1
    bright_cy_abs = bright_cy_local + y1

    # Bright excess of this cluster
    cluster_vals = data[component]
    bright_excess = float(cluster_vals.mean() - bg_mean)

    return (bright_cx_abs, bright_cy_abs), bright_excess


def extract_crop_at(ds, center_col, center_row, crop_size=CROP_SIZE):
    """Extract a crop_size x crop_size crop centered at (center_col, center_row)."""
    wc = int(round(center_col - crop_size / 2))
    wr = int(round(center_row - crop_size / 2))
    wc = max(0, min(ds.width - crop_size, wc))
    wr = max(0, min(ds.height - crop_size, wr))

    if wc < 0 or wr < 0:
        return None, None

    data = ds.read(1, window=Window(wc, wr, crop_size, crop_size))
    valid_ratio = np.count_nonzero(data) / data.size
    if valid_ratio < 0.3:
        return None, None

    return data, (wc, wr)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    e3_dir = os.path.join(base_dir, "datasets", "noaa-debris", "e3")
    out_dir = os.path.join(base_dir, "datasets", "noaa-debris", "e3_v2")
    qa_dir = os.path.join(out_dir, "qa")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)

    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    # Compute UTM
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

    print("=" * 70)
    print("E3 V2: Re-centering crops on actual sonar returns")
    print("=" * 70)

    crop_counter = 0
    meta_records = []
    target_counters = {t["target_id"]: 0 for t in targets_utm}
    bg_counter = 0

    # ── Process each target ──
    for target in targets_utm:
        split = target["split"]
        bw_px = target["bw_px"]
        bh_px = target["bh_px"]

        for tiff_idx, (tiff_name, ds) in enumerate(ds_list):
            inv = ~ds.transform
            contact_col, contact_row = inv * (target["utm_x"], target["utm_y"])

            # Find actual bright return
            result, bright_excess = find_actual_return_in_tiff(
                ds, contact_col, contact_row, WIDE_SEARCH_R, bw_px, bh_px
            )

            if result is None:
                continue

            return_col, return_row = result

            # Generate multiple crops with offsets around the actual return
            offsets = [(0, 0), (-48, -48), (48, 48), (-48, 48), (48, -48),
                       (-96, 0), (96, 0), (0, -96), (0, 96)]

            for off_idx, (off_x, off_y) in enumerate(offsets):
                # Crop centered on actual return + offset
                crop_col = return_col + off_x
                crop_row = return_row + off_y

                crop_data, (wc, wr) = extract_crop_at(ds, crop_col, crop_row)
                if crop_data is None:
                    continue

                # Compute bbox in crop frame
                bcx = return_col - wc  # bright return position in crop
                bcy = return_row - wr

                # Check bbox fits in crop
                x1 = bcx - bw_px / 2
                y1 = bcy - bh_px / 2
                x2 = bcx + bw_px / 2
                y2 = bcy + bh_px / 2

                if x1 < 0 or y1 < 0 or x2 > CROP_SIZE or y2 > CROP_SIZE:
                    continue

                # Normalize
                xc_norm = bcx / CROP_SIZE
                yc_norm = bcy / CROP_SIZE
                w_norm = bw_px / CROP_SIZE
                h_norm = bh_px / CROP_SIZE

                # Verify: check if bbox region has actual signal
                bbox_region = crop_data[max(0, int(y1)):min(CROP_SIZE, int(y2)),
                                        max(0, int(x1)):min(CROP_SIZE, int(x2))]
                local_ctx = crop_data[crop_data > 0]
                if len(local_ctx) == 0 or bbox_region.size == 0:
                    continue

                bbox_mean = float(bbox_region.mean())
                ctx_mean = float(local_ctx.mean())

                if bbox_mean < ctx_mean * 0.8:
                    # Bbox is darker than background — skip this offset
                    continue

                # Write crop
                crop_counter += 1
                target_counters[target["target_id"]] += 1
                crop_id = f"E3v2_{target['target_id']}_{tiff_idx+1}_{off_idx+1:02d}"

                img = Image.fromarray(crop_data)
                img_path = os.path.join(out_dir, "images", split, f"{crop_id}.png")
                img.save(img_path, format="PNG")

                lbl_path = os.path.join(out_dir, "labels", split, f"{crop_id}.txt")
                with open(lbl_path, "w") as f:
                    f.write(f"{CLASS_ID} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                # QA visualization
                vis = img.convert("RGB")
                draw = ImageDraw.Draw(vis)
                px_x1, px_y1 = int(xc_norm * CROP_SIZE - w_norm * CROP_SIZE / 2), int(yc_norm * CROP_SIZE - h_norm * CROP_SIZE / 2)
                px_x2, px_y2 = int(xc_norm * CROP_SIZE + w_norm * CROP_SIZE / 2), int(yc_norm * CROP_SIZE + h_norm * CROP_SIZE / 2)
                draw.rectangle([px_x1, px_y1, px_x2, px_y2], outline="lime", width=2)
                draw.text((px_x1, max(0, px_y1 - 14)),
                         f"{target['target_id']} excess={bright_excess:.1f}", fill="lime")
                vis.save(os.path.join(qa_dir, f"QA_{crop_id}.png"))

                meta_records.append({
                    "crop_id": crop_id,
                    "split": split,
                    "target_id": target["target_id"],
                    "source_tiff": tiff_name,
                    "offset": (off_x, off_y),
                    "bright_excess": round(bright_excess, 2),
                    "bbox_signal": round(float(bbox_mean / max(ctx_mean, 1)), 2),
                })

    # ── Background crops (no targets) ──
    print("\nExtracting background crops...")
    bg_regions = [
        ("train", "H11833_1of2.tif", 262500, 3199000),
        ("train", "H11833_1of2.tif", 264500, 3201500),
        ("train", "H11833_2of2.tif", 263500, 3199500),
        ("train", "H11833_2of2.tif", 264000, 3202000),
        ("val", "H11833_1of2.tif", 276000, 3210000),
        ("val", "H11833_2of2.tif", 277000, 3209000),
        ("test", "H11833_1of2.tif", 269500, 3201000),
    ]

    for split, tiff_name, utm_x, utm_y in bg_regions:
        ds = next((d for n, d in ds_list if n == tiff_name), None)
        if ds is None:
            continue

        inv = ~ds.transform
        col, row = inv * (utm_x, utm_y)

        result = extract_crop_at(ds, col, row)
        if result is None or result[0] is None:
            continue
        crop_data, (wc, wr) = result

        # Verify no target nearby
        has_target_nearby = False
        for t in targets_utm:
            t_col, t_row = inv * (t["utm_x"], t["utm_y"])
            if abs(t_col - col) < CROP_SIZE and abs(t_row - row) < CROP_SIZE:
                has_target_nearby = True
                break
        if has_target_nearby:
            continue

        bg_counter += 1
        crop_id = f"E3v2_BG_{bg_counter:04d}"

        img = Image.fromarray(crop_data)
        img_path = os.path.join(out_dir, "images", split, f"{crop_id}.png")
        img.save(img_path, format="PNG")

        lbl_path = os.path.join(out_dir, "labels", split, f"{crop_id}.txt")
        with open(lbl_path, "w") as f:
            pass  # Empty label = background

        meta_records.append({
            "crop_id": crop_id,
            "split": split,
            "target_id": "BG",
            "source_tiff": tiff_name,
            "offset": (0, 0),
            "bright_excess": 0,
            "bbox_signal": 0,
        })

    # Close TIFFs
    for _, ds in ds_list:
        ds.close()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"E3 V2 DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total positive crops: {crop_counter}")
    print(f"Total background crops: {bg_counter}")
    print(f"\nPer-target crops:")
    for tid, count in sorted(target_counters.items()):
        print(f"  {tid}: {count}")

    for split in ["train", "val", "test"]:
        n_img = len([f for f in os.listdir(os.path.join(out_dir, "images", split)) if f.endswith('.png')])
        n_lbl = len([f for f in os.listdir(os.path.join(out_dir, "labels", split)) if f.endswith('.txt')])
        n_pos = sum(1 for f in os.listdir(os.path.join(out_dir, "labels", split))
                    if f.endswith('.txt') and os.path.getsize(os.path.join(out_dir, "labels", split, f)) > 0)
        print(f"\n  {split}: {n_img} images, {n_pos} positive, {n_lbl - n_pos} negative")

    # Metadata
    meta_path = os.path.join(out_dir, "crop_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "split", "target_id", "source_tiff",
                                                "offset", "bright_excess", "bbox_signal"])
        writer.writeheader()
        writer.writerows(meta_records)
    print(f"\n[✓] Saved crop_metadata.csv ({len(meta_records)} records)")

    # Data YAML
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# YOLOv8 Dataset - NOAA H11833 SSS Marine Debris E3 V2
# Crops re-centered on actual sonar returns (not contact coordinates)
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
