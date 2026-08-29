#!/usr/bin/env python3
"""
Generate hard negative backgrounds using WINDOWED reading (no OOM).
Scans the valid strip in blocks, finds bright clusters, extracts crops.
"""

import os
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image
from scipy.ndimage import uniform_filter
import re

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]
OUT_DIR = os.path.join(BASE_DIR, "datasets", "noaa-debris", "e4")
CROP_SIZE = 512
PIXEL_SIZE_M = 0.5

VERIFIED_TARGETS = [
    {"target_id": "TGT001", "lat_str": "28\u00b0 59' 03.3504\" N", "lon_str": "089\u00b0 18' 09.3672\" W"},
    {"target_id": "TGT002", "lat_str": "28\u00b0 58' 58.6344\" N", "lon_str": "089\u00b0 18' 07.3908\" W"},
    {"target_id": "TGT003", "lat_str": "29\u00b0 01' 35.0112\" N", "lon_str": "089\u00b0 17' 25.5372\" W"},
    {"target_id": "TGT004", "lat_str": "28\u00b0 54' 58.5288\" N", "lon_str": "089\u00b0 21' 56.8800\" W"},
    {"target_id": "TGT005", "lat_str": "28\u00b0 54' 48.1320\" N", "lon_str": "089\u00b0 22' 05.1348\" W"},
    {"target_id": "TGT006", "lat_str": "28\u00b0 54' 54.197\" N", "lon_str": "089\u00b0 22' 16.974\" W"},
    {"target_id": "TGT007", "lat_str": "28\u00b0 54' 10.6380\" N", "lon_str": "089\u00b0 25' 30.9252\" W"},
    {"target_id": "TGT008", "lat_str": "28\u00b0 54' 42.4764\" N", "lon_str": "089\u00b0 25' 24.0312\" W"},
    {"target_id": "TGT009", "lat_str": "28\u00b0 54' 41.62\" N", "lon_str": "089\u00b0 25' 22.30\" W"},
    {"target_id": "TGT010", "lat_str": "28\u00b0 54' 38.79\" N", "lon_str": "089\u00b0 25' 17.40\" W"},
    {"target_id": "TGT011", "lat_str": "28\u00b0 54' 59.10\" N", "lon_str": "089\u00b0 25' 34.45\" W"},
    {"target_id": "TGT012", "lat_str": "28\u00b0 54' 59.78\" N", "lon_str": "089\u00b0 25' 32.53\" W"},
    {"target_id": "TGT013", "lat_str": "28\u00b0 54' 32.396\" N", "lon_str": "089\u00b0 25' 59.387\" W"},
    {"target_id": "TGT014", "lat_str": "28\u00b0 54' 30.07\" N", "lon_str": "089\u00b0 26' 03.29\" W"},
    {"target_id": "TGT015", "lat_str": "28\u00b0 54' 18.14\" N", "lon_str": "089\u00b0 25' 45.09\" W"},
    {"target_id": "TGT016", "lat_str": "28\u00b0 53' 35.68\" N", "lon_str": "089\u00b0 25' 44.09\" W"},
    {"target_id": "TGT017", "lat_str": "28\u00b0 53' 19.464\" N", "lon_str": "089\u00b0 26' 14.276\" W"},
]


def dms_to_dd(s):
    parts = re.split(r'[°\'\"\s/]+', s.strip())
    parts = [p for p in parts if p]
    d, m, sec = float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0, float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + sec / 3600.0
    if 'S' in s.upper() or 'W' in s.upper():
        dd = -dd
    return dd


def main():
    print("=" * 70)
    print("HARD NEGATIVE GENERATOR (windowed, no OOM)")
    print("=" * 70)

    # Transform targets to UTM -> pixel coords
    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)
    target_pixels_all = {}

    for tiff_rel in TIFF_FILES:
        tiff_name = os.path.basename(tiff_rel)
        ds = rasterio.open(os.path.join(BASE_DIR, tiff_rel))
        inv = ~ds.transform
        target_pixels_all[tiff_name] = []
        for t in VERIFIED_TARGETS:
            lat = dms_to_dd(t["lat_str"])
            lon = dms_to_dd(t["lon_str"])
            if lon > 0:
                lon = -lon
            ux, uy = tf.transform(lon, lat)
            col, row = inv * (ux, uy)
            target_pixels_all[tiff_name].append((col, row))
        ds.close()

    # PHASE 1: Scan in blocks to find bright regions (windowed)
    all_candidates = []
    BLOCK_H = 2048  # process 2048 rows at a time
    SCAN_STEP = 128  # grid step within valid strip

    for tiff_rel in TIFF_FILES:
        tiff_name = os.path.basename(tiff_rel)
        ds = rasterio.open(os.path.join(BASE_DIR, tiff_rel))
        px_targets = target_pixels_all[tiff_name]

        print(f"\nScanning {tiff_name} ({ds.width}x{ds.height})...")

        # Find valid rows
        sample = ds.read(1, window=Window(0, 0, min(5000, ds.width), min(5000, ds.height)))
        valid_sample = sample[sample > 0]
        if len(valid_sample) < 100:
            ds.close()
            continue
        p80 = np.percentile(valid_sample, 80)
        p90 = np.percentile(valid_sample, 90)
        print(f"  Percentiles: p80={p80:.0f}, p90={p90:.0f}")

        n_candidates = 0

        for row_start in range(0, ds.height - CROP_SIZE, BLOCK_H // 2):
            row_end = min(row_start + BLOCK_H, ds.height)
            block = ds.read(1, window=Window(0, row_start, ds.width, row_end - row_start))

            # Find valid columns in this block
            col_valid = np.mean(block > 0, axis=0) > 0.05
            valid_cols = np.where(col_valid)[0]
            if len(valid_cols) < CROP_SIZE:
                continue

            col_min, col_max = valid_cols[0], valid_cols[-1]

            # Scan grid within this block
            for row in range(CROP_SIZE // 2, row_end - row_start - CROP_SIZE // 2, SCAN_STEP):
                for col in range(col_min + CROP_SIZE // 2, col_max - CROP_SIZE // 2, SCAN_STEP):
                    abs_row = row_start + row

                    # Skip if near a target
                    too_close = False
                    for tc, tr in px_targets:
                        if abs(col - tc) < CROP_SIZE and abs(abs_row - tr) < CROP_SIZE:
                            too_close = True
                            break
                    if too_close:
                        continue

                    # Extract crop from block
                    br = row - CROP_SIZE // 2
                    bc = col - CROP_SIZE // 2
                    if br < 0 or bc < 0 or br + CROP_SIZE > block.shape[0] or bc + CROP_SIZE > block.shape[1]:
                        continue
                    crop = block[br:br+CROP_SIZE, bc:bc+CROP_SIZE]

                    valid = crop[crop > 0]
                    if len(valid) < 200:
                        continue

                    crop_mean = float(valid.mean())
                    crop_max = float(valid.max())
                    crop_std = float(valid.std())

                    # Score based on how "debris-like" this background is
                    # 1. Max brightness relative to seabed
                    brightness_score = crop_max / max(p80, 1)
                    # 2. How many pixels are bright (>p90)
                    n_bright = np.sum(crop > p90)
                    bright_frac = n_bright / max(len(valid), 1)
                    # 3. Local contrast (center vs edges)
                    center = crop[CROP_SIZE//4:3*CROP_SIZE//4, CROP_SIZE//4:3*CROP_SIZE//4]
                    center_valid = center[center > 0]
                    if len(center_valid) < 50:
                        continue
                    edge_top = crop[:CROP_SIZE//8, :]
                    edge_bot = crop[-CROP_SIZE//8:, :]
                    edge_valid = np.concatenate([edge_top[edge_top > 0], edge_bot[edge_bot > 0]]) if len(edge_top[edge_top > 0]) > 0 and len(edge_bot[edge_bot > 0]) > 0 else center_valid
                    edge_mean = float(edge_valid.mean())
                    center_mean = float(center_valid.mean())
                    contrast = center_mean / max(edge_mean, 1)

                    # Texture
                    texture = crop_std / max(crop_mean, 1)

                    # Hardness = combination
                    hardness = (brightness_score * 0.25 +
                               bright_frac * 0.30 +
                               contrast * 0.25 +
                               texture * 0.20)

                    # Only keep "hard" backgrounds — those with some bright content
                    if bright_frac < 0.02:  # need at least 2% bright pixels
                        continue
                    if crop_max < p80:  # need at least one bright pixel
                        continue

                    all_candidates.append({
                        'crop': crop.copy(),
                        'hardness': hardness,
                        'brightness': brightness_score,
                        'bright_frac': bright_frac,
                        'contrast': contrast,
                        'texture': texture,
                        'max_val': crop_max,
                        'mean_val': crop_mean,
                        'n_bright': int(n_bright),
                        'tiff': tiff_name,
                        'col': col,
                        'row': abs_row,
                    })
                    n_candidates += 1

            # Progress
            pct = 100 * (row_start + BLOCK_H // 2) / ds.height
            print(f"  Progress: {pct:.0f}% ({n_candidates} candidates so far)")

        print(f"  Total from {tiff_name}: {n_candidates} candidates")
        ds.close()

    # PHASE 2: Sort, deduplicate, save
    print(f"\nTotal candidates: {len(all_candidates)}")
    if len(all_candidates) == 0:
        print("No hard negatives found!")
        return

    all_candidates.sort(key=lambda x: -x['hardness'])

    # Deduplicate
    selected = []
    used = set()
    for cand in all_candidates:
        key = (cand['tiff'], cand['col'] // 256, cand['row'] // 256)
        if key in used:
            continue
        used.add(key)
        selected.append(cand)
        if len(selected) >= 300:
            break

    print(f"Selected {len(selected)} hard negatives (deduplicated)")

    hardnesses = [s['hardness'] for s in selected]
    bright_fracs = [s['bright_frac'] for s in selected]
    print(f"  Hardness: {np.mean(hardnesses):.3f} +/- {np.std(hardnesses):.3f}")
    print(f"  Bright fraction: {np.mean(bright_fracs)*100:.1f}% +/- {np.std(bright_fracs)*100:.1f}%")

    # Save
    hard_neg_dir = os.path.join(BASE_DIR, "datasets", "noaa-debris", "hard_negatives")
    os.makedirs(hard_neg_dir, exist_ok=True)

    for i, cand in enumerate(selected):
        img = Image.fromarray(cand['crop'])
        img.save(os.path.join(hard_neg_dir, f"HARD_BG_{i+1:04d}.png"))

    meta_path = os.path.join(hard_neg_dir, "hard_negatives_meta.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "hardness", "brightness", "bright_frac", "contrast", "texture", "max_val", "mean_val", "n_bright", "tiff", "col", "row"])
        writer.writeheader()
        for i, cand in enumerate(selected):
            writer.writerow({
                "crop_id": f"HARD_BG_{i+1:04d}",
                "hardness": round(cand['hardness'], 4),
                "brightness": round(cand['brightness'], 3),
                "bright_frac": round(cand['bright_frac'], 4),
                "contrast": round(cand['contrast'], 3),
                "texture": round(cand['texture'], 3),
                "max_val": round(cand['max_val'], 1),
                "mean_val": round(cand['mean_val'], 1),
                "n_bright": cand['n_bright'],
                "tiff": cand['tiff'],
                "col": cand['col'],
                "row": cand['row'],
            })

    print(f"\nSaved {len(selected)} hard negatives to: {hard_neg_dir}")

    print("\nTop 10 hardest:")
    for i, c in enumerate(selected[:10]):
        print(f"  {i+1}. hard={c['hardness']:.3f} bright={c['bright_frac']*100:.1f}% "
              f"contrast={c['contrast']:.2f} max={c['max_val']:.0f} ({c['tiff']})")


if __name__ == "__main__":
    main()
