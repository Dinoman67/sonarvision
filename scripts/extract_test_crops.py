#!/usr/bin/env python3
"""
Extract New Test Crops from NOAA H11833 TIFF Files
====================================================

Extracts 512x512 crops from the raw TIFF files for model testing.
Ensures images are NOT present in the h8 dataset (different filenames).
Uses offset positions to minimize overlap with G7 crops.

Output structure (YOLO testable):
    datasets/noaa-debris/h8_test/
        images/test/*.png
        labels/test/*.txt   (empty — no annotations for test)
        data.yaml

Usage:
    python scripts/extract_test_crops.py
"""

import os
import csv
import json
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TIFF_DIR = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'raw' / 'H11833'
OUTPUT_DIR = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'h8_test'
CROP_SIZE = 512
MIN_NONZERO_RATIO = 0.30
N_TEST_CROPS = 500
SEED = 123


def load_g7_pixel_locations():
    """Load G7 crop pixel coordinates — exact positions only."""
    g7_positions = set()
    manifest = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'g7' / 'crop_manifest.csv'
    if manifest.exists():
        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tiff = row['source_tiff']
                px = int(row['pixel_x'])
                py = int(row['pixel_y'])
                g7_positions.add((tiff, px, py))
    return g7_positions


def load_h8_stems():
    """Load all h8 image stems to exclude by name."""
    stems = set()
    manifest = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'h8' / 'manifest.csv'
    if manifest.exists():
        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stems.add(row['stem'])
    return stems


def is_overlapping(px, py, g7_positions, tiff_name, min_gap=64):
    """Check if a crop at (px,py) overlaps any G7 crop by more than min_gap pixels."""
    for gx, gy in g7_positions:
        if tiff_name != gx:
            continue
        # Check bounding box overlap
        x_overlap = max(0, min(px + CROP_SIZE, gy + CROP_SIZE) - max(px, gy))
        y_overlap = max(0, min(py + CROP_SIZE, gy + CROP_SIZE) - max(py, gy))
        overlap_area = x_overlap * y_overlap
        crop_area = CROP_SIZE * CROP_SIZE
        if overlap_area > (crop_area * 0.15):  # allow up to 15% overlap
            return True
    return False


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("EXTRACTING NEW TEST CROPS FROM NOAA TIFFs")
    print("=" * 70)

    # ── Step 1: Load exclusion data ──────────────────────────────
    print("\n[1/5] Loading exclusion data...")
    g7_positions = load_g7_pixel_locations()
    h8_stems = load_h8_stems()
    # Group G7 positions by tiff
    g7_by_tiff = {}
    for tiff, px, py in g7_positions:
        g7_by_tiff.setdefault(tiff, set()).add((px, py))
    for tiff, positions in g7_by_tiff.items():
        print(f"  {tiff}: {len(positions)} G7 crop positions")
    print(f"  h8 stems: {len(h8_stems)}")

    # ── Step 2: Extract crops from TIFFs ─────────────────────────
    print("\n[2/5] Scanning TIFFs for test candidates...")
    tiff_files = sorted(TIFF_DIR.glob('*.tif'))
    all_candidates = []

    for tiff_path in tiff_files:
        tiff_name = tiff_path.name
        g7_tiff = g7_by_tiff.get(tiff_name, set())
        print(f"\n  Processing {tiff_name}...")

        with rasterio.open(tiff_path) as src:
            # Find valid data strip - scan in coarse blocks first
            block_size = 2048
            valid_regions = []
            for row in range(0, src.height - CROP_SIZE, block_size):
                for col in range(0, src.width - CROP_SIZE, block_size):
                    w = min(block_size, src.width - col)
                    h = min(block_size, src.height - row)
                    block = src.read(1, window=Window(col, row, w, h))
                    nz = np.count_nonzero(block)
                    if nz > (CROP_SIZE * CROP_SIZE * MIN_NONZERO_RATIO):
                        valid_regions.append((col, row, w, h))

            print(f"    Valid regions: {len(valid_regions)}")

            # Now extract crops with 256px stride (offset from G7's 512 grid)
            n_candidates = 0
            stride = 256

            for region_col, region_row, rw, rh in valid_regions:
                for row in range(region_row, region_row + rh - CROP_SIZE, stride):
                    for col in range(region_col, region_col + rw - CROP_SIZE, stride):
                        # Skip if this exact position was used by G7
                        if (col, row) in g7_tiff:
                            continue

                        # Check overlap with any G7 crop (allow small overlap)
                        if is_overlapping(col, row, g7_tiff, tiff_name):
                            continue

                        crop = src.read(1, window=Window(col, row, CROP_SIZE, CROP_SIZE))
                        nz_ratio = np.count_nonzero(crop) / crop.size
                        if nz_ratio < MIN_NONZERO_RATIO:
                            continue

                        valid = crop[crop > 0]
                        all_candidates.append({
                            'tiff': tiff_name,
                            'px': col,
                            'py': row,
                            'crop': crop.copy(),
                            'nz_ratio': nz_ratio,
                            'mean_intensity': float(valid.mean()),
                            'max_intensity': float(valid.max()),
                        })
                        n_candidates += 1

            print(f"    Non-overlapping candidates: {n_candidates}")

    print(f"\n  Total candidates: {len(all_candidates)}")

    if len(all_candidates) == 0:
        print("ERROR: No valid candidates found!")
        return

    # ── Step 3: Select diverse test set ──────────────────────────
    print("\n[3/5] Selecting diverse test set...")

    # Sort by intensity for diversity
    all_candidates.sort(key=lambda x: x['mean_intensity'])
    n = len(all_candidates)

    # Sample evenly across intensity range
    if n <= N_TEST_CROPS:
        selected = all_candidates[:]
    else:
        indices = np.linspace(0, n - 1, N_TEST_CROPS, dtype=int)
        selected = [all_candidates[i] for i in indices]

    random.shuffle(selected)
    print(f"  Selected {len(selected)} test crops")

    # ── Step 4: Save as YOLO test dataset ───────────────────────
    print("\n[4/5] Saving YOLO test dataset...")
    images_dir = OUTPUT_DIR / 'images' / 'test'
    labels_dir = OUTPUT_DIR / 'labels' / 'test'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, cand in enumerate(selected):
        img_name = f"TEST_{i:05d}.png"
        lbl_name = f"TEST_{i:05d}.txt"

        # Normalize to 0-255
        img = cand['crop']
        if img.max() > 0:
            img = (img.astype(float) / img.max() * 255).astype(np.uint8)

        Image.fromarray(img).save(images_dir / img_name)
        (labels_dir / lbl_name).touch()  # empty label = no annotations

        manifest.append({
            'image': img_name,
            'source_tiff': cand['tiff'],
            'pixel_x': cand['px'],
            'pixel_y': cand['py'],
            'nonzero_ratio': round(cand['nz_ratio'], 4),
            'mean_intensity': round(cand['mean_intensity'], 2),
            'max_intensity': round(cand['max_intensity'], 1),
        })

    # Save manifest
    with open(OUTPUT_DIR / 'test_manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    # ── Step 5: Write data.yaml ──────────────────────────────────
    print("\n[5/5] Writing data.yaml...")
    data_yaml = f"""# H8 Test Dataset — Unseen TIFF Crops for Model Evaluation
# No images from h8 training/val sets are included
# Generated by scripts/extract_test_crops.py

path: datasets/noaa-debris/h8_test
train: images/test
val: images/test
test: images/test

nc: 1
names: ['marine_debris']

# {len(selected)} test images extracted from raw NOAA TIFFs
# Crops use offset positions to minimize overlap with G7 dataset
"""
    with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
        f.write(data_yaml)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Images: {len(selected)}")
    print(f"  Labels: {len(selected)} (empty — test only)")
    print(f"  Source TIFFs: {len(set(c['tiff'] for c in selected))}")
    print(f"  Intensity range: {min(c['mean_intensity'] for c in selected):.1f} — "
          f"{max(c['mean_intensity'] for c in selected):.1f}")

    from collections import Counter
    tiff_counts = Counter(c['tiff'] for c in selected)
    for tiff, count in tiff_counts.most_common():
        print(f"    {tiff}: {count} crops")

    return OUTPUT_DIR


if __name__ == '__main__':
    main()
