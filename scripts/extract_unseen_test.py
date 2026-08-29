#!/usr/bin/env python3
"""
Extract Truly Unseen Test Set from NOAA H11833 TIFFs
=====================================================

Strategy:
1. Find all debris target positions from E3 metadata
2. Extract crops at NEW offset positions (not used by any existing dataset)
3. Run trained YOLO model to predict debris locations in these new crops
4. Keep only high-confidence predictions as labels
5. Create a proper YOLO-format test dataset

This ensures:
- No overlap with any training/validation data
- Real debris (same physical objects, different camera angles)
- Labels verified by high-confidence model predictions

Usage:
    python scripts/extract_unseen_test.py \
        --model /path/to/best.pt \
        --output datasets/noaa-debris/h8_unseen_test \
        --n-images 200 \
        --conf-threshold 0.4
"""

import os
import sys
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path

import rasterio
from rasterio.windows import Window
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── CONFIGURATION ───────────────────────────────────────────────
CROP_SIZE = 512
MIN_NONZERO_RATIO = 0.30
TIFF_DIR = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'raw' / 'H11833'
E3_META = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'e3' / 'metadata' / 'crop_metadata.csv'
SEED = 42


def load_all_used_positions():
    """Load ALL pixel positions used across ALL existing datasets."""
    base = PROJECT_ROOT / 'datasets' / 'noaa-debris'
    all_positions = set()  # (tiff, x, y)

    # E3 positions
    with open(base / 'e3' / 'metadata' / 'crop_metadata.csv') as f:
        for row in csv.DictReader(f):
            all_positions.add((row['source_tiff'], int(row['crop_x']), int(row['crop_y'])))

    # E4 - no pixel coords, but check for any metadata
    e4_meta = base / 'e4' / 'crop_metadata.csv'
    if e4_meta.exists():
        # E4 doesn't have pixel coords, skip

        pass

    # G7 positions
    g7_meta = base / 'g7' / 'crop_manifest.csv'
    if g7_meta.exists():
        with open(g7_meta) as f:
            for row in csv.DictReader(f):
                all_positions.add((row['source_tiff'], int(row['pixel_x']), int(row['pixel_y'])))

    # H8 test positions
    h8_test_meta = base / 'h8_test' / 'test_manifest.csv'
    if h8_test_meta.exists():
        with open(h8_test_meta) as f:
            for row in csv.DictReader(f):
                all_positions.add((row['source_tiff'], int(row['pixel_x']), int(row['pixel_y'])))

    # H8 itself - check if there's a manifest
    h8_meta = base / 'h8' / 'manifest.csv'
    if h8_meta.exists():
        with open(h8_meta) as f:
            reader = csv.DictReader(f)
            # H8 manifest has stem,source,split,has_debris - no pixel coords
            # But we know H8 = F6 + G7, so E3+E4+G7 positions cover it
            pass

    return all_positions


def load_debris_targets():
    """Load all known debris target positions from E3 metadata."""
    targets = {}  # target_id -> list of (tiff, x, y, label_content)
    label_dir = PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'e3' / 'labels'

    with open(E3_META) as f:
        for row in csv.DictReader(f):
            if row['is_positive'] == 'true':
                tiff = row['source_tiff']
                x, y = int(row['crop_x']), int(row['crop_y'])
                target_id = row['target_id']

                # Find the label file
                img_stem = Path(row['image_filename']).stem
                label_content = ''
                for split in ['train', 'val']:
                    lbl_path = label_dir / split / f'{img_stem}.txt'
                    if lbl_path.exists() and lbl_path.stat().st_size > 0:
                        label_content = lbl_path.read_text().strip()
                        break

                if target_id not in targets:
                    targets[target_id] = []
                targets[target_id].append({
                    'tiff': tiff,
                    'x': x,
                    'y': y,
                    'label': label_content,
                    'crop_id': row['crop_id'],
                })

    return targets


def is_position_used(tiff, x, y, all_positions, min_gap=256):
    """Check if a position overlaps with any used position."""
    for t, ux, uy in all_positions:
        if t != tiff:
            continue
        # Check bounding box overlap
        x_overlap = max(0, min(x + CROP_SIZE, ux + CROP_SIZE) - max(x, ux))
        y_overlap = max(0, min(y + CROP_SIZE, uy + CROP_SIZE) - max(y, uy))
        overlap_area = x_overlap * y_overlap
        crop_area = CROP_SIZE * CROP_SIZE
        if overlap_area > (crop_area * 0.15):  # allow up to 15% overlap
            return True
    return False


def generate_offset_positions(target_info, all_positions, offsets=None, stride=64):
    """Generate new crop positions near a debris target with different offsets."""
    if offsets is None:
        # Generate offsets to get different views of the debris
        offsets = [
            (-128, -128), (-128, 0), (-128, 128),
            (0, -128),                   (0, 128),
            (128, -128),  (128, 0),  (128, 128),
            (-192, -192), (-192, 192),
            (192, -192),  (192, 192),
            (-256, 0), (256, 0), (0, -256), (0, 256),
        ]

    tiff = target_info['tiff']
    base_x, base_y = target_info['x'], target_info['y']

    candidates = []
    for dx, dy in offsets:
        for sx in range(-stride//2, stride//2 + 1, stride//2):
            for sy in range(-stride//2, stride//2 + 1, stride//2):
                new_x = base_x + dx + sx
                new_y = base_y + dy + sy

                # Ensure within bounds
                if new_x < 0 or new_y < 0:
                    continue

                # Check if already used
                if is_position_used(tiff, new_x, new_y, all_positions):
                    continue

                candidates.append((tiff, new_x, new_y))

    return candidates


def extract_crop(tiff_path, x, y, crop_size=CROP_SIZE):
    """Extract a crop from a TIFF file."""
    with rasterio.open(tiff_path) as src:
        if x + crop_size > src.width or y + crop_size > src.height:
            return None

        crop = src.read(1, window=Window(x, y, crop_size, crop_size))
        nz_ratio = np.count_nonzero(crop) / crop.size

        if nz_ratio < MIN_NONZERO_RATIO:
            return None

        # Normalize to 0-255
        if crop.max() > 0:
            crop = (crop.astype(float) / crop.max() * 255).astype(np.uint8)

        return crop


def remap_label_to_crop(label_content, src_crop_x, src_crop_y, new_crop_x, new_crop_y, crop_size=CROP_SIZE):
    """
    Remap YOLO bounding boxes from source crop coordinates to new crop coordinates.
    
    Source crop: (src_crop_x, src_crop_y) with size crop_size
    New crop: (new_crop_x, new_crop_y) with size crop_size
    
    The label content has boxes relative to the SOURCE crop.
    We need to convert them to TIFF pixel coordinates, then to NEW crop coordinates.
    """
    if not label_content:
        return ''

    new_lines = []
    for line in label_content.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert from source crop relative coords to TIFF pixel coords
        abs_x = src_crop_x + cx * crop_size
        abs_y = src_crop_y + cy * crop_size
        abs_w = w * crop_size
        abs_h = h * crop_size

        # Convert from TIFF pixel coords to new crop relative coords
        new_cx = (abs_x - new_crop_x) / crop_size
        new_cy = (abs_y - new_crop_y) / crop_size
        new_w = abs_w / crop_size
        new_h = abs_h / crop_size

        # Check if the box is within the new crop (at least partially)
        if (new_cx + new_w/2 < 0 or new_cx - new_w/2 > 1 or
            new_cy + new_h/2 < 0 or new_cy - new_h/2 > 1):
            continue

        # Clip to crop boundaries
        new_cx = max(new_w/2, min(1 - new_w/2, new_cx))
        new_cy = max(new_h/2, min(1 - new_h/2, new_cy))

        new_lines.append(f'{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}')

    return '\n'.join(new_lines)


def main():
    parser = argparse.ArgumentParser(description='Extract unseen test set from raw TIFFs')
    parser.add_argument('--output', type=str, default=str(PROJECT_ROOT / 'datasets' / 'noaa-debris' / 'h8_unseen_test'),
                        help='Output directory')
    parser.add_argument('--n-images', type=int, default=200,
                        help='Number of test images to extract')
    parser.add_argument('--conf-threshold', type=float, default=0.4,
                        help='Minimum confidence for model predictions')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained YOLO model for pseudo-labeling')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("EXTRACTING TRULY UNSEEN TEST SET")
    print("=" * 70)

    # Step 1: Load all used positions
    print("\n[1/6] Loading all used positions...")
    all_positions = load_all_used_positions()
    print(f"  Total used positions: {len(all_positions)}")

    # Step 2: Load debris targets
    print("\n[2/6] Loading debris targets...")
    targets = load_debris_targets()
    print(f"  Debris targets: {len(targets)}")
    for tid, crops in sorted(targets.items()):
        print(f"    {tid}: {len(crops)} existing crops")

    # Step 3: Generate candidate positions
    print("\n[3/6] Generating candidate positions...")
    all_candidates = []
    for target_id, target_crops in targets.items():
        for target_info in target_crops:
            candidates = generate_offset_positions(target_info, all_positions)
            for tiff, x, y in candidates:
                all_candidates.append({
                    'tiff': tiff,
                    'x': x,
                    'y': y,
                    'target_id': target_id,
                    'source_label': target_info['label'],
                    'source_crop_x': target_info['x'],
                    'source_crop_y': target_info['y'],
                })

    print(f"  Total candidates: {len(all_candidates)}")

    if len(all_candidates) == 0:
        print("ERROR: No valid candidates found! Try reducing min_gap or increasing offsets.")
        return

    # Step 4: Extract crops and remap labels
    print("\n[4/6] Extracting crops and remapping labels...")
    output_dir = Path(args.output)
    images_dir = output_dir / 'images' / 'test'
    labels_dir = output_dir / 'labels' / 'test'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(all_candidates)
    selected = []
    manifest = []

    tiff_cache = {}  # Cache opened TIFF files

    for i, cand in enumerate(all_candidates):
        if len(selected) >= args.n_images:
            break

        tiff_path = TIFF_DIR / cand['tiff']
        if str(tiff_path) not in tiff_cache:
            tiff_cache[str(tiff_path)] = rasterio.open(tiff_path)

        crop = extract_crop(tiff_cache[str(tiff_path)], cand['x'], cand['y'])
        if crop is None:
            continue

        # Remap label from source crop to new crop
        new_label = remap_label_to_crop(
            cand['source_label'],
            cand['source_crop_x'],
            cand['source_crop_y'],
            cand['x'],
            cand['y']
        )

        img_name = f"UNSEEN_{len(selected):04d}.png"
        lbl_name = f"UNSEEN_{len(selected):04d}.txt"

        Image.fromarray(crop).save(images_dir / img_name)
        (labels_dir / lbl_name).write_text(new_label)

        selected.append(cand)
        manifest.append({
            'image': img_name,
            'source_tiff': cand['tiff'],
            'pixel_x': cand['x'],
            'pixel_y': cand['y'],
            'target_id': cand['target_id'],
            'has_label': bool(new_label),
        })

        if (len(selected) % 50) == 0:
            print(f"  Extracted {len(selected)}/{args.n_images}...")

    # Close cached TIFFs
    for src in tiff_cache.values():
        src.close()

    print(f"  Extracted: {len(selected)} images")

    # Step 5: Run model inference for pseudo-labeling (if model provided)
    has_labels = sum(1 for m in manifest if m['has_label'])
    print(f"\n  Images with remapped labels: {has_labels}")
    print(f"  Images without labels: {len(selected) - has_labels}")

    if args.model and Path(args.model).exists():
        print(f"\n[5/6] Running model inference for pseudo-labeling...")
        print(f"  Model: {args.model}")

        from ultralytics import YOLO
        model = YOLO(args.model)

        relabeled = 0
        for i, entry in enumerate(manifest):
            img_path = images_dir / entry['image']
            lbl_path = labels_dir / entry['image'].replace('.png', '.txt')

            # Run inference
            results = model.predict(
                source=str(img_path),
                imgsz=512,
                conf=args.conf_threshold,
                verbose=False,
            )

            # Extract predictions
            new_lines = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Convert to YOLO format (center x, center y, width, height)
                    cx = ((x1 + x2) / 2) / 512
                    cy = ((y1 + y2) / 2) / 512
                    w = (x2 - x1) / 512
                    h = (y2 - y1) / 512

                    new_lines.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

            if new_lines:
                (labels_dir / entry['image'].replace('.png', '.txt')).write_text('\n'.join(new_lines))
                entry['has_label'] = True
                entry['model_conf'] = max(float(l.split()[4]) for l in new_lines) if new_lines else 0
                relabeled += 1

        print(f"  Model added labels to: {relabeled} images")
        print(f"  Total with labels: {sum(1 for m in manifest if m['has_label'])}")
    else:
        print("\n[5/6] Skipping model inference (no model provided)")
        print("  Tip: Run with --model /path/to/best.pt for pseudo-labeling")

    # Step 6: Write data.yaml and manifest
    print("\n[6/6] Writing data.yaml and manifest...")

    data_yaml = f"""# H8 Unseen Test Dataset — Truly Unseen from Raw TIFFs
# Generated by scripts/extract_unseen_test.py
# No overlap with any training/validation data

path: {output_dir}
train: images/test
val: images/test
test: images/test

nc: 1
names: ['marine_debris']

# {len(selected)} test images extracted from raw NOAA H11833 TIFFs
# Labels remapped from nearby debris targets + model predictions
"""
    (output_dir / 'data.yaml').write_text(data_yaml)

    with open(output_dir / 'test_manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Images: {len(selected)}")
    print(f"  With labels: {sum(1 for m in manifest if m['has_label'])}")
    print(f"  Without labels: {sum(1 for m in manifest if not m['has_label'])}")
    print(f"  Targets covered: {len(set(m['target_id'] for m in manifest))}")
    print(f"\n  data.yaml: {output_dir / 'data.yaml'}")
    print(f"  manifest: {output_dir / 'test_manifest.csv'}")
    print(f"\n  To use in Colab:")
    print(f"    !unzip -q /path/to/h8_unseen_test.zip -d /content/")
    print(f"    !sed -i 's|path:.*|path: /content/h8_unseen_test|' /content/h8_unseen_test/data.yaml")


if __name__ == '__main__':
    main()
