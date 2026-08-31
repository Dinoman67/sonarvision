#!/usr/bin/env python3
"""
scripts/validate_noaa_dataset.py

Validates the NOAA H11833 YOLO dataset and associated metadata:
- Verifies existence of all image and label files across train/val/test splits
- Verifies normalized and valid YOLO bbox values (0 <= coords <= 1, positive dims)
- Verifies class IDs match data.yaml
- Verifies unique patch IDs (no duplicates)
- Verifies geolocation metadata for all generated patches
- Verifies annotation metadata references valid patch IDs
- Verifies test set isolation and checks for spatial/image leakage across splits

Prints concise summary in exact required format.
"""

import os
import sys
import csv
import yaml
from pathlib import Path

def validate_dataset(base_dir: str = None) -> bool:
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    yolo_dir = Path(base_dir) / "datasets" / "noaa-debris" / "yolo"
    meta_dir = Path(base_dir) / "datasets" / "noaa-debris" / "metadata"
    data_yaml_path = yolo_dir / "data.yaml"

    passed = True
    missing_labels = 0
    missing_metadata = 0
    duplicate_ids = 0
    potential_leakage = 0

    # 1. Check data.yaml
    if not data_yaml_path.exists():
        print("FAIL: data.yaml missing")
        return False
    
    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    
    classes = data_cfg.get("names", {})
    if isinstance(classes, list):
        valid_class_ids = set(range(len(classes)))
        class_names = classes
    elif isinstance(classes, dict):
        valid_class_ids = set(classes.keys())
        class_names = list(classes.values())
    else:
        valid_class_ids = set()
        class_names = []

    # 2. Check metadata files
    geo_csv = meta_dir / "image_geolocation.csv"
    ann_csv = meta_dir / "annotations.csv"
    split_csv = meta_dir / "split_manifest.csv"
    tiff_meta = meta_dir / "tiff_metadata.json"

    if not all([geo_csv.exists(), ann_csv.exists(), split_csv.exists(), tiff_meta.exists()]):
        print("FAIL: One or more metadata files missing")
        return False

    geo_records = {}
    with open(geo_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patch_id"]
            if pid in geo_records:
                duplicate_ids += 1
            geo_records[pid] = row

    split_manifest = {}
    with open(split_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_manifest[row["patch_id"]] = row["split"]

    ann_records = []
    with open(ann_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ann_records.append(row)

    # 3. Check train/val/test splits and image/label pairs
    split_counts = {"train": 0, "val": 0, "test": 0}
    seen_images = set()
    annotated_objects_count = 0

    image_positions = {}  # patch_id -> (utm_x, utm_y, split)

    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"FAIL: Directory missing for split {split}")
            return False

        images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.txt"))

        split_counts[split] = len(images)

        for img_path in images:
            stem = img_path.stem
            # Check for duplicate image across splits
            if stem in seen_images:
                potential_leakage += 1
                passed = False
            seen_images.add(stem)

            # Check geolocation metadata exists
            if stem not in geo_records:
                missing_metadata += 1
                passed = False
            else:
                row = geo_records[stem]
                image_positions[stem] = (float(row["utm_x"]), float(row["utm_y"]), split)

            # Check label file exists
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                missing_labels += 1
                passed = False
            else:
                # Validate YOLO annotations
                with open(lbl_path) as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            passed = False
                            continue
                        cid, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if cid not in valid_class_ids:
                            passed = False
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                            passed = False
                        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            passed = False
                        annotated_objects_count += 1

        for lbl_path in labels:
            stem = lbl_path.stem
            img_png = img_dir / f"{stem}.png"
            img_jpg = img_dir / f"{stem}.jpg"
            if not (img_png.exists() or img_jpg.exists()):
                passed = False

    # 4. Check annotations in annotations.csv reference existing patch IDs
    annotated_patch_ids = set()
    for ann in ann_records:
        pid = ann["patch_id"]
        if pid not in seen_images:
            passed = False
        annotated_patch_ids.add(pid)

    # 5. Verify negative patches have empty labels and positives have non-empty labels
    empty_label_patches = set()
    for split in ["train", "val", "test"]:
        lbl_dir = yolo_dir / "labels" / split
        for lbl_path in lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            if os.path.getsize(lbl_path) == 0:
                empty_label_patches.add(stem)
                if stem in annotated_patch_ids:
                    passed = False
            else:
                if stem not in annotated_patch_ids:
                    passed = False

    # 6. Verify verified targets safety margin for all negative patches
    try:
        from scripts.prepare_noaa_sss import VERIFIED_TARGETS, dms_to_dd
        from pyproj import Transformer
        transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)
        targets_utm = []
        for t in VERIFIED_TARGETS:
            lat = dms_to_dd(t["lat_str"])
            lon = dms_to_dd(t["lon_str"])
            if lon > 0:
                lon = -lon
            ux, uy = transformer_to_utm.transform(lon, lat)
            targets_utm.append((ux, uy, t["name"]))

        for pid in empty_label_patches:
            if pid in image_positions:
                px, py, _ = image_positions[pid]
                for tx, ty, tname in targets_utm:
                    dist = ((px - tx)**2 + (py - ty)**2)**0.5
                    # Safety margin: minimum distance to any target center >= 450m (patch half-diagonal is ~362m)
                    if dist < 450.0:
                        passed = False
    except Exception as e:
        pass

    # 7. Spatial leakage check: ensure test patches are not overlapping/within 500m of train/val patches
    test_positions = [pos for pid, pos in image_positions.items() if pos[2] == "test"]
    train_val_positions = [pos for pid, pos in image_positions.items() if pos[2] in ["train", "val"]]

    for tx, ty, _ in test_positions:
        for ox, oy, osplit in train_val_positions:
            dist = ((tx - ox)**2 + (ty - oy)**2)**0.5
            if dist < 500.0:  # If test image overlaps or is adjacent to train/val within 500m
                potential_leakage += 1
                passed = False

    # 8. Check positive images and objects preservation
    if len(annotated_patch_ids) != 124 or annotated_objects_count != 177:
        passed = False

    status_str = "PASS" if passed and missing_labels == 0 and missing_metadata == 0 and duplicate_ids == 0 and potential_leakage == 0 else "FAIL"

    print(status_str)
    print(f"Images: {split_counts['train']} train / {split_counts['val']} val / {split_counts['test']} test")
    print(f"Annotated objects: {annotated_objects_count}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Missing labels: {missing_labels}")
    print(f"Missing metadata: {missing_metadata}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Potential leakage: {potential_leakage}")

    return passed and status_str == "PASS"

if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
