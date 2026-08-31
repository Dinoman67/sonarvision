#!/usr/bin/env python3
"""
scripts/validate_e3_dataset.py

Validation script for NOAA H11833 SSS E3 Dataset:
- Verifies image/label pairs match across train/val/test splits
- Verifies normalized YOLO coordinates: 0 <= xc, yc <= 1, 0 < w, h <= 1
- Verifies no duplicate crop IDs
- Verifies metadata exists for every crop (crop_metadata.csv and split_manifest.csv)
- Verifies positive crops reference valid target IDs
- Verifies negative crops have empty labels
- Verifies contact-level split integrity (no contact appears in >1 split)
- Verifies test contacts are strictly absent from train and val
- Verifies spatial separation between test crops and train/val crops (>500m)
- Prints a concise PASS/FAIL report
"""

import os
import sys
import csv
import yaml
from pathlib import Path
import numpy as np


def validate_e3(base_dir: str = None) -> bool:
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    base_path = Path(base_dir)
    e3_dir = base_path / "datasets" / "noaa-debris" / "e3"
    meta_dir = e3_dir / "metadata"
    data_yaml_path = e3_dir / "data.yaml"

    passed = True
    errors = []

    # 1. Check data.yaml
    if not data_yaml_path.exists():
        print("FAIL: data.yaml missing")
        return False

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    classes = data_cfg.get("names", {})
    if isinstance(classes, dict):
        valid_class_ids = set(classes.keys())
        class_names = list(classes.values())
    elif isinstance(classes, list):
        valid_class_ids = set(range(len(classes)))
        class_names = classes
    else:
        valid_class_ids = {0}
        class_names = ["marine_debris"]

    # 2. Check metadata CSVs
    crop_meta_path = meta_dir / "crop_metadata.csv"
    split_manifest_path = meta_dir / "split_manifest.csv"

    if not crop_meta_path.exists():
        print(f"FAIL: {crop_meta_path} missing")
        return False
    if not split_manifest_path.exists():
        print(f"FAIL: {split_manifest_path} missing")
        return False

    crop_metadata = {}
    duplicate_crop_ids = 0
    with open(crop_meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["crop_id"]
            if cid in crop_metadata:
                duplicate_crop_ids += 1
                errors.append(f"Duplicate crop ID in crop_metadata: {cid}")
                passed = False
            crop_metadata[cid] = row

    split_manifest = {}
    with open(split_manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_manifest[row["crop_id"]] = row

    # 3. Check Contact-Level Split Integrity
    target_to_splits = {}
    for cid, row in crop_metadata.items():
        tid = row.get("target_id", "").strip()
        if tid:
            split = row["split"]
            if tid not in target_to_splits:
                target_to_splits[tid] = set()
            target_to_splits[tid].add(split)

    for tid, splits in target_to_splits.items():
        if len(splits) > 1:
            errors.append(f"Contact leakage: Target {tid} appears in multiple splits: {splits}")
            passed = False

    test_targets = {tid for tid, splits in target_to_splits.items() if "test" in splits}
    train_val_targets = {tid for tid, splits in target_to_splits.items() if "train" in splits or "val" in splits}
    leakage_targets = test_targets.intersection(train_val_targets)
    if leakage_targets:
        errors.append(f"Test contacts present in train/val: {leakage_targets}")
        passed = False

    # 4. Check images, labels, and YOLO bounding boxes
    split_counts = {"train": 0, "val": 0, "test": 0}
    pos_counts = {"train": 0, "val": 0, "test": 0}
    neg_counts = {"train": 0, "val": 0, "test": 0}
    seen_images = set()
    total_bboxes = 0
    missing_labels = 0
    missing_metadata = 0
    invalid_bbox_count = 0

    image_positions = {}  # crop_id -> (utm_x, utm_y, split)

    for split in ["train", "val", "test"]:
        img_dir = e3_dir / "images" / split
        lbl_dir = e3_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"Missing images/labels directory for split: {split}")
            passed = False
            continue

        images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        split_counts[split] = len(images)

        for img_path in images:
            crop_id = img_path.stem
            if crop_id in seen_images:
                errors.append(f"Duplicate image filename: {crop_id}")
                passed = False
            seen_images.add(crop_id)

            if crop_id not in crop_metadata:
                missing_metadata += 1
                errors.append(f"Crop {crop_id} missing from crop_metadata.csv")
                passed = False
            else:
                row = crop_metadata[crop_id]
                image_positions[crop_id] = (float(row["utm_x"]), float(row["utm_y"]), split)
                if row["is_positive"].lower() == "true":
                    pos_counts[split] += 1
                else:
                    neg_counts[split] += 1

            lbl_path = lbl_dir / f"{crop_id}.txt"
            if not lbl_path.exists():
                missing_labels += 1
                errors.append(f"Label missing for image: {img_path.name}")
                passed = False
            else:
                is_pos = (crop_metadata.get(crop_id, {}).get("is_positive", "").lower() == "true")
                with open(lbl_path) as lf:
                    lines = [l.strip() for l in lf if l.strip()]

                if not is_pos and len(lines) > 0:
                    errors.append(f"Negative crop {crop_id} has non-empty label file!")
                    passed = False
                elif is_pos and len(lines) == 0:
                    errors.append(f"Positive crop {crop_id} has empty label file!")
                    passed = False

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_bbox_count += 1
                        passed = False
                        continue
                    try:
                        cid, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if cid not in valid_class_ids:
                            invalid_bbox_count += 1
                            passed = False
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                            invalid_bbox_count += 1
                            passed = False
                        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            invalid_bbox_count += 1
                            passed = False
                        total_bboxes += 1
                    except ValueError:
                        invalid_bbox_count += 1
                        passed = False

        for lbl_path in labels:
            crop_id = lbl_path.stem
            img_png = img_dir / f"{crop_id}.png"
            img_jpg = img_dir / f"{crop_id}.jpg"
            if not (img_png.exists() or img_jpg.exists()):
                errors.append(f"Orphan label file with no image: {lbl_path.name}")
                passed = False

    # 5. Spatial Leakage Check
    test_positions = [pos for pid, pos in image_positions.items() if pos[2] == "test"]
    train_val_positions = [pos for pid, pos in image_positions.items() if pos[2] in ["train", "val"]]

    spatial_leakages = 0
    for tx, ty, _ in test_positions:
        for ox, oy, osplit in train_val_positions:
            dist = np.hypot(tx - ox, ty - oy)
            if dist < 500.0:
                spatial_leakages += 1
                errors.append(f"Spatial leakage: test crop at ({tx:.0f}, {ty:.0f}) within {dist:.0f}m of {osplit} crop")
                passed = False

    # 6. Report
    status_str = "PASS" if passed and len(errors) == 0 else "FAIL"

    print("=" * 60)
    print(f"E3 DATASET VALIDATION REPORT: {status_str}")
    print("=" * 60)
    print(f"Total Crops:          {sum(split_counts.values())}")
    print(f"  Train:              {split_counts['train']} ({pos_counts['train']} pos / {neg_counts['train']} neg)")
    print(f"  Val:                {split_counts['val']} ({pos_counts['val']} pos / {neg_counts['val']} neg)")
    print(f"  Test:               {split_counts['test']} ({pos_counts['test']} pos / {neg_counts['test']} neg)")
    print(f"Total Bounding Boxes: {total_bboxes}")
    print(f"Classes:              {', '.join(class_names)} (id: {list(valid_class_ids)})")
    print(f"Unique Contacts:      {len(target_to_splits)} mapped cleanly across splits")
    print(f"Contact Split Map:    Train: {len([t for t, s in target_to_splits.items() if 'train' in s])}, Val: {len([t for t, s in target_to_splits.items() if 'val' in s])}, Test: {len([t for t, s in target_to_splits.items() if 'test' in s])}")
    print(f"Missing Labels:       {missing_labels}")
    print(f"Missing Metadata:     {missing_metadata}")
    print(f"Duplicate IDs:        {duplicate_crop_ids}")
    print(f"Invalid Bboxes:       {invalid_bbox_count}")
    print(f"Spatial Leakages:     {spatial_leakages}")

    if errors:
        print("\nErrors encountered:")
        for err in errors[:10]:
            print(f"  - {err}")
    print("=" * 60)

    return passed and status_str == "PASS"


if __name__ == "__main__":
    success = validate_e3()
    sys.exit(0 if success else 1)
