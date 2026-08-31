#!/usr/bin/env python3
"""
scripts/validate_ghostvision_yolo.py

Comprehensive validation script for the GhostVision SSS YOLO dataset:
- Verifies data.yaml configuration and class mapping
- Verifies conversion_manifest.csv completeness and consistency
- Verifies 1:1 image and label pairs matching across train/val/test splits
- Verifies valid YOLO format: class_id xc yc w h
- Verifies normalized coordinates (0 <= xc, yc <= 1, 0 < w, h <= 1)
- Verifies class IDs are strictly in {0, 1}
- Verifies no duplicate filenames across or within splits
- Verifies source-sequence grouping integrity (0 sequence leakage between train and val)
- Verifies original GhostVision test split isolation
- Prints a concise PASS/FAIL report
"""

import os
import sys
import csv
import yaml
import re
from pathlib import Path
from collections import defaultdict


def get_sequence_id(filename: str) -> str:
    """Extract source sequence / recording group from filename."""
    base = filename.split(".rf.")[0] if ".rf." in filename else filename
    parts = base.split("_")
    if base.startswith("BC_POST_"):
        return "_".join(parts[:3])
    if base.startswith("baycove_"):
        return "_".join(parts[:2])
    if base.startswith("Contact_"):
        return "Contact"
    if base.startswith("BB_"):
        return "_".join(parts[:2])
    if base.startswith("MC"):
        return parts[0]
    if base.startswith("TI"):
        return parts[0]
    if base.startswith("Rec"):
        if "Sensor_Depth" in base:
            m = re.match(r"^(Rec\d+_Sensor_Depth)", base)
            if m:
                return m.group(1)
        elif "_wcp_" in base:
            m = re.match(r"^(Rec\d+_wcp)", base)
            if m:
                return m.group(1)
        return "_".join(parts[:2])
    return parts[0]


def validate_ghostvision_yolo(
    dataset_dir: str = None,
) -> bool:
    project_root = Path(__file__).resolve().parent.parent
    if dataset_dir is None:
        dataset_dir = str(project_root / "datasets" / "ghostvision_sss_yolo")
    ds_path = Path(dataset_dir)
    errors = []
    passed = True

    print("=" * 65)
    print("GHOSTVISION SSS YOLO DATASET VALIDATION")
    print("=" * 65)

    # 1. Check data.yaml
    yaml_path = ds_path / "data.yaml"
    if not yaml_path.exists():
        errors.append(f"data.yaml not found at {yaml_path}")
        print(f"FAIL: data.yaml missing at {yaml_path}")
        return False

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", {})
    if isinstance(names, dict):
        valid_class_ids = set(names.keys())
        class_names = dict(names)
    elif isinstance(names, list):
        valid_class_ids = set(range(len(names)))
        class_names = dict(enumerate(names))
    else:
        errors.append("Invalid 'names' configuration in data.yaml")
        valid_class_ids = {0, 1}
        class_names = {0: "Crab-Pot", 1: "Maybe-Crab-Pot"}

    if valid_class_ids != {0, 1}:
        errors.append(f"Expected class IDs {{0, 1}}, got {valid_class_ids}")

    # 2. Check conversion_manifest.csv
    manifest_path = ds_path / "metadata" / "conversion_manifest.csv"
    if not manifest_path.exists():
        errors.append(f"conversion_manifest.csv not found at {manifest_path}")
        print(f"FAIL: conversion_manifest.csv missing at {manifest_path}")
        return False

    manifest_records = {}
    manifest_filenames = set()
    manifest_duplicates = 0

    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        req_cols = {"filename", "source_split", "output_split", "width", "height", "num_objects", "classes_present"}
        if not req_cols.issubset(set(reader.fieldnames or [])):
            missing_cols = req_cols - set(reader.fieldnames or [])
            errors.append(f"Manifest missing required columns: {missing_cols}")

        for row in reader:
            fn = row["filename"]
            if fn in manifest_filenames:
                manifest_duplicates += 1
                errors.append(f"Duplicate filename in manifest: {fn}")
            manifest_filenames.add(fn)
            manifest_records[fn] = row

    # 3. Check image/label pairs and YOLO annotations
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_objects = {"train": 0, "val": 0, "test": 0}
    class_counts = defaultdict(int)
    empty_images = {"train": 0, "val": 0, "test": 0}
    positive_images = {"train": 0, "val": 0, "test": 0}

    all_image_filenames = set()
    duplicate_image_filenames = 0
    missing_labels = 0
    orphan_labels = 0
    invalid_bboxes = 0
    manifest_mismatches = 0

    split_sequences = defaultdict(set)

    for split in ["train", "val", "test"]:
        img_dir = ds_path / "images" / split
        lbl_dir = ds_path / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"Missing directory for split: {split}")
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        split_counts[split] = len(images)

        lbl_map = {lbl.name: lbl for lbl in labels}

        for img_path in images:
            fn = img_path.name
            stem = fn.rsplit(".", 1)[0]
            expected_lbl = f"{stem}.txt"

            if fn in all_image_filenames:
                duplicate_image_filenames += 1
                errors.append(f"Duplicate image across dataset: {fn}")
            all_image_filenames.add(fn)

            # Sequence tracking
            seq_id = get_sequence_id(fn)
            split_sequences[split].add(seq_id)

            # Manifest check
            if fn not in manifest_records:
                manifest_mismatches += 1
                errors.append(f"Image {fn} missing from conversion_manifest.csv")

            # Check matching label file
            if expected_lbl not in lbl_map:
                missing_labels += 1
                errors.append(f"Missing label file {expected_lbl} for {fn} in {split}")
                continue

            lbl_path = lbl_map[expected_lbl]

            with open(lbl_path, "r") as lf:
                lines = [l.strip() for l in lf if l.strip()]

            # Manifest num_objects check
            if fn in manifest_records:
                exp_nobjs = int(manifest_records[fn]["num_objects"])
                if len(lines) != exp_nobjs:
                    manifest_mismatches += 1
                    errors.append(f"Object count mismatch for {fn}: manifest={exp_nobjs}, label_file={len(lines)}")

            if len(lines) == 0:
                empty_images[split] += 1
            else:
                positive_images[split] += 1

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    invalid_bboxes += 1
                    errors.append(f"Invalid YOLO token count ({len(parts)}) in {lbl_path.name}: '{line}'")
                    continue

                try:
                    cid = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    invalid_bboxes += 1
                    errors.append(f"Non-numeric tokens in {lbl_path.name}: '{line}'")
                    continue

                if cid not in valid_class_ids:
                    invalid_bboxes += 1
                    errors.append(f"Invalid class ID {cid} (valid: {valid_class_ids}) in {lbl_path.name}")

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    invalid_bboxes += 1
                    errors.append(f"Out of bounds center ({xc}, {yc}) in {lbl_path.name}")

                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    invalid_bboxes += 1
                    errors.append(f"Out of bounds dimension ({w}, {h}) in {lbl_path.name}")

                class_counts[cid] += 1
                split_objects[split] += 1

        # Check orphan label files
        img_stems = {img.name.rsplit(".", 1)[0] for img in images}
        for lbl in labels:
            if lbl.stem not in img_stems:
                orphan_labels += 1
                errors.append(f"Orphan label file without image: {lbl.name} in {split}")

    # 4. Sequence Isolation & Leakage Checks
    train_seqs = split_sequences["train"]
    val_seqs = split_sequences["val"]
    test_seqs = split_sequences["test"]

    train_val_leakage = train_seqs & val_seqs
    train_test_leakage = train_seqs & test_seqs
    val_test_leakage = val_seqs & test_seqs

    if train_val_leakage:
        errors.append(f"Sequence leakage between train and val: {train_val_leakage}")
    if train_test_leakage:
        errors.append(f"Sequence leakage between train and test: {train_test_leakage}")
    if val_test_leakage:
        errors.append(f"Sequence leakage between val and test: {val_test_leakage}")

    total_images = sum(split_counts.values())
    total_objects = sum(split_objects.values())

    if len(errors) > 0:
        passed = False

    status_str = "PASS" if passed else "FAIL"

    # Print concise report
    print(f"STATUS:               {status_str}")
    print(f"Total Images:         {total_images}")
    print(f"  Train Split:        {split_counts['train']} ({positive_images['train']} pos / {empty_images['train']} background)")
    print(f"  Val Split:          {split_counts['val']} ({positive_images['val']} pos / {empty_images['val']} background)")
    print(f"  Test Split:         {split_counts['test']} ({positive_images['test']} pos / {empty_images['test']} background)")
    print(f"Total Objects:        {total_objects}")
    print(f"  Crab-Pot (0):       {class_counts[0]}")
    print(f"  Maybe-Crab-Pot (1): {class_counts[1]}")
    print(f"Distinct Sequences:   Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
    print(f"Sequence Leakage:     Train/Val: {len(train_val_leakage)}, Train/Test: {len(train_test_leakage)}, Val/Test: {len(val_test_leakage)}")
    print(f"Missing Labels:       {missing_labels}")
    print(f"Orphan Labels:        {orphan_labels}")
    print(f"Invalid Bboxes:       {invalid_bboxes}")
    print(f"Duplicate Filenames:  {duplicate_image_filenames + manifest_duplicates}")
    print(f"Manifest Mismatches:  {manifest_mismatches}")

    if errors:
        print("\nErrors (first 10 shown):")
        for err in errors[:10]:
            print(f"  - {err}")
    print("=" * 65)

    return passed


if __name__ == "__main__":
    success = validate_ghostvision_yolo()
    sys.exit(0 if success else 1)
