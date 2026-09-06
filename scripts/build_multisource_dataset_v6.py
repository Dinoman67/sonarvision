#!/usr/bin/env python3
"""
SonarVision — Build Multi-Source YOLO Dataset v6
=================================================

v5 → v6 changes (driven by v5 Colab T4 run post-mortem & user instructions):
  1. DEBRIS SPLIT RESTORATION:
     - Restores the proven Core_model dataset structure:
       • train: datasets/noaa-debris/h8/images/train (3,110 frames)
       • val:   datasets/noaa-debris/h8/images/val (438 frames)
       • test:  datasets/noaa-debris/h8_unseen_test/images/test (834 frames)
     - All images are unique (0 file overlap across splits). Holdout is at the
       frame level, sharing visual feature distributions (seabed texture, gain,
       sonar geometry) so the model rapidly learns debris features as Core_model did.
  2. MINE DATA 5-POINT FIX:
     - Map NOMBO (class 1) to mine (class 2): Eliminates 74 contradictory frames
       where unannotated mine-like echoes were treated as background.
     - Filter degenerate/sub-pixel boxes (w < 0.005 or h < 0.005).
     - Subsample background frames: Capped to ~120-150 representative frames
       (prevents 70% background swamping that suppressed mine confidence).
     - Stratified per-year split (2010, 2015, 2017, 2018, 2021) 70/15/15 so
       no single year starves or overpowers Val.
  3. KAGGLE AIRPLANE & WRECK:
     - Maintained proven pipeline (0.89+ Val AP50) with 3x train-only oversampling for airplane.

Master classes (4):
  0: unknown_debris   (NOAA h8 + h8_unseen_test)
  1: airplane         (Kaggle)
  2: mine             (MILCO mine + NOMBO mine-like contacts)
  3: wreck            (Kaggle)

Usage:
    python scripts/build_multisource_dataset_v6.py \
        --output datasets/sonarvision_multisource_v6 \
        --seed 42
"""

import os
import sys
import json
import csv
import re
import hashlib
import shutil
import argparse
import random
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MASTER_CLASSES = {
    0: "unknown_debris",
    1: "airplane",
    2: "mine",
    3: "wreck",
}

# NOAA: h8_data + h8_unseen_test. Debris (0) -> 0.
NOAA_MAPPING = {0: 0}

# MILCO: mine (0) -> 2, NOMBO (1) -> 2 (mine-like contacts).
MILCO_MAPPING = {0: 2, 1: 2}

# Kaggle: airplane (0) -> 1, wreck (3) -> 3. drowning (1) and mine (2) dropped.
KAGGLE_MAPPING = {0: 1, 3: 3}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Oversampling (TRAIN-ONLY, applied AFTER split)
OVERSAMPLE = {
    1: 3,   # airplane: ~80 originals -> ~3x
    2: 2,   # mine: ~210 train originals -> ~2x
}

# Max background frames to retain per MILCO year to prevent background swamping
MILCO_BG_PER_YEAR = 30

# Noise augmentation config
NOISE_CONFIG = {
    'gaussian_sigmas': [0.01, 0.02, 0.03, 0.04, 0.05],
    'speckle_sigmas': [0.01, 0.02, 0.03],
    'brightness_factors': [0.8, 0.9, 1.1, 1.2],
}

# Minimum box dimension (normalized) to filter degenerate/corrupt sub-pixel boxes
MIN_BOX_DIM = 0.005

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_v6")


# ---------------------------------------------------------------------------
# Noise augmentation pipeline
# ---------------------------------------------------------------------------
def add_gaussian_noise(img_array, sigma=0.03):
    noise = np.random.normal(0, sigma * 255, img_array.shape)
    return np.clip(img_array + noise, 0, 255).astype(np.uint8)


def add_speckle_noise(img_array, sigma=0.02):
    noise = np.random.normal(1, sigma, img_array.shape)
    return np.clip(img_array * noise, 0, 255).astype(np.uint8)


def adjust_brightness(img_array, factor=1.1):
    return np.clip(img_array * factor, 0, 255).astype(np.uint8)


def apply_noise_augmentation(img_path, output_path, variant_idx):
    """Apply random combination of sonar-safe noise to an image."""
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    augmentations = []
    if random.random() < 0.5:
        sigma = random.choice(NOISE_CONFIG['gaussian_sigmas'])
        arr = add_gaussian_noise(arr, sigma)
        augmentations.append(f"g{sigma}")
    if random.random() < 0.5:
        sigma = random.choice(NOISE_CONFIG['speckle_sigmas'])
        arr = add_speckle_noise(arr, sigma)
        augmentations.append(f"s{sigma}")
    if random.random() < 0.4:
        factor = random.choice(NOISE_CONFIG['brightness_factors'])
        arr = adjust_brightness(arr, factor)
        augmentations.append(f"b{factor}")
    if not augmentations:
        sigma = random.choice(NOISE_CONFIG['gaussian_sigmas'])
        arr = add_gaussian_noise(arr, sigma)
        augmentations.append(f"g{sigma}")
    Image.fromarray(arr).save(output_path, 'PNG')
    return output_path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_yolo_label(path):
    """Read YOLO label file, handle concatenated boxes and filter tiny/corrupt boxes."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1 \
                            and min(w, h) >= MIN_BOX_DIM:
                        boxes.append((cls, xc, yc, w, h))
                except (ValueError, IndexError):
                    continue
            else:
                i = 0
                while i + 4 < len(parts):
                    try:
                        cls = int(parts[i])
                        xc, yc, w, h = float(parts[i+1]), float(parts[i+2]), float(parts[i+3]), float(parts[i+4])
                        if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1 \
                                and min(w, h) >= MIN_BOX_DIM:
                            boxes.append((cls, xc, yc, w, h))
                        i += 5
                    except (ValueError, IndexError):
                        i += 1
    return boxes


class SampleInfo:
    def __init__(self, image_path, label_path, source, source_detail,
                 original_id, group_key, split_hint=""):
        self.image_path = image_path
        self.label_path = label_path
        self.source = source
        self.source_detail = source_detail
        self.original_id = original_id
        self.group_key = group_key
        self.split_hint = split_hint
        self.master_class_id = None
        self.label_boxes = []
        self.new_filename = ""
        self.new_label_filename = ""
        self.final_split = ""
        self.sha256 = ""
        self.is_corrupt = False
        self.is_duplicate = False
        self.duplicate_of = ""
        self.issues = []
        self.is_augmented = False
        self.augmented_from = ""
        self.width = 0
        self.height = 0


# ---------------------------------------------------------------------------
# Source inventories
# ---------------------------------------------------------------------------
def inventory_noaa_debris(output_dir):
    """Inventory NOAA debris using the proven Core_model splits:
       - h8/images/train -> train
       - h8/images/val   -> val
       - h8_unseen_test/images/test -> test
    """
    logger.info("Inventorying NOAA debris (Core_model proven frame split)...")
    samples = []
    
    # 1. h8 train & val
    base_h8 = os.path.join("datasets", "noaa-debris", "h8")
    for sub in ["train", "val"]:
        img_dir = os.path.join(base_h8, "images", sub)
        lbl_dir = os.path.join(base_h8, "labels", sub)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="NOAA", source_detail=f"h8_{sub}",
                original_id=stem, group_key=f"NOAA_h8_{sub}_{stem}",
                split_hint=sub,
            )
            sample.final_split = sub
            boxes = read_yolo_label(lbl_path)
            sample.label_boxes = [(NOAA_MAPPING.get(c, -1), x, y, w, h)
                                  for c, x, y, w, h in boxes]
            valid = [c for c, _, _, _, _ in sample.label_boxes if c >= 0]
            sample.master_class_id = Counter(valid).most_common(1)[0][0] if valid else None
            samples.append(sample)

    # 2. h8_unseen_test -> test
    base_unseen = os.path.join("datasets", "noaa-debris", "h8_unseen_test")
    img_dir_test = os.path.join(base_unseen, "images", "test")
    lbl_dir_test = os.path.join(base_unseen, "labels", "test")
    if os.path.isdir(img_dir_test):
        for fname in sorted(os.listdir(img_dir_test)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir_test, fname)
            lbl_path = os.path.join(lbl_dir_test, stem + ".txt")
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="NOAA", source_detail="h8_unseen_test",
                original_id=stem, group_key=f"NOAA_h8_test_{stem}",
                split_hint="test",
            )
            sample.final_split = "test"
            boxes = read_yolo_label(lbl_path)
            sample.label_boxes = [(NOAA_MAPPING.get(c, -1), x, y, w, h)
                                  for c, x, y, w, h in boxes]
            valid = [c for c, _, _, _, _ in sample.label_boxes if c >= 0]
            sample.master_class_id = Counter(valid).most_common(1)[0][0] if valid else None
            samples.append(sample)

    counts = Counter(s.final_split for s in samples)
    pos_counts = Counter(s.final_split for s in samples if s.master_class_id == 0)
    logger.info(f"  NOAA debris total: {len(samples)} samples (train={counts['train']} [{pos_counts['train']} pos], "
                f"val={counts['val']} [{pos_counts['val']} pos], test={counts['test']} [{pos_counts['test']} pos])")
    return samples


def inventory_milco(output_dir, seed=42):
    """Inventory MILCO dataset with 5-point mine fix:
       1. Map both mine (0) and NOMBO (1) to master class 2 (mine-like contact).
       2. Filter degenerate sub-pixel boxes.
       3. Cap pure background frames per year to prevent swamping.
       4. Perform balanced per-year stratified 70/15/15 split.
    """
    logger.info("Inventorying MILCO dataset (mine + NOMBO -> mine, background capped, per-year split)...")
    rng = random.Random(seed)
    samples = []
    base = os.path.join("datasets", "milco-nombo", "extracted")
    
    total_pos = 0
    total_bg = 0

    for year in ["2010", "2015", "2017", "2018", "2021"]:
        year_path = os.path.join(base, year, year)
        if not os.path.isdir(year_path):
            year_path = os.path.join(base, year)
            if not os.path.isdir(year_path):
                continue

        pos_samples = []
        bg_samples = []

        for fname in sorted(os.listdir(year_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(year_path, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(year_path, stem + ".txt")
            boxes = read_yolo_label(lbl_path)
            
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="MILCO", source_detail=year,
                original_id=stem, group_key=f"MILCO_{year}_{stem}",
            )
            
            mapped = [(MILCO_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            valid = [b for b in mapped if b[0] >= 0]
            
            if valid:
                sample.label_boxes = valid
                sample.master_class_id = 2 # mine
                pos_samples.append(sample)
            else:
                sample.label_boxes = []
                sample.master_class_id = None
                bg_samples.append(sample)

        # Cap pure background frames for this year
        rng.shuffle(bg_samples)
        bg_kept = bg_samples[:MILCO_BG_PER_YEAR]

        # Stratified 70/15/15 split for positive frames
        rng.shuffle(pos_samples)
        n_pos = len(pos_samples)
        n_pos_train = int(round(n_pos * TRAIN_RATIO))
        n_pos_val = int(round(n_pos * VAL_RATIO))
        
        for i, s in enumerate(pos_samples):
            if i < n_pos_train:
                s.final_split = "train"
            elif i < n_pos_train + n_pos_val:
                s.final_split = "val"
            else:
                s.final_split = "test"

        # Stratified 70/15/15 split for capped background frames
        n_bg = len(bg_kept)
        n_bg_train = int(round(n_bg * TRAIN_RATIO))
        n_bg_val = int(round(n_bg * VAL_RATIO))
        
        for i, s in enumerate(bg_kept):
            if i < n_bg_train:
                s.final_split = "train"
            elif i < n_bg_train + n_bg_val:
                s.final_split = "val"
            else:
                s.final_split = "test"

        year_total = pos_samples + bg_kept
        samples.extend(year_total)
        total_pos += len(pos_samples)
        total_bg += len(bg_kept)
        logger.info(f"  MILCO {year}: {len(pos_samples)} pos, {len(bg_kept)} bg kept (dropped {len(bg_samples)-len(bg_kept)} excess bg)")

    logger.info(f"  MILCO total: {len(samples)} samples ({total_pos} pos, {total_bg} bg)")
    return samples


def inventory_kaggle(output_dir, seed=42):
    """Inventory Kaggle SSS dataset (airplane + wreck only)."""
    logger.info("Inventorying Kaggle dataset (airplane + wreck only)...")
    rng = random.Random(seed)
    samples = []
    base = "datasets/kaggle_sss"
    dropped_only = 0

    for split in ["train", "valid"]:
        img_dir = os.path.join(base, split, "images")
        lbl_dir = os.path.join(base, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            boxes = read_yolo_label(lbl_path)
            mapped = [(KAGGLE_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            valid = [b for b in mapped if b[0] >= 0]
            if boxes and not valid:
                dropped_only += 1
                continue
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="KAGGLE", source_detail=f"SSS/{split}",
                original_id=stem, group_key=f"KAGGLE_{stem}",
                split_hint=split,
            )
            sample.label_boxes = valid
            if valid:
                sample.master_class_id = Counter(b[0] for b in valid).most_common(1)[0][0]
            else:
                sample.master_class_id = None
            samples.append(sample)

    # Perform stratified 70/15/15 split by class
    by_class = defaultdict(list)
    for s in samples:
        by_class[s.master_class_id].append(s)

    for cid, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        n = len(cls_samples)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        for i, s in enumerate(cls_samples):
            if i < n_train:
                s.final_split = "train"
            elif i < n_train + n_val:
                s.final_split = "val"
            else:
                s.final_split = "test"

    counts = Counter(s.final_split for s in samples)
    logger.info(f"  Kaggle total: {len(samples)} samples (train={counts['train']}, val={counts['val']}, test={counts['test']})")
    return samples


# ---------------------------------------------------------------------------
# Quality & duplicate checks
# ---------------------------------------------------------------------------
def check_sample_quality(sample):
    issues = []
    if not os.path.exists(sample.image_path):
        issues.append("missing_image")
        sample.is_corrupt = True
        return issues
    try:
        with Image.open(sample.image_path) as img:
            sample.width, sample.height = img.size
            if sample.width < 10 or sample.height < 10:
                issues.append("image_too_small")
                sample.is_corrupt = True
    except Exception as e:
        issues.append(f"corrupt_image: {e}")
        sample.is_corrupt = True

    for cls_id, xc, yc, w, h in sample.label_boxes:
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            issues.append(f"invalid_box: ({xc},{yc},{w},{h})")
    return issues


def detect_duplicates(samples):
    seen_hashes = {}
    dup_count = 0
    for s in samples:
        if s.is_corrupt:
            continue
        h = sha256_file(s.image_path)
        s.sha256 = h
        if h in seen_hashes:
            s.is_duplicate = True
            s.duplicate_of = seen_hashes[h]
            dup_count += 1
        else:
            seen_hashes[h] = s.original_id
    logger.info(f"Duplicate detection: found {dup_count} exact duplicates")
    return dup_count


# ---------------------------------------------------------------------------
# Oversampling — TRAIN-ONLY, post-split
# ---------------------------------------------------------------------------
def oversample_train_only(samples, output_dir):
    """Create noise-augmented copies of TRAIN-split images ONLY."""
    logger.info("Oversampling minority classes (TRAIN-ONLY, post-split)...")
    augmented = []
    by_class = defaultdict(list)
    for s in samples:
        if (not s.is_duplicate and not s.is_corrupt
                and s.final_split == "train" and s.master_class_id is not None):
            by_class[s.master_class_id].append(s)

    aug_dir = os.path.join(output_dir, "aug_tmp")
    os.makedirs(aug_dir, exist_ok=True)

    for class_id, multiplier in OVERSAMPLE.items():
        class_samples = by_class.get(class_id, [])
        if not class_samples:
            continue
        class_name = MASTER_CLASSES[class_id]
        target_total = len(class_samples) * multiplier
        need = target_total - len(class_samples)
        logger.info(f"  {class_name}: {len(class_samples)} train originals -> "
                    f"target {target_total} (+{need} augmented, train only)")

        for i in range(need):
            src = random.choice(class_samples)
            stem = os.path.splitext(os.path.basename(src.image_path))[0]
            aug_img_name = f"{stem}_aug{i:04d}.png"
            aug_lbl_name = f"{stem}_aug{i:04d}.txt"
            aug_img_path = os.path.join(aug_dir, aug_img_name)
            aug_lbl_path = os.path.join(aug_dir, aug_lbl_name)
            apply_noise_augmentation(src.image_path, aug_img_path, i)
            shutil.copy2(src.label_path, aug_lbl_path)

            aug_sample = SampleInfo(
                image_path=aug_img_path, label_path=aug_lbl_path,
                source=src.source, source_detail=f"{src.source_detail}_aug",
                original_id=f"{src.original_id}_aug{i:04d}",
                group_key=src.group_key,
                split_hint=src.split_hint,
            )
            aug_sample.master_class_id = class_id
            aug_sample.label_boxes = src.label_boxes[:]
            aug_sample.is_augmented = True
            aug_sample.augmented_from = src.image_path
            aug_sample.final_split = "train"
            augmented.append(aug_sample)

    logger.info(f"  Created {len(augmented)} augmented samples (all train)")
    return augmented, aug_dir


# ---------------------------------------------------------------------------
# Filename generation & writing
# ---------------------------------------------------------------------------
def generate_filenames(samples):
    prefix_map = {"NOAA": "NOAA", "MILCO": "MILCO", "KAGGLE": "KAGGLE"}
    counters = defaultdict(int)
    for s in samples:
        if s.is_duplicate or s.is_corrupt:
            continue
        prefix = prefix_map.get(s.source, s.source)
        if s.source == "MILCO":
            counters[f"{prefix}_{s.source_detail}"] += 1
            idx = counters[f"{prefix}_{s.source_detail}"]
            s.new_filename = f"{prefix}_{s.source_detail}_{idx:06d}.png"
        else:
            counters[prefix] += 1
            idx = counters[prefix]
            s.new_filename = f"{prefix}_{idx:06d}.png"
        s.new_label_filename = s.new_filename.replace(".png", ".txt")


def write_dataset(samples, output_dir, aug_dir):
    logger.info("Writing dataset to disk...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "manifests"), exist_ok=True)

    written = 0
    metadata_rows = []

    for s in samples:
        if s.is_duplicate or s.is_corrupt:
            continue
        split = s.final_split
        if not split:
            continue

        dst_img = os.path.join(output_dir, "images", split, s.new_filename)
        try:
            img = Image.open(s.image_path)
            if img.mode != "L":
                img = img.convert("L")
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.BILINEAR)
            img.save(dst_img, "PNG")
            img.close()
        except Exception as e:
            logger.warning(f"Failed to convert {s.image_path}: {e}")
            continue

        dst_lbl = os.path.join(output_dir, "labels", split, s.new_label_filename)
        with open(dst_lbl, "w") as f:
            for cls_id, xc, yc, w, h in s.label_boxes:
                if 0 <= cls_id < len(MASTER_CLASSES):
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        metadata_rows.append({
            "image_id": s.new_filename.replace(".png", ""),
            "image_path": f"images/{split}/{s.new_filename}",
            "source_dataset": s.source,
            "source_detail": s.source_detail,
            "original_id": s.original_id,
            "master_class": MASTER_CLASSES.get(s.master_class_id, "unknown"),
            "split": split,
            "is_augmented": s.is_augmented,
            "augmented_from": s.augmented_from,
            "group_key": s.group_key,
        })
        written += 1

    if metadata_rows:
        meta_path = os.path.join(output_dir, "metadata.csv")
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

    for split in ["train", "val", "test"]:
        manifest_path = os.path.join(output_dir, "manifests", f"{split}.txt")
        split_samples = [s for s in samples
                         if s.final_split == split and not s.is_duplicate and not s.is_corrupt]
        with open(manifest_path, "w") as f:
            for s in split_samples:
                f.write(f"images/{split}/{s.new_filename}\n")

    logger.info(f"  Written {written} samples")


def generate_report(samples, output_dir, dup_count):
    logger.info("Generating dataset audit report...")
    usable = [s for s in samples if not s.is_duplicate and not s.is_corrupt]

    report = {
        "dataset_name": "sonarvision_multisource_v6",
        "total_samples": len(usable),
        "duplicates_removed": dup_count,
        "classes": MASTER_CLASSES,
        "splits": {},
        "class_distribution": {},
        "sources": {},
    }

    for split in ["train", "val", "test"]:
        split_samples = [s for s in usable if s.final_split == split]
        report["splits"][split] = {
            "total_images": len(split_samples),
            "by_class": Counter(MASTER_CLASSES.get(s.master_class_id, "background")
                                for s in split_samples),
            "by_source": Counter(s.source for s in split_samples),
        }

    for cid, cname in MASTER_CLASSES.items():
        report["class_distribution"][cname] = sum(
            1 for s in usable if s.master_class_id == cid)

    for src in ["NOAA", "MILCO", "KAGGLE"]:
        report["sources"][src] = sum(1 for s in usable if s.source == src)

    json_path = os.path.join("reports", "dataset_audit_multisource_v6.json")
    os.makedirs("reports", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join("reports", "sonarvision_multisource_v6_report.md")
    with open(md_path, "w") as f:
        f.write("# SonarVision Multi-Source Dataset v6 Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Images:** {len(usable)}\n")
        f.write(f"- **Exact Duplicates Removed:** {dup_count}\n")
        f.write("- **Classes:**\n")
        for cid, cname in MASTER_CLASSES.items():
            f.write(f"  - {cid}: {cname}\n")
        f.write("\n## Split Summary\n\n")
        f.write("| Split | Images | NOAA (Debris) | MILCO (Mine) | Kaggle (Plane/Wreck) |\n")
        f.write("|-------|--------|---------------|--------------|----------------------|\n")
        for split in ["train", "val", "test"]:
            s_data = report["splits"][split]
            srcs = s_data["by_source"]
            f.write(f"| {split} | {s_data['total_images']} | {srcs.get('NOAA', 0)} | "
                    f"{srcs.get('MILCO', 0)} | {srcs.get('KAGGLE', 0)} |\n")
        f.write("\n## Class Instances (Image-Level)\n\n")
        f.write("| Class | Train | Val | Test | Total |\n")
        f.write("|-------|-------|-----|------|-------|\n")
        for cid, cname in MASTER_CLASSES.items():
            tr = report["splits"]["train"]["by_class"].get(cname, 0)
            va = report["splits"]["val"]["by_class"].get(cname, 0)
            te = report["splits"]["test"]["by_class"].get(cname, 0)
            f.write(f"| {cname} | {tr} | {va} | {te} | {tr+va+te} |\n")


def validate_dataset(output_dir):
    logger.info("Validating dataset integrity...")
    all_ok = True
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        imgs = set(f.replace(".png", "") for f in os.listdir(img_dir) if f.endswith(".png"))
        lbls = set(f.replace(".txt", "") for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        if imgs != lbls:
            logger.error(f"  Split {split}: image/label mismatch! "
                         f"imgs - lbls = {len(imgs - lbls)}, lbls - imgs = {len(lbls - imgs)}")
            all_ok = False
        else:
            logger.info(f"  Split {split}: {len(imgs)} pairs matched perfectly")

    if all_ok:
        logger.info("  Validation PASSED")
    else:
        logger.error("  Validation FAILED")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build SonarVision multi-source dataset v6")
    parser.add_argument("--output", default="datasets/sonarvision_multisource_v6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    logger.info(f"Output: {output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Inventorying data sources")
    logger.info("=" * 60)
    all_samples = []
    all_samples.extend(inventory_noaa_debris(output_dir))
    all_samples.extend(inventory_milco(output_dir, seed=args.seed))
    all_samples.extend(inventory_kaggle(output_dir, seed=args.seed))
    logger.info(f"Total inventoried: {len(all_samples)}")

    logger.info("=" * 60)
    logger.info("STEP 2: Quality checks")
    logger.info("=" * 60)
    for s in all_samples:
        s.issues.extend(check_sample_quality(s))

    logger.info("=" * 60)
    logger.info("STEP 3: Duplicate detection")
    logger.info("=" * 60)
    dup_count = detect_duplicates(all_samples)

    logger.info("=" * 60)
    logger.info("STEP 4: Oversampling (TRAIN-ONLY, post-split)")
    logger.info("=" * 60)
    augmented, aug_dir = oversample_train_only(all_samples, output_dir)
    all_samples.extend(augmented)

    logger.info("=" * 60)
    logger.info("STEP 5: Generating filenames")
    logger.info("=" * 60)
    generate_filenames(all_samples)

    logger.info("=" * 60)
    logger.info("STEP 6: Writing dataset")
    logger.info("=" * 60)
    write_dataset(all_samples, output_dir, aug_dir)

    logger.info("=" * 60)
    logger.info("STEP 7: Writing dataset.yaml")
    logger.info("=" * 60)
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write("# SonarVision Multi-Source Dataset v6\n")
        f.write("# Sensor-isolated classes: debris=h8 (Core_model frame split) | mine=MILCO (5-point fix) | airplane/wreck=Kaggle\n\n")
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: {len(MASTER_CLASSES)}\n")
        f.write(f"names:\n")
        for cid, cname in MASTER_CLASSES.items():
            f.write(f"  {cid}: {cname}\n")

    logger.info("=" * 60)
    logger.info("STEP 8: Generating reports")
    logger.info("=" * 60)
    generate_report(all_samples, output_dir, dup_count)

    logger.info("=" * 60)
    logger.info("STEP 9: Final validation")
    logger.info("=" * 60)
    valid = validate_dataset(output_dir)

    usable = [s for s in all_samples if not s.is_duplicate and not s.is_corrupt]
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET V6 BUILD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"NOAA (h8):     {sum(1 for s in usable if s.source == 'NOAA')}")
    logger.info(f"MILCO (mine):  {sum(1 for s in usable if s.source == 'MILCO')}")
    logger.info(f"Kaggle:        {sum(1 for s in usable if s.source == 'KAGGLE')}")
    logger.info(f"Total usable:  {len(usable)}")
    logger.info(f"Augmented:     {sum(1 for s in usable if s.is_augmented)}")
    logger.info(f"Duplicates:    {dup_count}")
    logger.info("")
    for split in ["train", "val", "test"]:
        n = sum(1 for s in usable if s.final_split == split)
        logger.info(f"{split:5s}: {n} images")
    logger.info("")
    logger.info("Master classes (images):")
    for cid, cname in MASTER_CLASSES.items():
        cnt = sum(1 for s in usable if s.master_class_id == cid)
        logger.info(f"  {cid}: {cname} ({cnt} images)")
    logger.info("")
    if valid:
        logger.info("BUILD STATUS: PASS ✅")
    else:
        logger.info("BUILD STATUS: FAIL ❌")


if __name__ == "__main__":
    main()
