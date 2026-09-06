#!/usr/bin/env python3
"""
SonarVision — Build Multi-Source YOLO Dataset v4 (FINAL)
==========================================================

v3 → v4 changes — driven by the v3 training post-mortem:
  The v3 model converged at mAP50 ~0.45 but per-class AP50 showed the two
  real failures (all from the v3 run's eval):
    - unknown_debris 0.058 val / 0.000 test — the ONLY mixed-sensor class
      (NOAA h8 + NOAA e4 + MILCO NOMBO = THREE visual modes: brightness
      46/74/53, contrast 19/32/31) while every single-identity class
      worked (airplane .81, wreck .76, drowning .24-.33, mine .43 val).
    - mine 0.432 val / 0.000 test — the v3 random group split put 88% of
      mine images in train and ZERO 2015 mines (the dominant look, 111
      imgs) in val/test, so mine was only evaluated on unseen 2010/2018
      batches.

v4 fixes:
  1. NOMBO becomes its OWN class (5: nombo_contact), NOT unknown_debris.
     unknown_debris = NOAA only (e4 + h8). One visual identity per class,
     applied consistently (the same rule that already fixed mines).
  2. STRATIFIED group split: each SOURCE x VISUAL-MODE stratum (NOAA-e4,
     NOAA-h8, MILCO-<year>, KAGGLE) is split 70/15/15 independently so
     every class's dominant look has representative val/test coverage
     (incl. 2015 mines + h8 debris). Sequences stay intact (0 leakage).
  3. Kept from v3: Kaggle mines dropped (bimodal), NOAA sequence-aware
     grouping, augmented copies inherit source group, junk-box filter,
     512x512 normalize, deterministic seed.

Master classes (6):
  0: unknown_debris   (NOAA e4 + h8 marine_debris)
  1: airplane         (Kaggle)
  2: drowning_victim  (Kaggle — oversampled)
  3: mine             (MILCO only)
  4: wreck            (Kaggle wreck/shipwreck)
  5: nombo_contact    (MILCO NOMBO — mine-like echoes, own class)

Usage:
    python scripts/build_multisource_dataset_v4.py \
        --output datasets/sonarvision_multisource_v4 \
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
from PIL import Image, ImageEnhance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MASTER_CLASSES = {
    0: "unknown_debris",
    1: "airplane",
    2: "drowning_victim",
    3: "mine",
    4: "wreck",
    5: "nombo_contact",
}

NOAA_MAPPING = {0: 0}  # marine_debris -> unknown_debris
# v4: NOMBO is its OWN class (5), NOT unknown_debris. The v3 mixed
# unknown_debris = NOAA(h8: bright 74/contrast 32) + NOAA(e4: dim 46/
# contrast 19) + NOMBO (mid/high-contrast, 3x bigger) was TRI-MODAL and
# the model could not learn any debris mode (AP50 0.058 val / 0.000 test)
# while every single-identity class worked (airplane .81, wreck .76,
# drowning .24-.33, mine .43). One visual identity per class, again.
MILCO_MAPPING = {0: 3, 1: 5}  # MILCO -> mine, NOMBO -> nombo_contact
# Kaggle class 2 (mine) is intentionally DROPPED in v3: mine = MILCO only.
# Mixing Kaggle renders with real MILCO mine contacts produced a bimodal
# class (crop corr 0.26, opposite AR, 19x scale) that confused training.
KAGGLE_MAPPING = {0: 1, 1: 2, 3: 4}  # airplane, drowning, wreck

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Oversampling targets (class_id: multiplier)
OVERSAMPLE = {
    2: 10,   # drowning_victim: 19 images → ~190
    1: 3,    # airplane: 80 images → ~240
}

# Noise augmentation config
NOISE_CONFIG = {
    'gaussian_sigmas': [0.01, 0.02, 0.03, 0.04, 0.05],
    'speckle_sigmas': [0.01, 0.02, 0.03],
    'brightness_factors': [0.8, 0.9, 1.1, 1.2],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_v4")


# ---------------------------------------------------------------------------
# Group keys (anti-leakage)
# ---------------------------------------------------------------------------
def noaa_group_key(stem):
    """
    Group NOAA sonar frames that are temporally correlated so the whole group
    lands in ONE split. Consecutive frames of the same target pass (or the
    same background survey) are near-duplicates; scattering them across
    train/val/test leaks test answers into training.

    Rules (e4 + h8 sources share the same E4/E3/H11833 naming):
      E4_TGT###_<pass>_<frame>  -> whole pass is one group  (9 frames)
      E3_H11833_TGT###_<frame>  -> whole run is one group   (<= ~20 frames)
      E4_BG_#### / E3_BG_#### / H11833_#### -> temporally ordered frames;
        block by index (50 per block) since they are sampled from long videos.
    """
    m = re.match(r"^(E4_TGT\d+_\d+)_\d+$", stem)
    if m:
        return f"NOAA_{m.group(1)}"
    m = re.match(r"^(E3_H11833_TGT\d+)_\d+$", stem)
    if m:
        return f"NOAA_{m.group(1)}"
    m = re.match(r"^E4_BG_(\d+)$", stem)
    if m:
        return f"NOAA_E4_BG_block{int(m.group(1)) // 50}"
    m = re.match(r"^E3_H11833_BG_(\d+)$", stem)
    if m:
        return f"NOAA_E3_BG_block{int(m.group(1)) // 50}"
    m = re.match(r"^H11833_(\d+)$", stem)
    if m:
        return f"NOAA_H11833_block{int(m.group(1)) // 50}"
    # h8 survey clips: G<clip>_<frame> (e.g. G7_000000..G7_007298 are
    # consecutively sampled frames of ONE long survey line). Without this,
    # every frame was its own group and near-duplicate seafloor frames
    # scattered across train/val/test (~2300 imgs). Block by 50 like the
    # other background surveys.
    m = re.match(r"^G(\d+)_(\d+)$", stem)
    if m:
        return f"NOAA_G{m.group(1)}_block{int(m.group(2)) // 50}"
    return f"NOAA_{stem}"


# ---------------------------------------------------------------------------
# Noise augmentation pipeline
# ---------------------------------------------------------------------------
def add_gaussian_noise(img_array, sigma=0.03):
    """Add Gaussian noise to image array (0-255)."""
    noise = np.random.normal(0, sigma * 255, img_array.shape)
    return np.clip(img_array + noise, 0, 255).astype(np.uint8)


def add_speckle_noise(img_array, sigma=0.02):
    """Add speckle (multiplicative) noise."""
    noise = np.random.normal(1, sigma, img_array.shape)
    return np.clip(img_array * noise, 0, 255).astype(np.uint8)


def adjust_brightness(img_array, factor=1.1):
    """Adjust brightness."""
    return np.clip(img_array * factor, 0, 255).astype(np.uint8)


def apply_noise_augmentation(img_path, output_path, variant_idx):
    """Apply a random combination of sonar-safe noise to an image.
    Returns the path to the augmented image."""
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)

    # Apply 1-2 random augmentations
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
        # At least apply Gaussian noise
        sigma = random.choice(NOISE_CONFIG['gaussian_sigmas'])
        arr = add_gaussian_noise(arr, sigma)
        augmentations.append(f"g{sigma}")

    aug_img = Image.fromarray(arr)
    aug_img.save(output_path, 'PNG')
    return output_path


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Minimum box dimension (normalized) worth keeping. Junk annotation lines
# (e.g. w=0.000005, a ~0.003px box pinned at x=0) appear in a few MILCO
# source files; they are unusable at 512x512 and only pollute the dataset.
# 0.002 normalized ~= 1px at 512 / 1.3px at 640 — real annotations are larger.
MIN_BOX_DIM = 0.002


def read_yolo_label(path):
    """Read YOLO label file, handle concatenated boxes."""
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
def inventory_noaa(output_dir):
    """Inventory NOAA e4 + h8 datasets."""
    logger.info("Inventorying NOAA datasets (e4 + h8)...")
    samples = []

    # --- NOAA e4 ---
    e4_base = "datasets/noaa-debris/e4"
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(e4_base, "images", split)
        lbl_dir = os.path.join(e4_base, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="NOAA", source_detail=f"e4/{split}",
                original_id=stem, group_key=noaa_group_key(stem),
                split_hint=split,
            )
            boxes = read_yolo_label(lbl_path)
            sample.label_boxes = [(NOAA_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            valid = [c for c, _, _, _, _ in sample.label_boxes if c >= 0]
            sample.master_class_id = Counter(valid).most_common(1)[0][0] if valid else None
            samples.append(sample)

    # --- NOAA h8 ---
    h8_base = "datasets/noaa-debris/h8"
    for split in ["train", "val"]:
        img_dir = os.path.join(h8_base, "images", split)
        lbl_dir = os.path.join(h8_base, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="NOAA", source_detail=f"h8/{split}",
                original_id=stem, group_key=noaa_group_key(stem),
                split_hint=split,
            )
            boxes = read_yolo_label(lbl_path)
            sample.label_boxes = [(NOAA_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            valid = [c for c, _, _, _, _ in sample.label_boxes if c >= 0]
            sample.master_class_id = Counter(valid).most_common(1)[0][0] if valid else None
            samples.append(sample)

    logger.info(f"  NOAA total: {len(samples)} samples")
    return samples


def inventory_milco(output_dir):
    """Inventory MILCO-NOMBO dataset."""
    logger.info("Inventorying MILCO-NOMBO dataset...")
    samples = []
    base = "datasets/milco-nombo/extracted"
    BATCH_SIZE = 50

    for year in ["2010", "2015", "2017", "2018", "2021"]:
        year_path = os.path.join(base, year, year)
        if not os.path.isdir(year_path):
            year_path = os.path.join(base, year)
            if not os.path.isdir(year_path):
                continue

        year_samples = []
        for fname in sorted(os.listdir(year_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(year_path, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(year_path, stem + ".txt")
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="MILCO", source_detail=year,
                original_id=stem, group_key=f"MILCO_{year}_batch0",
                split_hint="",
            )
            year_samples.append(sample)

        for i in range(0, len(year_samples), BATCH_SIZE):
            batch = year_samples[i:i+BATCH_SIZE]
            batch_idx = i // BATCH_SIZE
            for s in batch:
                s.group_key = f"MILCO_{year}_batch{batch_idx}"
                boxes = read_yolo_label(s.label_path)
                s.label_boxes = [(MILCO_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
                if boxes:
                    mapped_classes = [MILCO_MAPPING.get(c, -1) for c, _, _, _, _ in boxes]
                    valid = [c for c in mapped_classes if c >= 0]
                    if valid:
                        s.master_class_id = Counter(valid).most_common(1)[0][0]
                    else:
                        s.master_class_id = None
                else:
                    s.master_class_id = None
            samples.extend(batch)

    logger.info(f"  MILCO total: {len(samples)} samples")
    return samples


def inventory_kaggle(output_dir):
    """Inventory Kaggle SSS dataset."""
    logger.info("Inventorying Kaggle SSS dataset...")
    samples = []
    base = "datasets/kaggle_sss"

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
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="KAGGLE", source_detail=f"SSS/{split}",
                original_id=stem, group_key=f"KAGGLE_{stem}",
                split_hint=split,
            )
            boxes = read_yolo_label(lbl_path)
            mapped_boxes = [(KAGGLE_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            valid_boxes = [b for b in mapped_boxes if b[0] >= 0]
            if boxes and not valid_boxes:
                # v3: Kaggle images whose ONLY labels were mines (class 2) are
                # dropped entirely. Keeping them as unlabeled background would
                # teach the model that mine-looking echoes are false positives
                # while MILCO mines are positives — a contradictory signal.
                logger.info(f"    Dropping Kaggle mine-only image: {stem}")
                continue
            sample.label_boxes = mapped_boxes
            if valid_boxes:
                sample.master_class_id = Counter(b[0] for b in valid_boxes).most_common(1)[0][0]
            else:
                sample.master_class_id = None
            samples.append(sample)

    logger.info(f"  Kaggle total: {len(samples)} samples")
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
    if os.path.getsize(sample.image_path) == 0:
        issues.append("zero_byte_image")
        sample.is_corrupt = True
        return issues
    try:
        img = Image.open(sample.image_path)
        sample.width, sample.height = img.size
        img.close()
    except Exception:
        issues.append("unreadable_image")
        sample.is_corrupt = True
        return issues
    if sample.width < 10 or sample.height < 10:
        issues.append("too_small_image")
        sample.is_corrupt = True
    if not os.path.exists(sample.label_path):
        issues.append("missing_label")
        return issues
    for cls_id, xc, yc, w, h in sample.label_boxes:
        if cls_id < 0 or cls_id >= len(MASTER_CLASSES):
            issues.append(f"invalid_class_{cls_id}")
        if w <= 0 or h <= 0:
            issues.append("zero_area_box")
    return issues


def detect_duplicates(samples):
    logger.info("Detecting duplicates...")
    hash_map = defaultdict(list)
    dup_count = 0
    for s in samples:
        if s.is_corrupt:
            continue
        s.sha256 = sha256_file(s.image_path)
        hash_map[s.sha256].append(s)
    for h, group in hash_map.items():
        if len(group) > 1:
            for s in group[1:]:
                s.is_duplicate = True
                s.duplicate_of = group[0].image_path
                dup_count += 1
    logger.info(f"  Found {dup_count} duplicates")
    return dup_count


# ---------------------------------------------------------------------------
# Oversampling with noise augmentation
# ---------------------------------------------------------------------------
def oversample_minority_classes(samples, output_dir):
    """Oversample minority classes by creating noise-augmented copies."""
    logger.info("Oversampling minority classes...")
    augmented = []

    # Group samples by class
    by_class = defaultdict(list)
    for s in samples:
        if not s.is_duplicate and not s.is_corrupt and s.master_class_id is not None:
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
        logger.info(f"  {class_name}: {len(class_samples)} originals → target {target_total} (+{need} augmented)")

        for i in range(need):
            src = random.choice(class_samples)
            # Create augmented copy
            stem = os.path.splitext(os.path.basename(src.image_path))[0]
            aug_img_name = f"{stem}_aug{i:04d}.png"
            aug_lbl_name = f"{stem}_aug{i:04d}.txt"
            aug_img_path = os.path.join(aug_dir, aug_img_name)
            aug_lbl_path = os.path.join(aug_dir, aug_lbl_name)

            # Apply noise augmentation
            apply_noise_augmentation(src.image_path, aug_img_path, i)

            # Copy label
            shutil.copy2(src.label_path, aug_lbl_path)

            # Create sample info.
            # CRITICAL: inherit the SOURCE group_key so the augmented copy is
            # forced into the same split as its source image. Giving it a fresh
            # group key lets near-duplicates (source vs noise copy) land in
            # different splits, leaking train images into val/test.
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
            augmented.append(aug_sample)

    logger.info(f"  Created {len(augmented)} augmented samples")
    return augmented, aug_dir


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def group_aware_split(samples, seed=42):
    """Stratified, CLASS-BALANCED group-aware splitting (v4).

    Why not the v3 random/pattern assignment:
      - mine AP50 0.432 val -> 0.000 test: 88% of mine images landed in
        train and ZERO 2015 mines (the dominant look) reached val/test,
        so mine was only evaluated on unseen 2010/2018 batches.
      - unknown_debris (then mixed NOAA+NOMBO) val/test coverage was also
        skewed, muddying the tri-modal diagnosis.

    v4 splits each SOURCE x VISUAL-MODE stratum (NOAA-e4, NOAA-h8,
    MILCO-<year>, KAGGLE) so that EVERY class present in the stratum gets
    ~70/15/15 of ITS OWN images across train/val/test — not a global
    ratio that heterogeneous group sizes (1..50 imgs) can skew. Group
    counts alone also cannot work: e.g. NOAA-h8 debris lives in only 20
    TGT-run groups among 2300 background frames.

    Mechanics per stratum:
      1. shuffle groups; compute each group's class set from its boxes
         ("bg" pseudo-class when a group has no boxes)
      2. greedily assign each group to the split with the largest relative
         remaining quota summed over the classes the group contains
      3. class floors: any real class with >=2 groups gets >=1 group in
         train; with >=3 groups it also gets >=1 group in val and test
         (move the smallest qualifying group if a split was starved)
    Sequences (groups) are never split across splits (anti-leakage).
    """
    logger.info("Performing stratified, class-balanced group-aware splitting...")
    rng = random.Random(seed)
    usable = [s for s in samples if not s.is_duplicate and not s.is_corrupt]

    groups = defaultdict(list)
    for s in usable:
        groups[s.group_key].append(s)

    def stratum_of(s):
        if s.source == "NOAA":
            return f"NOAA-{s.source_detail.split('/')[0]}"
        if s.source == "MILCO":
            return f"MILCO-{s.group_key.split('_')[1]}"
        return "KAGGLE"

    def group_class_set(gs):
        cs = set()
        for s in gs:
            for cls_id, *_ in s.label_boxes:
                if 0 <= cls_id < len(MASTER_CLASSES):
                    cs.add(cls_id)
        if not cs:
            cs.add("bg")
        return cs

    by_stratum = defaultdict(list)
    for gk, gs in groups.items():
        by_stratum[stratum_of(gs[0])].append((gk, gs))

    splits = ("train", "val", "test")
    ratio = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
    assign = {}  # group_key -> split
    total_counts = {"train": 0, "val": 0, "test": 0}

    for stratum in sorted(by_stratum):
        gk_list = list(by_stratum[stratum])
        rng.shuffle(gk_list)
        cls_of = {gk: group_class_set(gs) for gk, gs in gk_list}
        classes = sorted({c for cs in cls_of.values() for c in cs}, key=str)
        tot = {c: sum(len(gs) for gk, gs in gk_list if c in cls_of[gk]) for c in classes}
        tgt = {c: {sp: tot[c] * ratio[sp] for sp in splits} for c in classes}
        cnt = {c: {sp: 0 for sp in splits} for c in classes}

        def cost(gk, sp):
            gsize = len(groups[gk])
            return sum((cnt[c][sp] + gsize - tgt[c][sp]) / max(tgt[c][sp], 1.0)
                       for c in cls_of[gk])

        for gk, gs in gk_list:
            sp = min(splits, key=lambda s: cost(gk, s))
            assign[gk] = sp
            for c in cls_of[gk]:
                cnt[c][sp] += len(gs)

        def move(gk, to_sp):
            from_sp = assign[gk]
            assign[gk] = to_sp
            for c in cls_of[gk]:
                cnt[c][from_sp] -= len(groups[gk])
                cnt[c][to_sp] += len(groups[gk])

        # class floors: a rare mode must be learnable (train) and, when it
        # has enough groups, evaluated (val + test). bg is exempt.
        # Donor preference (all three floors): groups that contain ONLY this
        # class first (moving them cannot disturb another class's coverage),
        # then groups currently in train (the 70% surplus), then the smallest
        # remaining candidate. Without the pure-first rule, MILCO-2010-style
        # overlap (one group holding both mine + nombo) lets one class's fill
        # undo another's (val 0 again after the next class's train floor).
        for c in [c for c in classes if c != "bg"]:
            c_groups = [gk for gk, _ in gk_list if c in cls_of[gk]]
            if len(c_groups) < 2:
                continue
            for need in ("train", "val", "test"):
                if cnt[c][need] != 0:
                    continue
                if need == "train" and len(c_groups) < 2:
                    continue
                if need in ("val", "test") and len(c_groups) < 3:
                    continue
                donors = [gk for gk in c_groups if assign[gk] != need]
                if not donors:
                    continue
                pick = min(donors, key=lambda gk: (
                    cls_of[gk] != {c},        # 1. prefer pure groups
                    assign[gk] != "train",    # 2. prefer train donors
                    len(groups[gk]),           # 3. smallest
                ))
                move(pick, need)

        for gk, gs in gk_list:
            total_counts[assign[gk]] += len(gs)
        sums = {sp: sum(len(gs) for gk, gs in gk_list if assign[gk] == sp)
                for sp in splits}
        logger.info(f"  {stratum:<14} images: " +
                    ", ".join(f"{sp}={sums[sp]}" for sp in splits))
        for c in sorted(c for c in classes if c != "bg"):
            logger.info(f"      {MASTER_CLASSES.get(c, 'bg'):<16}" +
                        ", ".join(f"{sp}={cnt[c][sp]}" for sp in splits))

    for gk, sp in assign.items():
        for s in groups[gk]:
            s.final_split = sp

    logger.info(f"  Split: train={total_counts['train']}, val={total_counts['val']}, "
                f"test={total_counts['test']}")
    logger.info(f"  Groups: {len(assign)} sequences, all intact (0 leakage)")


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
            # Normalize ALL sources to 512x512. Source natives differ
            # (KAGGLE 640, MILCO 416/1024, NOAA 512/1024); YOLO labels are
            # normalized so boxes stay valid — only the pixel scale must be
            # uniform for consistent object scales at imgsz=512. All sources
            # are square, so a plain resize introduces no distortion.
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
            # Persist the group key so split integrity is auditable post-build.
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
        split_samples = [s for s in samples if s.final_split == split and not s.is_duplicate and not s.is_corrupt]
        with open(manifest_path, "w") as f:
            for s in split_samples:
                f.write(f"images/{split}/{s.new_filename}\n")

    logger.info(f"  Written {written} samples")

    # Cleanup aug temp dir
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(samples, output_dir, dup_count):
    logger.info("Generating reports...")
    usable = [s for s in samples if not s.is_duplicate and not s.is_corrupt]
    report_dir = os.path.join(os.path.dirname(output_dir), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)

    stats = {
        "total_samples": len(samples),
        "usable_samples": len(usable),
        "duplicate_count": dup_count,
        "augmented_count": sum(1 for s in usable if s.is_augmented),
        "original_count": sum(1 for s in usable if not s.is_augmented),
        "by_source": Counter(s.source for s in usable),
        "by_split": Counter(s.final_split for s in usable),
        "by_class": Counter(MASTER_CLASSES.get(s.master_class_id, "unknown") for s in usable),
        "by_class_split": Counter((MASTER_CLASSES.get(s.master_class_id, "unknown"), s.final_split) for s in usable),
        "by_source_split": Counter((s.source, s.final_split) for s in usable),
    }

    json_stats = {**stats}
    json_stats["by_source_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_source_split"].items()}
    json_stats["by_class_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_class_split"].items()}

    # JSON
    json_path = os.path.join(report_dir, "dataset_audit_multisource_v4.json")
    with open(json_path, "w") as f:
        json.dump(json_stats, f, indent=2, default=str)

    # Markdown
    md_path = os.path.join(report_dir, "sonarvision_multisource_v4_report.md")
    with open(md_path, "w") as f:
        f.write("# SonarVision Multi-Source Dataset v4 Report\n\n")
        f.write("## v3 → v4 Changes (post-mortem fixes)\n\n")
        f.write("- mine = MILCO ONLY (Kaggle mines dropped — bimodal class, corr 0.26)\n")
        f.write("- NOMBO = own class 5 (nombo_contact) — was mixed into debris, tri-modal failure\n")
        f.write("- stratified group split: representative val/test per source x mode (2015 mines, h8 debris)\n")
        f.write("- unknown_debris = NOAA ONLY (e4 + h8) — NOMBO moved to its own class 5\n")
        f.write("- airplane / drowning_victim / wreck = Kaggle\n")
        f.write("- Leakage-free grouping + 512x512 normalize kept from v2\n\n")
        f.write("## Overall\n\n")
        f.write(f"- Total samples: {stats['total_samples']}\n")
        f.write(f"- Usable samples: {stats['usable_samples']}\n")
        f.write(f"- Original samples: {stats['original_count']}\n")
        f.write(f"- Augmented samples: {stats['augmented_count']}\n")
        f.write(f"- Duplicates removed: {stats['duplicate_count']}\n\n")

        f.write("## By Source\n\n")
        for src, cnt in sorted(stats["by_source"].items()):
            f.write(f"- {src}: {cnt}\n")

        f.write("\n## By Split\n\n")
        for split, cnt in sorted(stats["by_split"].items()):
            f.write(f"- {split}: {cnt}\n")

        f.write("\n## By Class\n\n")
        for cls, cnt in sorted(stats["by_class"].items()):
            f.write(f"- {cls}: {cnt}\n")

        f.write("\n## Class × Split\n\n")
        f.write("| Class | Train | Val | Test |\n")
        f.write("|-------|-------|-----|------|\n")
        for cls_name in MASTER_CLASSES.values():
            tr = stats["by_class_split"].get((cls_name, "train"), 0)
            vl = stats["by_class_split"].get((cls_name, "val"), 0)
            te = stats["by_class_split"].get((cls_name, "test"), 0)
            f.write(f"| {cls_name} | {tr} | {vl} | {te} |\n")

        f.write("\n## Source × Split\n\n")
        f.write("| Source | Train | Val | Test |\n")
        f.write("|--------|-------|-----|------|\n")
        for src in ["NOAA", "MILCO", "KAGGLE"]:
            tr = stats["by_source_split"].get((src, "train"), 0)
            vl = stats["by_source_split"].get((src, "val"), 0)
            te = stats["by_source_split"].get((src, "test"), 0)
            f.write(f"| {src} | {tr} | {vl} | {te} |\n")

    logger.info(f"  Reports written to {report_dir}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(output_dir):
    logger.info("Running final validation...")
    all_ok = True
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        if not os.path.isdir(img_dir):
            logger.error(f"Missing directory: {img_dir}")
            all_ok = False
            continue
        images = set(os.listdir(img_dir))
        labels = set(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else set()
        for img in images:
            lbl = img.replace(".png", ".txt")
            if lbl not in labels:
                logger.error(f"Missing label for {split}/{img}")
                all_ok = False
        for lbl in labels:
            img = lbl.replace(".txt", ".png")
            if img not in images:
                logger.error(f"Missing image for {split}/{lbl}")
                all_ok = False
        logger.info(f"  {split}: {len(images)} images, {len(labels)} labels")

    if all_ok:
        logger.info("  ✅ All validation checks passed!")
    else:
        logger.error("  ❌ Validation FAILED")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build SonarVision multi-source dataset v4 (final)")
    parser.add_argument("--output", default="datasets/sonarvision_multisource_v4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    logger.info(f"Output: {output_dir}")

    # Deterministic rebuild: seed both RNGs so noise augmentation and
    # group-aware splitting reproduce exactly across runs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Inventory
    logger.info("=" * 60)
    logger.info("STEP 1: Inventorying data sources")
    logger.info("=" * 60)
    all_samples = []
    all_samples.extend(inventory_noaa(output_dir))
    all_samples.extend(inventory_milco(output_dir))
    all_samples.extend(inventory_kaggle(output_dir))
    logger.info(f"Total inventoried: {len(all_samples)}")

    # Step 2: Quality checks
    logger.info("=" * 60)
    logger.info("STEP 2: Quality checks")
    logger.info("=" * 60)
    for s in all_samples:
        s.issues.extend(check_sample_quality(s))

    # Step 3: Duplicate detection
    logger.info("=" * 60)
    logger.info("STEP 3: Duplicate detection")
    logger.info("=" * 60)
    dup_count = detect_duplicates(all_samples)

    # Step 4: Oversample minority classes
    logger.info("=" * 60)
    logger.info("STEP 4: Oversampling minority classes")
    logger.info("=" * 60)
    augmented, aug_dir = oversample_minority_classes(all_samples, output_dir)
    all_samples.extend(augmented)

    # Step 5: Generate filenames
    logger.info("=" * 60)
    logger.info("STEP 5: Generating filenames")
    logger.info("=" * 60)
    generate_filenames(all_samples)

    # Step 6: Split
    logger.info("=" * 60)
    logger.info("STEP 6: Stratified group-aware splitting")
    logger.info("=" * 60)
    group_aware_split(all_samples, seed=args.seed)

    # Step 7: Write dataset
    logger.info("=" * 60)
    logger.info("STEP 7: Writing dataset")
    logger.info("=" * 60)
    write_dataset(all_samples, output_dir, aug_dir)

    # Step 8: Write dataset.yaml
    logger.info("=" * 60)
    logger.info("STEP 8: Writing dataset.yaml")
    logger.info("=" * 60)
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# SonarVision Multi-Source Dataset v4 (FINAL)\n")
        f.write(f"# Sensor-isolated classes: mine=MILCO | debris=NOAA | nombo=own class |\n")
        f.write(f"# airplane/drowning/wreck=Kaggle\n")
        f.write(f"# Oversampled: drowning_victim 10x, airplane 3x\n")
        f.write(f"# Noise pipeline: Gaussian + speckle + brightness\n\n")
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: {len(MASTER_CLASSES)}\n")
        f.write(f"names:\n")
        for cid, cname in MASTER_CLASSES.items():
            f.write(f"  {cid}: {cname}\n")

    # Step 9: Reports
    logger.info("=" * 60)
    logger.info("STEP 9: Generating reports")
    logger.info("=" * 60)
    generate_report(all_samples, output_dir, dup_count)

    # Step 10: Validation
    logger.info("=" * 60)
    logger.info("STEP 10: Final validation")
    logger.info("=" * 60)
    valid = validate_dataset(output_dir)

    # Summary
    usable = [s for s in all_samples if not s.is_duplicate and not s.is_corrupt]
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET V4 BUILD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"NOAA (e4+h8):  {sum(1 for s in usable if s.source == 'NOAA')}")
    logger.info(f"MILCO:         {sum(1 for s in usable if s.source == 'MILCO')}")
    logger.info(f"Kaggle:        {sum(1 for s in usable if s.source == 'KAGGLE')}")
    logger.info(f"Total usable:  {len(usable)}")
    logger.info(f"Augmented:     {sum(1 for s in usable if s.is_augmented)}")
    logger.info(f"Duplicates:    {dup_count}")
    logger.info("")
    for split in ["train", "val", "test"]:
        n = sum(1 for s in usable if s.final_split == split)
        logger.info(f"{split:5s}: {n} images")
    logger.info("")
    logger.info("Master classes (boxes):")
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
