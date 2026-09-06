#!/usr/bin/env python3
"""
SonarVision — Build Multi-Source YOLO Dataset v5 (h8-only debris + 4 classes)
============================================================================

v4 → v5 changes (post-v4-run autopsy, driven by measured data):
  1. unknown_debris = the curated h8_data folder ONLY (NOAA-e4 source folder
     dropped entirely — the dim mode never scored on any test in v3/v4).
     NOMBO (class 5) is DROPPED as a class; NOMBO-only images are excluded
     from the dataset (unlabeling them as background would teach the model
     that mine-like echoes are background — contradictory signal for mine).
  2. drowning_victim DROPPED (19 originals; val/test were 3-original lotteries
     of near-duplicate noise copies). Kaggle = airplane + wreck only.
  3. CRITICAL SPLIT FIX — h8_data is re-split WHOLE-PASS, group-aware:
     the user's h8 train/val folders share 39/39 target passes (61/72 val
     positives come from passes in train) and the "h8_unseen_test" is a
     same-pass frame holdout (48/48 positives from passes in h8_data).
     Near-duplicate frames of one pass must land in ONE split (this is the
     anti-leakage rule from v2/v3/v4). h8_data is recombined and re-split
     70/15/15 with whole E3_H11833_TGT### runs, whole E4_TGT###_# passes,
     and block-50 groups for survey/BG frames. h8_unseen_test is NOT used
     as test (pass-level overlap) — documented in the report.
  4. MILCO re-split: finer groups (batch 15 vs 50) + enforced per-year
     70/15/15 so mine test stops being 76% 2015 while train is 21% 2015.
  5. Oversampling (airplane 3x, mine 2x) happens AFTER the split and only
     touches TRAIN — noise-aug copies can never enter val/test (v4 let
     augmented copies inherit group keys and land in val/test, inflating
     drowning AP50 to 0.57 with 3-origin test sets).

Master classes (4):
  0: unknown_debris   (h8_data only — one visual mode, brightness 42-50)
  1: airplane         (Kaggle)
  2: mine             (MILCO only)
  3: wreck            (Kaggle)

Usage:
    python scripts/build_multisource_dataset_v5.py \
        --output datasets/sonarvision_multisource_v5 \
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
    2: "mine",
    3: "wreck",
}

# NOAA: h8_data only (user-curated, single visual mode). All debris -> 0.
NOAA_MAPPING = {0: 0}
# MILCO: mine -> 2. NOMBO (class 1) DROPPED — nombo-only images excluded.
MILCO_MAPPING = {0: 2}
# Kaggle: airplane -> 1, wreck -> 3. drowning (1) and mine (2) DROPPED.
KAGGLE_MAPPING = {0: 1, 3: 3}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Oversampling (TRAIN-ONLY, applied AFTER the split — augs never reach
# val/test). Keyed by class_id -> multiplier.
OVERSAMPLE = {
    1: 3,   # airplane: 80 originals -> ~3x
    2: 2,   # mine: ~150 train originals -> ~2x
}

# MILCO group size (images per anti-leakage batch). v4 used 50, which made
# per-year 70/15/15 impossible for small classes (2015 mine test = 76% of
# test). 15 frames per batch keeps temporal near-duplicates together while
# letting the per-year quota be enforced.
MILCO_BATCH_SIZE = 15

# Noise augmentation config
NOISE_CONFIG = {
    'gaussian_sigmas': [0.01, 0.02, 0.03, 0.04, 0.05],
    'speckle_sigmas': [0.01, 0.02, 0.03],
    'brightness_factors': [0.8, 0.9, 1.1, 1.2],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_v5")


# ---------------------------------------------------------------------------
# Group keys (anti-leakage)
# ---------------------------------------------------------------------------
def h8_group_key(stem):
    """
    Group h8_data frames that are temporally correlated so the whole group
    lands in ONE split. Consecutive frames of the same target pass (or the
    same background survey line) are near-duplicates; scattering them across
    train/val/test leaks test answers into training.

    AUDITED: the user's original h8 train/val folders share 39/39 target
    passes (61/72 val positives from passes in train), and the hand-built
    h8_unseen_test draws 48/48 positives from passes already in h8_data.
    Whole-pass grouping below is what fixes that.

    Rules:
      E3_H11833_TGT###_<frame>  -> whole RUN is one group  (<= ~18 frames)
      E4_TGT###_<pass>_<frame>  -> whole PASS is one group (9 frames)
      E3_H11833_BG_####         -> temporally ordered frames; block by 50
      H11833_####               -> survey frames; block by 50
      G<clip>_<frame>           -> survey clips; block by 50
    """
    m = re.match(r"^(E3_H11833_TGT\d+)_\d+$", stem)
    if m:
        return f"NOAA_{m.group(1)}"
    m = re.match(r"^(E4_TGT\d+_\d+)_\d+$", stem)
    if m:
        return f"NOAA_{m.group(1)}"
    m = re.match(r"^E3_H11833_BG_(\d+)$", stem)
    if m:
        return f"NOAA_E3_BG_block{int(m.group(1)) // 50}"
    m = re.match(r"^H11833_(\d+)$", stem)
    if m:
        return f"NOAA_H11833_block{int(m.group(1)) // 50}"
    m = re.match(r"^G(\d+)_(\d+)$", stem)
    if m:
        return f"NOAA_G{m.group(1)}_block{int(m.group(2)) // 50}"
    return f"NOAA_{stem}"


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
    """Apply a random combination of sonar-safe noise to an image."""
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
def inventory_h8(output_dir):
    """Inventory the curated h8_data folder (user-built, single visual mode).

    Both h8/train and h8/val are combined into ONE pool and re-split
    whole-pass in group_aware_split — their original split shares 39/39
    target passes (near-duplicate frames across train/val).
    """
    logger.info("Inventorying NOAA h8_data (combined train+val, re-split)...")
    samples = []
    base = os.path.join("datasets", "noaa-debris", "h8")
    for sub in ["train", "val"]:
        img_dir = os.path.join(base, "images", sub)
        lbl_dir = os.path.join(base, "labels", sub)
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
                source="NOAA", source_detail="h8",
                original_id=stem, group_key=h8_group_key(stem),
            )
            boxes = read_yolo_label(lbl_path)
            sample.label_boxes = [(NOAA_MAPPING.get(c, -1), x, y, w, h)
                                  for c, x, y, w, h in boxes]
            valid = [c for c, _, _, _, _ in sample.label_boxes if c >= 0]
            sample.master_class_id = Counter(valid).most_common(1)[0][0] if valid else None
            samples.append(sample)
    logger.info(f"  h8_data total: {len(samples)} samples")
    return samples


def inventory_milco(output_dir):
    """Inventory MILCO dataset — mine only (NOMBO dropped entirely).

    NOMBO (class 1) lines are dropped; images whose ONLY labels were NOMBO
    are excluded from the dataset (as unlabeled background they would teach
    the model that mine-like echoes are background).
    """
    logger.info("Inventorying MILCO dataset (mine only, NOMBO dropped)...")
    samples = []
    base = os.path.join("datasets", "milco-nombo", "extracted")
    dropped_nombo_only = 0

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
            boxes = read_yolo_label(lbl_path)
            if not boxes:
                # background frame: keep (mine-sensor seafloor = hard negative)
                sample = SampleInfo(
                    image_path=img_path, label_path=lbl_path,
                    source="MILCO", source_detail=year,
                    original_id=stem, group_key="",
                )
                sample.label_boxes = []
                sample.master_class_id = None
                year_samples.append(sample)
                continue
            mapped = [(MILCO_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            valid = [b for b in mapped if b[0] >= 0]
            if boxes and not valid:
                # only NOMBO labels on this image -> exclude entirely
                dropped_nombo_only += 1
                continue
            sample = SampleInfo(
                image_path=img_path, label_path=lbl_path,
                source="MILCO", source_detail=year,
                original_id=stem, group_key="",
            )
            sample.label_boxes = valid
            if valid:
                sample.master_class_id = Counter(b[0] for b in valid).most_common(1)[0][0]
            else:
                sample.master_class_id = None
            year_samples.append(sample)

        # anti-leakage batches: FINER than v4's 50 so per-year 70/15/15 is
        # achievable (v4's 50-img batches made mine test 76% 2015).
        for i in range(0, len(year_samples), MILCO_BATCH_SIZE):
            batch = year_samples[i:i + MILCO_BATCH_SIZE]
            batch_idx = i // MILCO_BATCH_SIZE
            for s in batch:
                s.group_key = f"MILCO_{year}_batch{batch_idx}"
            samples.extend(batch)

    logger.info(f"  MILCO total: {len(samples)} samples "
                f"({dropped_nombo_only} nombo-only images excluded)")
    return samples


def inventory_kaggle(output_dir):
    """Inventory Kaggle SSS dataset — airplane + wreck only.

    drowning (1) and mine (2) labels dropped; images whose only labels were
    dropped classes are excluded (same rule that already fixed mines in v3).
    """
    logger.info("Inventorying Kaggle dataset (airplane + wreck only)...")
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

    logger.info(f"  Kaggle total: {len(samples)} samples "
                f"({dropped_only} drowning/mine-only images excluded)")
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
# Splitting (v4's stratified class-balanced group-aware split, unchanged)
# ---------------------------------------------------------------------------
def group_aware_split(samples, seed=42):
    """Stratified, class-balanced group-aware splitting (v4 logic).

    Splits each SOURCE x VISUAL-MODE stratum (NOAA-h8, MILCO-<year>,
    KAGGLE) so that every class present in the stratum gets ~70/15/15 of
    ITS OWN images across train/val/test. Groups (sequences/passes) are
    never split across splits (anti-leakage). Class floors guarantee every
    real class with enough groups appears in all three splits.
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
    assign = {}
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

        # class floors: rare modes must be learnable (train) and evaluated
        # (val + test) when they have enough groups. bg is exempt.
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
        logger.info(f"  {stratum:<14} images: "
                    + ", ".join(f"{sp}={sums[sp]}" for sp in splits))
        for c in sorted(c for c in classes if c != "bg"):
            logger.info(f"      {MASTER_CLASSES.get(c, 'bg'):<16}"
                        + ", ".join(f"{sp}={cnt[c][sp]}" for sp in splits))

    for gk, sp in assign.items():
        for s in groups[gk]:
            s.final_split = sp

    logger.info(f"  Split: train={total_counts['train']}, val={total_counts['val']}, "
                f"test={total_counts['test']}")
    logger.info(f"  Groups: {len(assign)} sequences/passes, all intact (0 leakage)")


# ---------------------------------------------------------------------------
# Oversampling — TRAIN-ONLY, after the split
# ---------------------------------------------------------------------------
def oversample_train_only(samples, output_dir):
    """Create noise-augmented copies of TRAIN-split images ONLY.

    v4 created augs BEFORE the split and inherited group keys, so augmented
    copies landed in val/test alongside their source (drowning val AP50 0.57
    on 24 imgs = 3 originals + 21 near-duplicate copies). v5 augs are created
    AFTER final_split is assigned and forced into train — val/test contain
    pure originals only.
    """
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
                group_key=src.group_key,          # same pass -> same split
                split_hint=src.split_hint,
            )
            aug_sample.master_class_id = class_id
            aug_sample.label_boxes = src.label_boxes[:]
            aug_sample.is_augmented = True
            aug_sample.augmented_from = src.image_path
            aug_sample.final_split = "train"       # FORCED — never val/test
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
            # Normalize ALL sources to 512x512 (h8_data is already 512).
            # YOLO labels are normalized so boxes stay valid — only the
            # pixel scale must be uniform for consistent object scales.
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
        "by_class_split": Counter((MASTER_CLASSES.get(s.master_class_id, "unknown"),
                                   s.final_split) for s in usable),
        "by_source_split": Counter((s.source, s.final_split) for s in usable),
        # v5 addition: per-source-detail (year / mode) x split — lets us
        # audit that mine test is no longer 2015-heavy.
        "by_detail_split": Counter((s.source_detail.split('/')[0].split('_')[0],
                                    s.final_split) for s in usable),
    }

    json_stats = {**stats}
    json_stats["by_source_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_source_split"].items()}
    json_stats["by_class_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_class_split"].items()}
    json_stats["by_detail_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_detail_split"].items()}

    json_path = os.path.join(report_dir, "dataset_audit_multisource_v5.json")
    with open(json_path, "w") as f:
        json.dump(json_stats, f, indent=2, default=str)

    md_path = os.path.join(report_dir, "sonarvision_multisource_v5_report.md")
    with open(md_path, "w") as f:
        f.write("# SonarVision Multi-Source Dataset v5 Report\n\n")
        f.write("## v4 → v5 Changes (post-v4-run autopsy)\n\n")
        f.write("- unknown_debris = curated h8_data ONLY (one visual mode, brightness 42-50). "
                "NOAA-e4 source folder dropped (dim mode never scored on any test in v3/v4).\n")
        f.write("- NOMBO dropped as a class; NOMBO-only images excluded entirely "
                "(unlabeled mine-like echoes would poison mine training).\n")
        f.write("- drowning_victim dropped (19 originals; val/test were 3-original "
                "near-duplicate lotteries). Kaggle = airplane + wreck.\n")
        f.write("- h8_data re-split WHOLE-PASS group-aware: the user's train/val folders "
                "shared 39/39 target passes (61/72 val positives from train passes) and "
                "h8_unseen_test drew 48/48 positives from passes in h8_data — near-duplicate "
                "frames of one pass now land in ONE split. h8_unseen_test is NOT used as test.\n")
        f.write("- MILCO re-split: finer groups (batch 15 vs 50) + per-year 70/15/15 so mine "
                "test stops being 76% 2015 while train is 21% 2015.\n")
        f.write("- Oversampling (airplane 3x, mine 2x) applied AFTER the split, TRAIN-ONLY — "
                "noise-aug copies never enter val/test.\n")
        f.write("- 4 classes: unknown_debris (h8) / airplane (Kaggle) / mine (MILCO) / "
                "wreck (Kaggle)\n\n")
        f.write("## Overall\n\n")
        f.write(f"- Total samples: {stats['total_samples']}\n")
        f.write(f"- Usable samples: {stats['usable_samples']}\n")
        f.write(f"- Original samples: {stats['original_count']}\n")
        f.write(f"- Augmented samples (train only): {stats['augmented_count']}\n")
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

        f.write("\n## Detail (year/mode) × Split\n\n")
        f.write("| Detail | Train | Val | Test |\n")
        f.write("|--------|-------|-----|------|\n")
        for d in sorted({k[0] for k in stats["by_detail_split"]}):
            tr = stats["by_detail_split"].get((d, "train"), 0)
            vl = stats["by_detail_split"].get((d, "val"), 0)
            te = stats["by_detail_split"].get((d, "test"), 0)
            f.write(f"| {d} | {tr} | {vl} | {te} |\n")

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
    parser = argparse.ArgumentParser(description="Build SonarVision multi-source dataset v5")
    parser.add_argument("--output", default="datasets/sonarvision_multisource_v5")
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
    all_samples.extend(inventory_h8(output_dir))
    all_samples.extend(inventory_milco(output_dir))
    all_samples.extend(inventory_kaggle(output_dir))
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
    logger.info("STEP 4: Stratified group-aware splitting (whole-pass)")
    logger.info("=" * 60)
    group_aware_split(all_samples, seed=args.seed)

    logger.info("=" * 60)
    logger.info("STEP 5: Oversampling (TRAIN-ONLY, post-split)")
    logger.info("=" * 60)
    augmented, aug_dir = oversample_train_only(all_samples, output_dir)
    all_samples.extend(augmented)

    logger.info("=" * 60)
    logger.info("STEP 6: Generating filenames")
    logger.info("=" * 60)
    generate_filenames(all_samples)

    logger.info("=" * 60)
    logger.info("STEP 7: Writing dataset")
    logger.info("=" * 60)
    write_dataset(all_samples, output_dir, aug_dir)

    logger.info("=" * 60)
    logger.info("STEP 8: Writing dataset.yaml")
    logger.info("=" * 60)
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write("# SonarVision Multi-Source Dataset v5\n")
        f.write("# Sensor-isolated classes: debris=h8 | mine=MILCO | airplane/wreck=Kaggle\n")
        f.write("# h8 whole-pass re-split (anti-leakage); oversampling train-only\n\n")
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: {len(MASTER_CLASSES)}\n")
        f.write(f"names:\n")
        for cid, cname in MASTER_CLASSES.items():
            f.write(f"  {cid}: {cname}\n")

    logger.info("=" * 60)
    logger.info("STEP 9: Generating reports")
    logger.info("=" * 60)
    generate_report(all_samples, output_dir, dup_count)

    logger.info("=" * 60)
    logger.info("STEP 10: Final validation")
    logger.info("=" * 60)
    valid = validate_dataset(output_dir)

    usable = [s for s in all_samples if not s.is_duplicate and not s.is_corrupt]
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET V5 BUILD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"NOAA (h8):  {sum(1 for s in usable if s.source == 'NOAA')}")
    logger.info(f"MILCO:      {sum(1 for s in usable if s.source == 'MILCO')}")
    logger.info(f"Kaggle:     {sum(1 for s in usable if s.source == 'KAGGLE')}")
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