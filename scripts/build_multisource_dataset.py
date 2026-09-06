#!/usr/bin/env python3
"""
SonarVision — Build Multi-Source YOLO Training/Validation/Test Dataset
=====================================================================

Combines NOAA, MILCO-NOMBO, and Kaggle Side-Scan Sonar data into a single
leakage-safe, deduplicated, domain-aware YOLO dataset.

Master classes:
  0: unknown_debris   (NOAA marine_debris + MILCO-NOMBO NOMBO)
  1: airplane         (Kaggle)
  2: drowning_victim  (Kaggle)
  3: mine             (Kaggle mine + MILCO-NOMBO MILCO)
  4: wreck            (Kaggle wreck/shipwreck)

Usage:
    python scripts/build_multisource_dataset.py \
        --output datasets/sonarvision_multisource_v1 \
        --seed 42
"""

import os
import sys
import json
import csv
import hashlib
import shutil
import argparse
import random
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import struct
import io

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MASTER_CLASSES = {
    0: "unknown_debris",
    1: "airplane",
    2: "drowning_victim",
    3: "mine",
    4: "wreck",
}

CLASS_TO_ID = {v: k for k, v in MASTER_CLASSES.items()}

# Source-to-master class mappings
NOAA_MAPPING = {
    0: 0,  # marine_debris -> unknown_debris
}

MILCO_MAPPING = {
    0: 3,  # MILCO (mine-like contacts) -> mine
    1: 0,  # NOMBO (non-mine objects) -> unknown_debris
}

KAGGLE_MAPPING = {
    0: 1,  # airplane -> airplane
    1: 2,  # drowning victim -> drowning_victim
    2: 3,  # mine -> mine
    3: 4,  # wreck/shipwreck -> wreck
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_multisource")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def simple_phash(img_path: str, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash (average hash) for near-duplicate detection."""
    try:
        img = Image.open(img_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        return bits
    except Exception:
        return ""


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    if len(h1) != len(h2):
        return max(len(h1), len(h2))
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def read_yolo_label(path: str) -> List[Tuple[int, float, float, float, float]]:
    """Read a YOLO label file and return list of (class_id, x_c, y_c, w, h).
    
    Handles concatenated bounding boxes (multiple boxes on one line)
    by detecting groups of 5 values that start with an integer class ID.
    """
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
            
            # Try to parse as standard YOLO (5 values per line)
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls, xc, yc, w, h))
                except (ValueError, IndexError):
                    continue
            else:
                # Handle concatenated boxes: parse greedily in groups of 5
                i = 0
                while i + 4 < len(parts):
                    try:
                        cls = int(parts[i])
                        xc = float(parts[i + 1])
                        yc = float(parts[i + 2])
                        w = float(parts[i + 3])
                        h = float(parts[i + 4])
                        # Validate the values are in reasonable range for YOLO
                        if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            boxes.append((cls, xc, yc, w, h))
                            i += 5
                        else:
                            # Not a valid box, skip this value
                            i += 1
                    except (ValueError, IndexError):
                        i += 1
    return boxes


def get_image_info(path: str) -> Optional[Dict]:
    """Get image metadata."""
    try:
        img = Image.open(path)
        info = {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "channels": len(img.getbands()),
            "format": img.format,
        }
        img.close()
        return info
    except Exception as e:
        logger.warning(f"Cannot read image {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data source inventories
# ---------------------------------------------------------------------------
class SampleInfo:
    """Represents one image+label pair with full provenance."""
    def __init__(self, image_path: str, label_path: str, source: str, source_detail: str,
                 original_id: str, group_key: str, split_hint: str = ""):
        self.image_path = image_path
        self.label_path = label_path
        self.source = source  # "NOAA", "MILCO", "KAGGLE"
        self.source_detail = source_detail  # e.g., "2015", "e4"
        self.original_id = original_id
        self.group_key = group_key  # for scene/leakage-aware splitting
        self.split_hint = split_hint  # original split if known
        self.master_class_id: Optional[int] = None
        self.source_class_id: Optional[int] = None
        self.new_filename: str = ""
        self.new_label_filename: str = ""
        self.final_split: str = ""
        self.sha256: str = ""
        self.phash: str = ""
        self.width: int = 0
        self.height: int = 0
        self.channels: int = 0
        self.is_corrupt: bool = False
        self.is_duplicate: bool = False
        self.duplicate_of: str = ""
        self.issues: List[str] = []
        self.label_boxes: List[Tuple] = []


def inventory_noaa(output_dir: str) -> List[SampleInfo]:
    """Inventory the NOAA e4 dataset."""
    logger.info("Inventorying NOAA e4 dataset...")
    samples = []
    base = "datasets/noaa-debris/e4"

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base, "images", split)
        lbl_dir = os.path.join(base, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            sample = SampleInfo(
                image_path=img_path,
                label_path=lbl_path,
                source="NOAA",
                source_detail=f"e4/{split}",
                original_id=stem,
                group_key=f"NOAA_e4_{stem}",  # each crop is individual
                split_hint=split,
            )
            # Map class 0 -> master class 0 (unknown_debris)
            boxes = read_yolo_label(lbl_path)
            sample.label_boxes = [(NOAA_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
            sample.master_class_id = 0  # NOAA only has class 0
            samples.append(sample)

    logger.info(f"  NOAA e4: {len(samples)} samples")
    return samples


def inventory_milco(output_dir: str) -> List[SampleInfo]:
    """Inventory the MILCO-NOMBO dataset."""
    logger.info("Inventorying MILCO-NOMBO dataset...")
    samples = []
    base = "datasets/milco-nombo/extracted"
    BATCH_SIZE = 50  # sub-group size for balanced splitting

    for year_dir in ["2010", "2015", "2017", "2018", "2021"]:
        year_path = os.path.join(base, year_dir, year_dir)
        if not os.path.isdir(year_path):
            # Try one level up
            year_path = os.path.join(base, year_dir)
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
                image_path=img_path,
                label_path=lbl_path,
                source="MILCO",
                source_detail=year_dir,
                original_id=stem,
                group_key=f"MILCO_{year_dir}_batch0",  # will be updated below
                split_hint="",  # no original split
            )
            year_samples.append(sample)

        # Process annotations and split year into smaller batches
        for i in range(0, len(year_samples), BATCH_SIZE):
            batch = year_samples[i:i+BATCH_SIZE]
            batch_idx = i // BATCH_SIZE
            for s in batch:
                s.group_key = f"MILCO_{year_dir}_batch{batch_idx}"
                boxes = read_yolo_label(s.label_path)
                s.label_boxes = [(MILCO_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]
                # Determine primary class from annotations
                if boxes:
                    mapped_classes = [MILCO_MAPPING.get(c, -1) for c, _, _, _, _ in boxes]
                    valid = [c for c in mapped_classes if c >= 0]
                    if valid:
                        s.master_class_id = Counter(valid).most_common(1)[0][0]
                    else:
                        s.master_class_id = 0  # fallback
                else:
                    s.master_class_id = 0  # empty annotation -> unknown_debris
            samples.extend(batch)

    logger.info(f"  MILCO-NOMBO: {len(samples)} samples")
    return samples


def inventory_kaggle(output_dir: str) -> List[SampleInfo]:
    """Inventory the Kaggle SSS dataset."""
    logger.info("Inventorying Kaggle SSS dataset...")
    samples = []
    base = "datasets/kaggle_sss"

    for split in ["train", "valid", "text"]:
        img_dir = os.path.join(base, split, "images")
        lbl_dir = os.path.join(base, split, "labels") if split != "text" else None

        if not os.path.isdir(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""

            sample = SampleInfo(
                image_path=img_path,
                label_path=lbl_path,
                source="KAGGLE",
                source_detail=f"SSS/{split}",
                original_id=stem,
                group_key=f"KAGGLE_{stem}",  # individual images
                split_hint=split,
            )

            boxes = read_yolo_label(lbl_path) if lbl_path else []
            sample.label_boxes = [(KAGGLE_MAPPING.get(c, -1), x, y, w, h) for c, x, y, w, h in boxes]

            # Determine primary class
            if boxes:
                mapped = [KAGGLE_MAPPING.get(c, -1) for c, _, _, _, _ in boxes]
                valid = [c for c in mapped if c >= 0]
                if valid:
                    sample.master_class_id = Counter(valid).most_common(1)[0][0]
                else:
                    sample.master_class_id = 0
            else:
                sample.master_class_id = 0

            samples.append(sample)

    logger.info(f"  Kaggle SSS: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def check_sample_quality(sample: SampleInfo) -> List[str]:
    """Run quality checks on a sample. Returns list of issues."""
    issues = []

    # Check image
    if not os.path.exists(sample.image_path):
        issues.append("missing_image")
        sample.is_corrupt = True
        return issues

    if os.path.getsize(sample.image_path) == 0:
        issues.append("zero_byte_image")
        sample.is_corrupt = True
        return issues

    info = get_image_info(sample.image_path)
    if info is None:
        issues.append("unreadable_image")
        sample.is_corrupt = True
        return issues

    sample.width = info["width"]
    sample.height = info["height"]
    sample.channels = info["channels"]

    if sample.width < 10 or sample.height < 10:
        issues.append("too_small_image")
        sample.is_corrupt = True

    # Check label
    if not os.path.exists(sample.label_path):
        issues.append("missing_label")
        return issues

    if os.path.getsize(sample.label_path) == 0:
        # Empty label = background image (valid for YOLO)
        return issues

    # Validate boxes
    for cls_id, xc, yc, w, h in sample.label_boxes:
        if cls_id < 0 or cls_id >= len(MASTER_CLASSES):
            issues.append(f"invalid_class_{cls_id}")
        if w <= 0 or h <= 0:
            issues.append("zero_area_box")
        if xc < 0 or xc > 1 or yc < 0 or yc > 1:
            issues.append("out_of_bounds_box")
        if w > 1 or h > 1:
            issues.append("oversized_box")

    # Check for empty annotation file but mapped boxes exist
    if os.path.getsize(sample.label_path) > 0 and not sample.label_boxes:
        # File has content but no valid mapped boxes
        pass  # This is OK for MILCO with unmapped classes

    return issues


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def detect_duplicates(samples: List[SampleInfo]) -> int:
    """Detect exact and near-duplicate images. Returns count of duplicates found."""
    logger.info("Detecting duplicates...")
    hash_map: Dict[str, List[SampleInfo]] = defaultdict(list)
    phash_map: Dict[str, List[SampleInfo]] = defaultdict(list)
    dup_count = 0

    for s in samples:
        if s.is_corrupt:
            continue
        s.sha256 = sha256_file(s.image_path)
        s.phash = simple_phash(s.image_path)
        hash_map[s.sha256].append(s)
        if s.phash:
            phash_map[s.phash].append(s)

    # Exact duplicates
    for h, group in hash_map.items():
        if len(group) > 1:
            # Keep the first, mark rest as duplicates
            for s in group[1:]:
                s.is_duplicate = True
                s.duplicate_of = group[0].image_path
                dup_count += 1

    # Near-duplicates (phash distance <= 3)
    processed = set()
    for ph, group in phash_map.items():
        if len(group) <= 1:
            continue
        for i, s1 in enumerate(group):
            if s1.is_duplicate or id(s1) in processed:
                continue
            for s2 in group[i+1:]:
                if s2.is_duplicate or id(s2) in processed:
                    continue
                if hamming_distance(s1.phash, s2.phash) <= 3:
                    s2.is_duplicate = True
                    s2.duplicate_of = s1.image_path
                    dup_count += 1
                    processed.add(id(s2))

    logger.info(f"  Found {dup_count} duplicates")
    return dup_count


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def group_aware_split(samples: List[SampleInfo], seed: int = 42) -> None:
    """Perform group-aware train/val/test splitting."""
    logger.info("Performing group-aware splitting...")
    rng = random.Random(seed)

    # Filter out duplicates and corrupt
    usable = [s for s in samples if not s.is_duplicate and not s.is_corrupt]

    # Group by group_key
    groups: Dict[str, List[SampleInfo]] = defaultdict(list)
    for s in usable:
        groups[s.group_key].append(s)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n_total = sum(len(groups[k]) for k in group_keys)
    n_train_target = int(n_total * TRAIN_RATIO)
    n_val_target = int(n_total * VAL_RATIO)

    train_groups, val_groups, test_groups = [], [], []
    train_count, val_count, test_count = 0, 0, 0

    # Round-robin assignment with ratio-based balancing
    # Build a pattern based on target ratios: 70/15/15 -> 14 train, 3 val, 3 test per 20
    n_test_target = n_total - n_train_target - n_val_target
    total_parts = int(round(n_total / 50))  # granularity
    n_train_parts = max(1, round(total_parts * TRAIN_RATIO))
    n_val_parts = max(1, round(total_parts * VAL_RATIO))
    n_test_parts = max(1, total_parts - n_train_parts - n_val_parts)

    pattern = ["train"] * n_train_parts + ["val"] * n_val_parts + ["test"] * n_test_parts
    rng.shuffle(pattern)
    pattern_idx = 0

    for gk in group_keys:
        gsize = len(groups[gk])
        split = pattern[pattern_idx % len(pattern)]
        pattern_idx += 1

        if split == "train":
            train_groups.append(gk)
            train_count += gsize
        elif split == "val":
            val_groups.append(gk)
            val_count += gsize
        else:
            test_groups.append(gk)
            test_count += gsize

    # Assign splits
    for gk in train_groups:
        for s in groups[gk]:
            s.final_split = "train"
    for gk in val_groups:
        for s in groups[gk]:
            s.final_split = "val"
    for gk in test_groups:
        for s in groups[gk]:
            s.final_split = "test"

    logger.info(f"  Split: train={train_count}, val={val_count}, test={test_count}")
    logger.info(f"  Groups: train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}")


# ---------------------------------------------------------------------------
# Generate filenames and write dataset
# ---------------------------------------------------------------------------
def generate_filenames(samples: List[SampleInfo]) -> None:
    """Generate unique filenames for all samples."""
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


def write_dataset(samples: List[SampleInfo], output_dir: str) -> None:
    """Write the final dataset to disk."""
    logger.info("Writing dataset to disk...")

    # Create directories
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

        # Copy and convert image
        dst_img = os.path.join(output_dir, "images", split, s.new_filename)
        try:
            img = Image.open(s.image_path)
            # Convert to grayscale if RGB (for sonar data consistency)
            if img.mode != "L":
                img = img.convert("L")
            img.save(dst_img, "PNG")
            img.close()
        except Exception as e:
            logger.warning(f"Failed to convert {s.image_path}: {e}")
            s.issues.append("conversion_failed")
            continue

        # Write label
        dst_lbl = os.path.join(output_dir, "labels", split, s.new_label_filename)
        with open(dst_lbl, "w") as f:
            for cls_id, xc, yc, w, h in s.label_boxes:
                if 0 <= cls_id < len(MASTER_CLASSES):
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # Metadata row
        metadata_rows.append({
            "image_id": s.new_filename.replace(".png", ""),
            "image_path": f"images/{split}/{s.new_filename}",
            "label_path": f"labels/{split}/{s.new_label_filename}",
            "source_dataset": s.source,
            "source_detail": s.source_detail,
            "original_id": s.original_id,
            "original_path": s.image_path,
            "width": s.width,
            "height": s.height,
            "channels": s.channels,
            "object_count": len(s.label_boxes),
            "master_class": MASTER_CLASSES.get(s.master_class_id, "unknown"),
            "split": split,
            "group_key": s.group_key,
            "sha256": s.sha256,
        })
        written += 1

    # Write metadata.csv
    if metadata_rows:
        meta_path = os.path.join(output_dir, "metadata.csv")
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

    # Write manifest files
    for split in ["train", "val", "test"]:
        manifest_path = os.path.join(output_dir, "manifests", f"{split}.txt")
        split_samples = [s for s in samples if s.final_split == split and not s.is_duplicate and not s.is_corrupt]
        with open(manifest_path, "w") as f:
            for s in split_samples:
                f.write(f"images/{split}/{s.new_filename}\n")

    logger.info(f"  Written {written} samples")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(samples: List[SampleInfo], output_dir: str, dup_count: int) -> None:
    """Generate comprehensive dataset reports."""
    logger.info("Generating reports...")

    usable = [s for s in samples if not s.is_duplicate and not s.is_corrupt]
    report_dir = os.path.join(os.path.dirname(output_dir), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Statistics
    stats = {
        "total_samples": len(samples),
        "usable_samples": len(usable),
        "duplicate_count": dup_count,
        "corrupt_count": sum(1 for s in samples if s.is_corrupt),
        "by_source": Counter(s.source for s in usable),
        "by_split": Counter(s.final_split for s in usable),
        "by_class": Counter(MASTER_CLASSES.get(s.master_class_id, "unknown") for s in usable),
        "by_source_split": Counter((s.source, s.final_split) for s in usable),
        "by_class_split": Counter((MASTER_CLASSES.get(s.master_class_id, "unknown"), s.final_split) for s in usable),
    }

    # Convert tuple keys to strings for JSON serialization
    json_stats = {**stats}
    json_stats["by_source_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_source_split"].items()}
    json_stats["by_class_split"] = {f"{k[0]}_{k[1]}": v for k, v in stats["by_class_split"].items()}

    # Write JSON report
    json_path = os.path.join(report_dir, "dataset_audit_multisource_v1.json")
    with open(json_path, "w") as f:
        json.dump(json_stats, f, indent=2, default=str)

    # Write markdown report
    md_path = os.path.join(report_dir, "sonarvision_multisource_v1_report.md")
    with open(md_path, "w") as f:
        f.write("# SonarVision Multi-Source Dataset Report\n\n")
        f.write("## Overall\n\n")
        f.write(f"- Total samples: {stats['total_samples']}\n")
        f.write(f"- Usable samples: {stats['usable_samples']}\n")
        f.write(f"- Duplicates removed: {stats['duplicate_count']}\n")
        f.write(f"- Corrupt samples: {stats['corrupt_count']}\n\n")

        f.write("## By Source\n\n")
        for src, cnt in sorted(stats["by_source"].items()):
            f.write(f"- {src}: {cnt}\n")

        f.write("\n## By Split\n\n")
        for split, cnt in sorted(stats["by_split"].items()):
            f.write(f"- {split}: {cnt}\n")

        f.write("\n## By Class\n\n")
        for cls, cnt in sorted(stats["by_class"].items()):
            f.write(f"- {cls}: {cnt}\n")

        f.write("\n## Source × Split\n\n")
        f.write("| Source | Train | Val | Test |\n")
        f.write("|--------|-------|-----|------|\n")
        for src in ["NOAA", "MILCO", "KAGGLE"]:
            tr = stats["by_source_split"].get((src, "train"), 0)
            vl = stats["by_source_split"].get((src, "val"), 0)
            te = stats["by_source_split"].get((src, "test"), 0)
            f.write(f"| {src} | {tr} | {vl} | {te} |\n")

        f.write("\n## Class × Split\n\n")
        f.write("| Class | Train | Val | Test |\n")
        f.write("|-------|-------|-----|------|\n")
        for cls_name in MASTER_CLASSES.values():
            tr = stats["by_class_split"].get((cls_name, "train"), 0)
            vl = stats["by_class_split"].get((cls_name, "val"), 0)
            te = stats["by_class_split"].get((cls_name, "test"), 0)
            f.write(f"| {cls_name} | {tr} | {vl} | {te} |\n")

    # Write quality issues
    issues_path = os.path.join(report_dir, "data_quality_issues_v1.csv")
    with open(issues_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "dataset", "issue_type", "severity", "action_taken"])
        for s in samples:
            for issue in s.issues:
                severity = "high" if "corrupt" in issue or "missing" in issue else "medium"
                writer.writerow([s.image_path, s.source, issue, severity, "excluded" if "corrupt" in issue else "logged"])

    logger.info(f"  Reports written to {report_dir}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(output_dir: str) -> bool:
    """Run final validation checks."""
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

        # Check all images have labels
        for img in images:
            lbl = img.replace(".png", ".txt")
            if lbl not in labels:
                logger.error(f"Missing label for {split}/{img}")
                all_ok = False

        # Check all labels have images
        for lbl in labels:
            img = lbl.replace(".txt", ".png")
            if img not in images:
                logger.error(f"Missing image for {split}/{lbl}")
                all_ok = False

        # Check for duplicates across splits
        logger.info(f"  {split}: {len(images)} images, {len(labels)} labels")

    # Check no cross-split duplicates
    all_hashes = defaultdict(list)
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            fpath = os.path.join(img_dir, fname)
            h = sha256_file(fpath)
            all_hashes[h].append((split, fname))

    for h, files in all_hashes.items():
        splits = set(s for s, _ in files)
        if len(splits) > 1:
            logger.error(f"DUPLICATE ACROSS SPLITS: {files}")
            all_ok = False

    if all_ok:
        logger.info("  ✅ All validation checks passed!")
    else:
        logger.error("  ❌ Validation FAILED - see errors above")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build SonarVision multi-source dataset")
    parser.add_argument("--output", default="datasets/sonarvision_multisource_v1",
                        help="Output directory for the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Only inventory and report, don't write dataset")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {args.seed}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Inventory all sources
    logger.info("=" * 60)
    logger.info("STEP 1: Inventorying data sources")
    logger.info("=" * 60)
    all_samples = []
    all_samples.extend(inventory_noaa(output_dir))
    all_samples.extend(inventory_milco(output_dir))
    all_samples.extend(inventory_kaggle(output_dir))
    logger.info(f"Total samples inventoried: {len(all_samples)}")

    # Step 2: Quality checks
    logger.info("=" * 60)
    logger.info("STEP 2: Running quality checks")
    logger.info("=" * 60)
    issue_count = 0
    for s in all_samples:
        issues = check_sample_quality(s)
        s.issues.extend(issues)
        issue_count += len(issues)
    logger.info(f"Total issues found: {issue_count}")

    # Step 3: Duplicate detection
    logger.info("=" * 60)
    logger.info("STEP 3: Detecting duplicates")
    logger.info("=" * 60)
    dup_count = detect_duplicates(all_samples)

    # Step 4: Generate filenames
    logger.info("=" * 60)
    logger.info("STEP 4: Generating filenames")
    logger.info("=" * 60)
    generate_filenames(all_samples)

    # Step 5: Split
    logger.info("=" * 60)
    logger.info("STEP 5: Group-aware splitting")
    logger.info("=" * 60)
    group_aware_split(all_samples, seed=args.seed)

    # Step 6: Write dataset
    if not args.dry_run:
        logger.info("=" * 60)
        logger.info("STEP 6: Writing dataset")
        logger.info("=" * 60)
        write_dataset(all_samples, output_dir)

        # Step 7: Write dataset.yaml
        logger.info("=" * 60)
        logger.info("STEP 7: Writing dataset.yaml")
        logger.info("=" * 60)
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"# SonarVision Multi-Source Dataset v1\n")
            f.write(f"path: {output_dir}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            f.write(f"nc: {len(MASTER_CLASSES)}\n")
            f.write(f"names:\n")
            for cid, cname in MASTER_CLASSES.items():
                f.write(f"  {cid}: {cname}\n")

        # Step 8: Write class_mapping.yaml
        mapping_path = os.path.join(output_dir, "class_mapping.yaml")
        with open(mapping_path, "w") as f:
            f.write("# Source-to-Master Class Mapping\n\n")
            f.write("master_classes:\n")
            for cid, cname in MASTER_CLASSES.items():
                f.write(f"  {cid}: {cname}\n\n")
            f.write("NOAA_mapping:\n")
            f.write("  0: 0  # marine_debris -> unknown_debris\n\n")
            f.write("MILCO_mapping:\n")
            f.write("  0: 3  # MILCO -> mine\n")
            f.write("  1: 0  # NOMBO -> unknown_debris\n\n")
            f.write("KAGGLE_mapping:\n")
            f.write("  0: 1  # airplane -> airplane\n")
            f.write("  1: 2  # drowning_victim -> drowning_victim\n")
            f.write("  2: 3  # mine -> mine\n")
            f.write("  3: 4  # wreck -> wreck\n")

    # Step 9: Reports
    logger.info("=" * 60)
    logger.info("STEP 9: Generating reports")
    logger.info("=" * 60)
    generate_report(all_samples, output_dir, dup_count)

    # Step 10: Validation
    if not args.dry_run:
        logger.info("=" * 60)
        logger.info("STEP 10: Final validation")
        logger.info("=" * 60)
        valid = validate_dataset(output_dir)
    else:
        valid = True

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET BUILD SUMMARY")
    logger.info("=" * 60)

    usable = [s for s in all_samples if not s.is_duplicate and not s.is_corrupt]
    logger.info(f"NOAA images:       {sum(1 for s in usable if s.source == 'NOAA')}")
    logger.info(f"MILCO images:      {sum(1 for s in usable if s.source == 'MILCO')}")
    logger.info(f"Kaggle images:     {sum(1 for s in usable if s.source == 'KAGGLE')}")
    logger.info(f"Total usable:      {len(usable)}")
    logger.info(f"Duplicates removed: {dup_count}")
    logger.info(f"Corrupt removed:   {sum(1 for s in all_samples if s.is_corrupt)}")
    logger.info("")

    for split in ["train", "val", "test"]:
        n = sum(1 for s in usable if s.final_split == split)
        logger.info(f"{split.upper():5s}: {n} images")
    logger.info("")

    logger.info("Master classes:")
    for cid, cname in MASTER_CLASSES.items():
        cnt = sum(1 for s in usable if s.master_class_id == cid)
        logger.info(f"  {cid}: {cname} ({cnt} objects)")
    logger.info("")

    if valid:
        logger.info("DATASET BUILD STATUS: PASS")
    else:
        logger.info("DATASET BUILD STATUS: FAIL")
    logger.info("MODEL TRAINING STARTED: NO")


if __name__ == "__main__":
    main()
