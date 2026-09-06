#!/usr/bin/env python3
"""
SonarVision — Comprehensive Audit of Multi-Source Dataset
==========================================================

Validates the generated dataset for YOLO trainability:
1. Image format consistency
2. Label validity
3. Image-label pairing
4. Box coordinate sanity
5. Class distribution
6. File integrity
7. Cross-split leakage
"""

import os
import sys
import csv
import hashlib
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow required: pip install Pillow")
    sys.exit(1)

MASTER_CLASSES = {
    0: "unknown_debris",
    1: "airplane",
    2: "drowning_victim",
    3: "mine",
    4: "wreck",
}


def audit_dataset(dataset_dir: str):
    """Run full audit on the dataset."""
    print("=" * 60)
    print("SONARVISION MULTI-SOURCE DATASET AUDIT")
    print("=" * 60)
    
    issues = []
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "total_objects": 0,
        "by_split": {},
        "by_source": defaultdict(lambda: {"images": 0, "objects": 0, "classes": Counter()}),
        "by_class": Counter(),
        "image_formats": Counter(),
        "image_channels": Counter(),
        "image_dtypes": Counter(),
        "image_sizes": [],
        "issues": [],
    }
    
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        
        if not os.path.isdir(img_dir):
            issues.append(f"MISSING: {img_dir}")
            continue
        
        print(f"\n--- Auditing {split} split ---")
        
        split_stats = {
            "images": 0,
            "labels": 0,
            "objects": 0,
            "classes": Counter(),
            "sources": defaultdict(int),
            "corrupt_images": 0,
            "missing_labels": 0,
            "empty_labels": 0,
            "invalid_labels": 0,
            "invalid_boxes": 0,
        }
        
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            
            stats["total_images"] += 1
            split_stats["images"] += 1
            
            # Determine source
            source = "UNKNOWN"
            for s in ["NOAA", "MILCO", "KAGGLE"]:
                if stem.startswith(s + "_"):
                    source = s
                    break
            split_stats["sources"][source] += 1
            stats["by_source"][source]["images"] += 1
            
            # Validate image
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path)  # Re-open after verify
                
                stats["image_formats"][img.format or "UNKNOWN"] += 1
                stats["image_channels"][len(img.getbands())] += 1
                
                # Check for grayscale
                if img.mode not in ("L", "LA", "I", "I;16", "F"):
                    # RGB or other - record
                    stats["image_dtypes"][img.mode] += 1
                
                stats["image_sizes"].append((img.width, img.height))
                
                img.close()
            except Exception as e:
                split_stats["corrupt_images"] += 1
                issues.append(f"CORRUPT_IMAGE: {split}/{fname}: {e}")
            
            # Validate label
            if not os.path.exists(lbl_path):
                split_stats["missing_labels"] += 1
                issues.append(f"MISSING_LABEL: {split}/{fname}")
                continue
            
            if os.path.getsize(lbl_path) == 0:
                split_stats["empty_labels"] += 1
                stats["total_labels"] += 1
                continue
            
            split_stats["labels"] += 1
            stats["total_labels"] += 1
            
            with open(lbl_path, "r") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        split_stats["invalid_labels"] += 1
                        issues.append(f"INVALID_LABEL_LINE: {split}/{stem}.txt:{line_no}: {len(parts)} values")
                        continue
                    
                    # Parse as groups of 5
                    i = 0
                    while i + 4 < len(parts):
                        try:
                            cls_id = int(parts[i])
                            xc = float(parts[i+1])
                            yc = float(parts[i+2])
                            bw = float(parts[i+3])
                            bh = float(parts[i+4])
                            
                            # Validate class
                            if cls_id not in MASTER_CLASSES:
                                split_stats["invalid_boxes"] += 1
                                issues.append(f"INVALID_CLASS: {split}/{stem}.txt:{line_no}: class {cls_id}")
                                i += 5
                                continue
                            
                            # Validate coordinates
                            problems = []
                            if not (0 <= xc <= 1):
                                problems.append(f"xc={xc:.4f}")
                            if not (0 <= yc <= 1):
                                problems.append(f"yc={yc:.4f}")
                            if not (0 < bw <= 1):
                                problems.append(f"bw={bw:.4f}")
                            if not (0 < bh <= 1):
                                problems.append(f"bh={bh:.4f}")
                            
                            if problems:
                                split_stats["invalid_boxes"] += 1
                                issues.append(f"INVALID_BOX: {split}/{stem}.txt:{line_no}: {', '.join(problems)}")
                            else:
                                stats["total_objects"] += 1
                                split_stats["objects"] += 1
                                stats["by_class"][MASTER_CLASSES[cls_id]] += 1
                                stats["by_source"][source]["objects"] += 1
                                stats["by_source"][source]["classes"][MASTER_CLASSES[cls_id]] += 1
                                split_stats["classes"][MASTER_CLASSES[cls_id]] += 1
                            
                            i += 5
                        except (ValueError, IndexError):
                            split_stats["invalid_labels"] += 1
                            issues.append(f"PARSE_ERROR: {split}/{stem}.txt:{line_no}: could not parse at offset {i}")
                            break
        
        stats["by_split"][split] = split_stats
        
        print(f"  Images: {split_stats['images']}")
        print(f"  Labels found: {split_stats['labels']}")
        print(f"  Empty labels: {split_stats['empty_labels']}")
        print(f"  Objects: {split_stats['objects']}")
        print(f"  Sources: {dict(split_stats['sources'])}")
        print(f"  Corrupt images: {split_stats['corrupt_images']}")
        print(f"  Missing labels: {split_stats['missing_labels']}")
        print(f"  Invalid label lines: {split_stats['invalid_labels']}")
        print(f"  Invalid boxes: {split_stats['invalid_boxes']}")
        print(f"  Classes: {dict(split_stats['classes'])}")
    
    # Image size analysis
    if stats["image_sizes"]:
        widths = [s[0] for s in stats["image_sizes"]]
        heights = [s[1] for s in stats["image_sizes"]]
        print(f"\n--- Image Size Analysis ---")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.0f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.0f}")
        
        # Check for very small images
        small = [(w, h) for w, h in stats["image_sizes"] if w < 32 or h < 32]
        if small:
            print(f"  WARNING: {len(small)} images smaller than 32x32")
            issues.append(f"SMALL_IMAGES: {len(small)} images smaller than 32x32")
    
    # Cross-split leakage check
    print(f"\n--- Cross-Split Leakage Check ---")
    all_hashes = defaultdict(list)
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, fname)
            try:
                with open(img_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                all_hashes[h].append((split, fname))
            except Exception:
                pass
    
    leakage = {h: files for h, files in all_hashes.items() if len(set(s for s, _ in files)) > 1}
    if leakage:
        print(f"  FAILED: {len(leakage)} duplicate images across splits!")
        for h, files in list(leakage.items())[:5]:
            print(f"    {files}")
        issues.append(f"LEAKAGE: {len(leakage)} images appear in multiple splits")
    else:
        print(f"  PASSED: No cross-split duplicates")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total labels: {stats['total_labels']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"\nBy source:")
    for src in ["NOAA", "MILCO", "KAGGLE"]:
        s = stats["by_source"][src]
        print(f"  {src}: {s['images']} images, {s['objects']} objects")
        if s["classes"]:
            print(f"    Classes: {dict(s['classes'])}")
    print(f"\nBy class:")
    for cls_name, count in sorted(stats["by_class"].items()):
        print(f"  {cls_name}: {count}")
    print(f"\nImage formats: {dict(stats['image_formats'])}")
    print(f"Image channels: {dict(stats['image_channels'])}")
    print(f"Image modes: {dict(stats['image_dtypes'])}")
    print(f"\nIssues found: {len(issues)}")
    for issue in issues[:20]:
        print(f"  - {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more")
    
    # Write issues to CSV
    issues_path = os.path.join(os.path.dirname(dataset_dir), "..", "reports", "audit_issues_v1.csv")
    os.makedirs(os.path.dirname(issues_path), exist_ok=True)
    with open(issues_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["issue_type", "description"])
        for issue in issues:
            parts = issue.split(": ", 1)
            writer.writerow([parts[0], parts[1] if len(parts) > 1 else ""])
    print(f"\nIssues written to: {issues_path}")
    
    return issues, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/sonarvision_multisource_v1")
    args = parser.parse_args()
    
    issues, stats = audit_dataset(args.dataset)
    
    if issues:
        print(f"\n⚠️  {len(issues)} issues found - review before training")
        sys.exit(1)
    else:
        print(f"\n✅ Dataset audit PASSED - ready for YOLO training")
        sys.exit(0)
