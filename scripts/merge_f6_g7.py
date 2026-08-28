#!/usr/bin/env python3
"""
Merge F6 dataset (debris + BG) with G7 hard negatives for YOLO training.

Usage:
    python scripts/merge_f6_g7.py \
        --f6 datasets/noaa-debris/f6 \
        --g7 datasets/noaa-debris/g7 \
        --output datasets/noaa-debris/h8 \
        --max-bg 2000

Output structure:
    h8/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def merge_datasets(f6_root: str, g7_root: str, output_root: str,
                   max_bg: int = 2000, seed: int = 42):
    """Merge F6 (debris+bg) with G7 (hard negatives)."""
    random.seed(seed)
    f6 = Path(f6_root)
    g7 = Path(g7_root)
    out = Path(output_root)

    # ── Count what we have ──
    f6_debris_train = []
    f6_bg_train = []
    f6_debris_val = []
    f6_bg_val = []

    # F6 train
    f6_train_labels = f6 / "labels" / "train"
    if f6_train_labels.exists():
        for lbl in f6_train_labels.glob("*.txt"):
            with open(lbl) as f:
                content = f.read().strip()
            if content:  # Has debris annotations
                f6_debris_train.append(lbl.stem)
            else:
                f6_bg_train.append(lbl.stem)

    # F6 val
    f6_val_labels = f6 / "labels" / "val"
    if f6_val_labels.exists():
        for lbl in f6_val_labels.glob("*.txt"):
            with open(lbl) as f:
                content = f.read().strip()
            if content:
                f6_debris_val.append(lbl.stem)
            else:
                f6_bg_val.append(lbl.stem)

    # F6 also has images without labels (pure background)
    f6_train_images = f6 / "images" / "train"
    if f6_train_images.exists():
        labeled_stems = set(f6_debris_train + f6_bg_train)
        for img in f6_train_images.glob("*.png"):
            if img.stem not in labeled_stems:
                f6_bg_train.append(img.stem)
    f6_val_images = f6 / "images" / "val"
    if f6_val_images.exists():
        labeled_stems = set(f6_debris_val + f6_bg_val)
        for img in f6_val_images.glob("*.png"):
            if img.stem not in labeled_stems:
                f6_bg_val.append(img.stem)

    # G7 backgrounds
    g7_train_labels = g7 / "labels" / "train"
    g7_bg_train = []
    if g7_train_labels.exists():
        for lbl in g7_train_labels.glob("*.txt"):
            g7_bg_train.append(lbl.stem)
    else:
        g7_train_images = g7 / "images" / "train"
        if g7_train_images.exists():
            g7_bg_train = [img.stem for img in g7_train_images.glob("*.png")]

    g7_val_labels = g7 / "labels" / "val"
    g7_bg_val = []
    if g7_val_labels.exists():
        for lbl in g7_val_labels.glob("*.txt"):
            g7_bg_val.append(lbl.stem)
    else:
        g7_val_images = g7 / "images" / "val"
        if g7_val_images.exists():
            g7_bg_val = [img.stem for img in g7_val_images.glob("*.png")]

    print(f"F6: {len(f6_debris_train)} debris train, {len(f6_bg_train)} bg train")
    print(f"F6: {len(f6_debris_val)} debris val, {len(f6_bg_val)} bg val")
    print(f"G7: {len(g7_bg_train)} bg train, {len(g7_bg_val)} bg val")

    # ── Select G7 subset ──
    n_g7_train = min(max_bg, len(g7_bg_train))
    n_g7_val = min(int(max_bg * 0.15), len(g7_bg_val))
    selected_g7_train = random.sample(g7_bg_train, n_g7_train)
    selected_g7_val = random.sample(g7_bg_val, n_g7_val) if g7_bg_val else []

    print(f"\nSelected G7: {n_g7_train} train, {n_g7_val} val")

    # ── Create output directories ──
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Helper: find image for a stem ──
    def find_image(stem, sources):
        for src in sources:
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                p = src / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    # ── Copy files ──
    def copy_item(stem, split, sources_img, sources_lbl, src_root):
        # Find and copy image
        img = find_image(stem, sources_img)
        if img:
            shutil.copy2(img, out / "images" / split / img.name)

        # Find and copy label (or create empty)
        lbl = None
        for src in sources_lbl:
            p = src / f"{stem}.txt"
            if p.exists():
                lbl = p
                break

        if lbl:
            shutil.copy2(lbl, out / "labels" / split / lbl.name)
        else:
            (out / "labels" / split / f"{stem}.txt").touch()

    # Copy F6 debris
    for stem in f6_debris_train:
        copy_item(stem, "train",
                  [f6 / "images" / "train"],
                  [f6 / "labels" / "train"],
                  f6)
    for stem in f6_debris_val:
        copy_item(stem, "val",
                  [f6 / "images" / "val"],
                  [f6 / "labels" / "val"],
                  f6)

    # Copy F6 backgrounds
    for stem in f6_bg_train:
        copy_item(stem, "train",
                  [f6 / "images" / "train"],
                  [f6 / "labels" / "train"],
                  f6)
    for stem in f6_bg_val:
        copy_item(stem, "val",
                  [f6 / "images" / "val"],
                  [f6 / "labels" / "val"],
                  f6)

    # Copy G7 backgrounds
    for stem in selected_g7_train:
        copy_item(stem, "train",
                  [g7 / "images" / "train"],
                  [g7 / "labels" / "train"] if (g7 / "labels" / "train").exists() else [],
                  g7)
    for stem in selected_g7_val:
        copy_item(stem, "val",
                  [g7 / "images" / "val"],
                  [g7 / "labels" / "val"] if (g7 / "labels" / "val").exists() else [],
                  g7)

    # ── Count final stats ──
    train_imgs = list((out / "images" / "train").glob("*"))
    val_imgs = list((out / "images" / "val").glob("*"))
    train_lbls = list((out / "labels" / "train").glob("*.txt"))

    debris_count = 0
    bg_count = 0
    for lbl in train_lbls:
        with open(lbl) as f:
            content = f.read().strip()
        if content:
            debris_count += 1
        else:
            bg_count += 1

    print(f"\n{'='*50}")
    print(f"MERGED DATASET (H8) SUMMARY")
    print(f"{'='*50}")
    print(f"  Train: {len(train_imgs)} images")
    print(f"    Debris:  {debris_count}")
    print(f"    BG:      {bg_count}")
    print(f"    Ratio:   1:{bg_count/max(debris_count,1):.1f}")
    print(f"  Val:   {len(val_imgs)} images")
    print(f"{'='*50}")

    # ── Write data.yaml ──
    yaml_content = f"""path: {out}
train: images/train
val: images/val

nc: 1
names: ['marine_debris']

# H8 Dataset = F6 (debris+bg) + G7 (hard negatives)
# F6: {len(f6_debris_train)} debris + {len(f6_bg_train)} bg from E3+E4+YOLO
# G7: {n_g7_train} hard negative BG from raw TIFFs
"""
    (out / "data.yaml").write_text(yaml_content)
    print(f"\n  data.yaml: {out / 'data.yaml'}")

    # ── Write manifest ──
    manifest_lines = ["stem,source,split,has_debris\n"]
    for stem in f6_debris_train:
        manifest_lines.append(f"{stem},f6,train,1\n")
    for stem in f6_bg_train:
        manifest_lines.append(f"{stem},f6,train,0\n")
    for stem in selected_g7_train:
        manifest_lines.append(f"{stem},g7,train,0\n")
    for stem in f6_debris_val:
        manifest_lines.append(f"{stem},f6,val,1\n")
    for stem in f6_bg_val:
        manifest_lines.append(f"{stem},f6,val,0\n")
    for stem in selected_g7_val:
        manifest_lines.append(f"{stem},g7,val,0\n")
    (out / "manifest.csv").write_text("".join(manifest_lines))
    print(f"  manifest: {out / 'manifest.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge F6 + G7 datasets")
    parser.add_argument("--f6", required=True, help="Path to F6 dataset")
    parser.add_argument("--g7", required=True, help="Path to G7 dataset")
    parser.add_argument("--output", required=True, help="Output path (H8)")
    parser.add_argument("--max-bg", type=int, default=2000,
                        help="Max BG images from G7 (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merge_datasets(args.f6, args.g7, args.output, args.max_bg, args.seed)
