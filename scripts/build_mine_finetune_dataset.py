#!/usr/bin/env python3
"""
Build MILCO Mine-Only Finetune Dataset
======================================

Extracts only MILCO images with mine annotations (class 0 = MILCO = mine).
Single class: mine (0)
Purpose: Fine-tune after multi-source Stage 2 to improve mine detection.

Previous training was binary marine_debris.
Multi-source training has 5 classes but mine scores poorly.
This dataset focuses solely on MILCO military mine-like contacts.
"""

import os
import shutil
import random
from pathlib import Path

SEED = 42
MILCO_BASE = 'datasets/milco-nombo/extracted'
OUTPUT = 'datasets/milco_mine_finetune'
YEARS = ['2010', '2015', '2017', '2018', '2021']

def parse_yolo_boxes(lbl_path):
    """Parse YOLO label, return only class 0 (MILCO=mine) boxes."""
    boxes = []
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            i = 0
            while i + 4 < len(parts):
                try:
                    cls = int(parts[i])
                    xc, yc, w, h = float(parts[i+1]), float(parts[i+2]), float(parts[i+3]), float(parts[i+4])
                    if cls == 0 and 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
                        boxes.append((0, xc, yc, w, h))  # class 0 = mine
                    i += 5
                except (ValueError, IndexError):
                    break
    return boxes

def main():
    rng = random.Random(SEED)

    # Collect all MILCO mine images
    all_samples = []
    for year in YEARS:
        year_path = os.path.join(MILCO_BASE, year, year)
        if not os.path.isdir(year_path):
            continue
        for fname in sorted(os.listdir(year_path)):
            if not fname.endswith('.jpg'):
                continue
            lbl_path = os.path.join(year_path, fname.replace('.jpg', '.txt'))
            boxes = parse_yolo_boxes(lbl_path)
            if boxes:  # Only images with mine annotations
                all_samples.append({
                    'img': os.path.join(year_path, fname),
                    'lbl': lbl_path,
                    'year': year,
                    'stem': fname.replace('.jpg', ''),
                    'n_boxes': len(boxes),
                    'boxes': boxes,
                })

    print(f'Total MILCO mine images: {len(all_samples)}')

    # Split: 70/15/15
    rng.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        'train': all_samples[:n_train],
        'val': all_samples[n_train:n_train+n_val],
        'test': all_samples[n_train+n_val:],
    }

    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT, 'labels', split), exist_ok=True)

    # Copy and convert
    total_boxes = 0
    for split, samples in splits.items():
        for s in samples:
            # Convert JPG to PNG (grayscale)
            from PIL import Image
            img = Image.open(s['img']).convert('L')
            dst_img = os.path.join(OUTPUT, 'images', split, f"{s['stem']}.png")
            img.save(dst_img, 'PNG')
            img.close()

            # Write label (class 0 = mine)
            dst_lbl = os.path.join(OUTPUT, 'labels', split, f"{s['stem']}.txt")
            with open(dst_lbl, 'w') as f:
                for cls, xc, yc, w, h in s['boxes']:
                    f.write(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
            total_boxes += len(s['boxes'])

        print(f'  {split}: {len(samples)} images')

    # Write dataset.yaml
    yaml_content = f"""# MILCO Mine-Only Finetune Dataset
# Single class: mine (0)
# Purpose: Fine-tune after multi-source Stage 2 to improve mine detection
# Source: MILCO-NOMBO (military sonar mine-like contacts)

path: /home/ashish/sonar-vision/{OUTPUT}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: mine
"""
    with open(os.path.join(OUTPUT, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f'\nTotal boxes: {total_boxes}')
    print(f'Dataset: {OUTPUT}/dataset.yaml')
    print(f'\nTo fine-tune after Stage 2:')
    print(f'  Load best.pt from Stage 2')
    print(f'  Train 20-30 epochs on this 1-class mine dataset')
    print(f'  This specializes the model on mine detection')

if __name__ == '__main__':
    main()
