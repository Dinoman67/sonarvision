#!/usr/bin/env python3
"""
Build F6 Dataset — Combine All NOAA Data
=========================================

Combines all available datasets into one unified YOLO-format dataset:
- E3: 369 images (original NOAA H11833 patches)
- E4: 737 images (enhanced with more targets)
- YOLO: 282 images (separate annotation set)

Total: ~1388 unique images (0 overlap between datasets)

Output: datasets/noaa-debris/f6/
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def count_labels(label_dir):
    """Count images with labels and total boxes."""
    n_labeled = 0
    n_boxes = 0
    n_empty = 0
    
    if not label_dir.exists():
        return 0, 0, 0
    
    for f in label_dir.glob('*.txt'):
        if f.stat().st_size > 0:
            n_labeled += 1
            with open(f) as fh:
                n_boxes += len(fh.readlines())
        else:
            n_empty += 1
    
    return n_labeled, n_boxes, n_empty


def merge_labels(src_label, dst_label):
    """Merge label files (append if both exist)."""
    if not src_label.exists():
        return
    
    if dst_label.exists() and dst_label.stat().st_size > 0:
        # Both have content — merge (shouldn't happen with unique images)
        with open(src_label) as f:
            src_lines = f.readlines()
        with open(dst_label, 'a') as f:
            for line in src_lines:
                f.write(line)
    else:
        # Just copy
        shutil.copy(src_label, dst_label)


def build_f6_dataset(output_path, seed=42):
    """Build unified F6 dataset from all sources."""
    random.seed(seed)
    
    output_path = Path(output_path)
    base_path = Path('datasets/noaa-debris')
    
    print("="*70)
    print("BUILDING F6 DATASET — COMBINING ALL NOAA DATA")
    print("="*70)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images from all sources
    all_images = []  # (source, split, img_name)
    
    sources = [
        ('e3', base_path / 'e3'),
        ('e4', base_path / 'e4'),
        ('yolo', base_path / 'yolo'),
    ]
    
    for source_name, source_path in sources:
        for split in ['train', 'val', 'test']:
            img_dir = source_path / 'images' / split
            if not img_dir.exists():
                continue
            
            for img_file in img_dir.glob('*.png'):
                all_images.append((source_name, split, img_file.name))
    
    print(f"\nFound {len(all_images)} images across all sources:")
    
    # Count per source
    source_counts = defaultdict(lambda: defaultdict(int))
    for source, split, name in all_images:
        source_counts[source][split] += 1
    
    for source in sorted(source_counts.keys()):
        counts = source_counts[source]
        total = sum(counts.values())
        print(f"  {source:>8}: {total:>5} images ({', '.join(f'{s}:{c}' for s,c in sorted(counts.items()))})")
    
    # Now rebalance: collect all images, then split into train/val/test
    # First, gather all unique images with their labels
    all_unique = {}  # img_name -> (source, split, img_path, lbl_path)
    
    for source, split, img_name in all_images:
        source_path = base_path / source
        img_path = source_path / 'images' / split / img_name
        lbl_path = source_path / 'labels' / split / img_name.replace('.png', '.txt')
        
        if img_name not in all_unique:
            all_unique[img_name] = {
                'source': source,
                'orig_split': split,
                'img_path': img_path,
                'lbl_path': lbl_path,
            }
    
    print(f"\nUnique images: {len(all_unique)}")
    
    # Check how many have labels
    n_with_labels = sum(1 for v in all_unique.values() if v['lbl_path'].exists() and v['lbl_path'].stat().st_size > 0)
    n_without = sum(1 for v in all_unique.values() if not v['lbl_path'].exists() or v['lbl_path'].stat().st_size == 0)
    print(f"  With labels (debris): {n_with_labels}")
    print(f"  Without labels (BG): {n_without}")
    
    # Split: 80% train, 10% val, 10% test
    img_names = list(all_unique.keys())
    random.shuffle(img_names)
    
    n_train = int(len(img_names) * 0.8)
    n_val = int(len(img_names) * 0.1)
    
    splits = {}
    for i, name in enumerate(img_names):
        if i < n_train:
            splits[name] = 'train'
        elif i < n_train + n_val:
            splits[name] = 'val'
        else:
            splits[name] = 'test'
    
    # Copy files
    stats = {'train': {'debris': 0, 'bg': 0}, 'val': {'debris': 0, 'bg': 0}, 'test': {'debris': 0, 'bg': 0}}
    
    for img_name, split in splits.items():
        info = all_unique[img_name]
        
        # Copy image
        dst_img = output_path / 'images' / split / img_name
        shutil.copy(info['img_path'], dst_img)
        
        # Copy/create label
        dst_lbl = output_path / 'labels' / split / img_name.replace('.png', '.txt')
        if info['lbl_path'].exists() and info['lbl_path'].stat().st_size > 0:
            shutil.copy(info['lbl_path'], dst_lbl)
            stats[split]['debris'] += 1
        else:
            dst_lbl.touch()
            stats[split]['bg'] += 1
    
    # Write data.yaml
    data_yaml = f"""# F6 Dataset — Combined NOAA H11833 Data
# Sources: E3 (369) + E4 (737) + YOLO (282) = 1388 unique images
# Generated by scripts/build_f6_dataset.py
path: {output_path}
train: images/train
val: images/val
test: images/test

names:
  0: marine_debris
"""
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(data_yaml)
    
    # Print summary
    print("\n" + "="*70)
    print("F6 DATASET SUMMARY")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        total = stats[split]['debris'] + stats[split]['bg']
        print(f"  {split:>5}: {total:>5} images ({stats[split]['debris']} debris + {stats[split]['bg']} bg)")
    
    total_imgs = sum(s['debris'] + s['bg'] for s in stats.values())
    total_debris = sum(s['debris'] for s in stats.values())
    total_bg = sum(s['bg'] for s in stats.values())
    
    print(f"\n  {'TOTAL':>5}: {total_imgs:>5} images ({total_debris} debris + {total_bg} bg)")
    print(f"  Ratio: 1 debris : {total_bg/max(total_debris,1):.1f} bg")
    print(f"\n  This is {total_imgs/953:.1f}x more data than E5!")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build F6 dataset from all NOAA data')
    parser.add_argument('--output', default='datasets/noaa-debris/f6',
                       help='Output path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    build_f6_dataset(args.output, args.seed)
