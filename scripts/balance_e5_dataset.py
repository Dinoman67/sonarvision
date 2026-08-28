"""
Balance E5 Dataset — Oversample Debris to Fix Class Imbalance
==============================================================

Problem: E5 has 153 debris images vs 800 background images.
This 1:5 ratio causes the model to learn "predict nothing" → high precision, terrible recall.

Solution: Oversample debris images by copying them with augmentations to create
a balanced training set (~1:1 ratio). Validation/test sets stay untouched.

Key augmentation for oversampling:
- Random brightness/contrast (simulates different sonar conditions)
- Horizontal flip (SSS waterfall images are symmetric left-right)
- Slight translation (simulates different positioning)
- No rotation (SSS images have fixed orientation)

Output: datasets/noaa-debris/e5_balanced/
"""

import os
import sys
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

try:
    from scripts.generate_e5_noisy import apply_random_sss_noise
except ImportError:
    apply_random_sss_noise = None


def augment_debris_image(img, seed=None):
    """Apply SSS-realistic augmentation to a debris image.
    
    These augmentations simulate:
    - Different sonar gain settings (brightness/contrast)
    - Different towfish positions (translation)
    - Slight speckle noise
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    img_arr = np.array(img)
    
    # Random brightness (±30%)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Random contrast (±25%)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.75, 1.25))
    
    # Random horizontal flip (SSS waterfall is symmetric)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Small random translation (±5%)
    w, h = img.size
    tx = int(w * random.uniform(-0.05, 0.05))
    ty = int(h * random.uniform(-0.05, 0.05))
    img = img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty),
                         fillcolor=0)
    
    # Light speckle noise
    if apply_random_sss_noise is not None:
        img_arr = np.array(img)
        noise = np.random.lognormal(0, 0.1, img_arr.shape)
        img_arr = np.clip(img_arr * noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
    
    return img


def balance_dataset(e5_path, output_path, target_ratio=3, seed=42):
    """Create a balanced dataset by oversampling debris images.
    
    Args:
        e5_path: Path to original E5 dataset
        output_path: Output path for balanced dataset
        target_ratio: How many copies of each debris image (3 = 3x oversampling)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    e5_path = Path(e5_path)
    output_path = Path(output_path)
    
    print(f"Balancing E5 dataset...")
    print(f"  Source: {e5_path}")
    print(f"  Output: {output_path}")
    print(f"  Target oversampling: {target_ratio}x")
    print()
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'train': {'debris_orig': 0, 'debris_aug': 0, 'bg': 0, 'total': 0},
        'val': {'n': 0},
        'test': {'n': 0},
    }
    
    for split in ['train', 'val', 'test']:
        img_dir = e5_path / 'images' / split
        lbl_dir = e5_path / 'labels' / split
        
        if not img_dir.exists():
            print(f"  Skipping {split} — not found")
            continue
        
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        for img_name in images:
            stem = img_name.replace('.png', '')
            lbl_name = f"{stem}.txt"
            lbl_path = lbl_dir / lbl_name
            
            # Copy original
            shutil.copy(img_dir / img_name, output_path / 'images' / split / img_name)
            if lbl_path.exists():
                shutil.copy(lbl_path, output_path / 'labels' / split / lbl_name)
            else:
                (output_path / 'labels' / split / lbl_name).touch()
            
            is_debris = 'BG' not in img_name
            
            if split == 'train' and is_debris:
                stats['train']['debris_orig'] += 1
                
                # Generate augmented copies
                orig = Image.open(img_dir / img_name)
                for j in range(target_ratio):
                    aug_img = augment_debris_image(orig, seed=hash(img_name) + j)
                    aug_name = f"{stem}_A{j+1:02d}.png"
                    aug_img.save(output_path / 'images' / split / aug_name)
                    
                    # Copy the same label
                    if lbl_path.exists():
                        shutil.copy(lbl_path, output_path / 'labels' / split / f"{stem}_A{j+1:02d}.txt")
                    else:
                        (output_path / 'labels' / split / f"{stem}_A{j+1:02d}.txt").touch()
                    
                    stats['train']['debris_aug'] += 1
            elif split == 'train':
                stats['train']['bg'] += 1
    
    # Count totals
    for split in ['train', 'val', 'test']:
        img_dir = output_path / 'images' / split
        if img_dir.exists():
            stats[split]['total'] = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    # Write data.yaml
    data_yaml = f"""# E5 Balanced Dataset — Debris Oversampled for Class Balance
# Generated by scripts/balance_e5_dataset.py
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
    s = stats['train']
    total_train = s['debris_orig'] + s['debris_aug'] + s['bg']
    total_debris = s['debris_orig'] + s['debris_aug']
    
    print("=" * 60)
    print("BALANCED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Train: {total_train} images")
    print(f"    Debris (original):  {s['debris_orig']:>4}")
    print(f"    Debris (augmented): {s['debris_aug']:>4}")
    print(f"    Background:         {s['bg']:>4}")
    print(f"    Ratio:              1 debris : {s['bg'] / max(total_debris, 1):.1f} bg")
    print(f"  Val:   {stats['val']['total']:>4} images (untouched)")
    print(f"  Test:  {stats['test']['total']:>4} images (untouched)")
    print(f"\n  Key improvement: Debris is now {target_ratio}x oversampled")
    print(f"  Training ratio improved from ~1:5 to ~1:{s['bg'] / max(total_debris, 1):.1f}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance E5 dataset by oversampling debris')
    parser.add_argument('--e5', default='datasets/noaa-debris/e5',
                       help='Path to E5 dataset')
    parser.add_argument('--output', default='datasets/noaa-debris/e5_balanced',
                       help='Output path for balanced dataset')
    parser.add_argument('--ratio', type=int, default=3,
                       help='Oversampling ratio for debris (default: 3)')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    balance_dataset(args.e5, args.output, args.ratio, args.seed)
