"""
Generate E5 Dataset — E4 + Realistic SSS Noise
================================================

Problem: E4 background images are too clean. 80% have identical texture
to debris images. Model fails on real noisy SSS data.

Solution: Add SSS-specific noise patterns:
1. Speckle noise (multiplicative — characteristic of coherent imaging)
2. Nadir line artifacts (center dropout — from sonar geometry)
3. Acoustic shadows (bright→dark wedge transitions)
4. Brightness/contrast variation (sonar gain drift)
5. Bottom texture variation (sand ripples, rock fields)

Output: datasets/noaa-debris/e5/
"""

import os
import sys
import random
import shutil
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image, ImageFilter

# ═══════════════════════════════════════════════════════════════════
# SSS Noise Functions
# ═══════════════════════════════════════════════════════════════════

def speckle_noise(img, intensity=0.3):
    """Multiplicative speckle noise — the #1 SSS artifact.
    
    Real SSS data has speckle because it's coherent imaging.
    This is the most important augmentation.
    """
    noise = np.random.lognormal(0, intensity, img.shape)
    return np.clip(img * noise, 0, 255).astype(np.uint8)


def nadir_artifact(img, drop_range=(2, 12)):
    """Nadir line — center dropout from sonar geometry.
    
    The area directly below the towfish has no return,
    creating a dark line down the center of the waterfall.
    """
    h, w = img.shape[:2]
    center_x = w // 2 + random.randint(-w // 10, w // 10)
    nadir_w = random.randint(*drop_range)
    
    result = img.copy()
    x1 = max(0, center_x - nadir_w // 2)
    x2 = min(w, center_x + nadir_w // 2)
    
    # Smooth transition
    for x in range(x1, x2):
        dist = abs(x - center_x) / max(nadir_w / 2, 1)
        result[:, x] = (result[:, x] * (0.05 + 0.15 * dist)).astype(np.uint8)
    
    return result


def acoustic_shadow(img, num_shadows=1):
    """Acoustic shadows — bright-to-dark transitions.
    
    Objects cast acoustic shadows in SSS data. These create
    sharp brightness transitions that can confuse detectors.
    """
    h, w = img.shape[:2]
    result = img.copy()
    
    for _ in range(num_shadows):
        # Shadow parameters
        angle = random.uniform(15, 75)
        cx = random.randint(0, w)
        cy = random.randint(0, h // 3)
        darkness = random.uniform(0.1, 0.4)
        
        Y, X = np.ogrid[:h, :w]
        mask = ((X - cx) * np.tan(np.radians(angle)) + cy) < Y
        
        # Smooth shadow edge
        shadow = np.ones_like(result, dtype=float)
        shadow[mask] = darkness
        result = np.clip(result * shadow, 0, 255).astype(np.uint8)
    
    return result


def brightness_contrast_jitter(img, alpha_range=(0.6, 1.4), beta_range=(-30, 30)):
    """Sonar gain variation — brightness and contrast drift.
    
    Real SSS data varies in brightness depending on:
    - Seafloor composition (sand vs rock)
    - Water depth
    - Towfish altitude
    """
    alpha = random.uniform(*alpha_range)  # contrast
    beta = random.randint(*beta_range)    # brightness
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)


def bottom_texture_overlay(img, texture_type='sand_ripple'):
    """Overlay realistic seabed textures.
    
    Adds patterns like:
    - Sand ripples (periodic horizontal lines)
    - Rock fields (random bright spots)
    - Mud flat (low-contrast smooth areas)
    """
    h, w = img.shape[:2]
    result = img.copy().astype(float)
    
    if texture_type == 'sand_ripple':
        # Periodic horizontal lines
        freq = random.uniform(0.02, 0.08)
        phase = random.uniform(0, 2 * np.pi)
        amplitude = random.uniform(5, 20)
        Y = np.arange(h).reshape(-1, 1)
        ripple = amplitude * np.sin(2 * np.pi * freq * Y + phase)
        result += ripple
    
    elif texture_type == 'rock_field':
        # Random bright spots (rocks)
        n_rocks = random.randint(5, 30)
        for _ in range(n_rocks):
            cy, cx = random.randint(0, h), random.randint(0, w)
            radius = random.randint(2, 8)
            brightness = random.uniform(30, 80)
            Y, X = np.ogrid[:h, :w]
            mask = ((X - cx)**2 + (Y - cy)**2) < radius**2
            result[mask] += brightness
    
    elif texture_type == 'seagrass':
        # Vertical wavy lines
        freq = random.uniform(0.05, 0.15)
        amplitude = random.uniform(3, 10)
        X = np.arange(w).reshape(1, -1)
        wave = amplitude * np.sin(2 * np.pi * freq * X)
        result += wave
    
    return np.clip(result, 0, 255).astype(np.uint8)


def horizontal_stripes(img, n_stripes=(2, 6)):
    """Horizontal intensity stripes — from sonar gain variation along track."""
    h, w = img.shape[:2]
    result = img.copy()
    n = random.randint(*n_stripes)
    
    for _ in range(n):
        y1 = random.randint(0, h - 10)
        stripe_h = random.randint(2, 15)
        y2 = min(y1 + stripe_h, h)
        gain = random.uniform(0.5, 1.5)
        result[y1:y2] = np.clip(result[y1:y2] * gain, 0, 255).astype(np.uint8)
    
    return result


def apply_random_sss_noise(img, intensity='medium'):
    """Apply a random combination of SSS noise patterns.
    
    intensity: 'light', 'medium', 'heavy'
    """
    configs = {
        'light': {'n_augments': (1, 2), 'speckle': 0.15, 'contrast': (0.8, 1.2)},
        'medium': {'n_augments': (2, 3), 'speckle': 0.3, 'contrast': (0.7, 1.3)},
        'heavy': {'n_augments': (3, 5), 'speckle': 0.5, 'contrast': (0.5, 1.5)},
    }
    cfg = configs[intensity]
    
    result = img.copy()
    
    # Always add speckle
    result = speckle_noise(result, cfg['speckle'])
    
    # Random additional augmentations
    extra_augs = [nadir_artifact, acoustic_shadow, brightness_contrast_jitter,
                  bottom_texture_overlay, horizontal_stripes]
    n_extra = random.randint(*cfg['n_augments'])
    
    for _ in range(n_extra):
        aug = random.choice(extra_augs)
        if aug == bottom_texture_overlay:
            tex = random.choice(['sand_ripple', 'rock_field', 'seagrass'])
            result = aug(result, tex)
        elif aug == brightness_contrast_jitter:
            result = aug(result, alpha_range=cfg['contrast'])
        else:
            result = aug(result)
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Dataset Generation
# ═══════════════════════════════════════════════════════════════════

def generate_e5(e4_path, e5_path, n_noisy_per_bg=3, seed=42):
    """Generate E5 dataset with noisy backgrounds.
    
    Args:
        e4_path: Path to E4 dataset
        e5_path: Output path for E5 dataset
        n_noisy_per_bg: Number of noisy versions per BG image
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    e4_path = Path(e4_path)
    e5_path = Path(e5_path)
    
    print(f"Generating E5 dataset from E4...")
    print(f"  Source: {e4_path}")
    print(f"  Output: {e5_path}")
    print(f"  Noisy versions per BG: {n_noisy_per_bg}")
    print()
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (e5_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (e5_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {'train': {'pos': 0, 'bg_orig': 0, 'bg_noisy': 0},
             'val': {'pos': 0, 'bg': 0},
             'test': {'pos': 0, 'bg': 0}}
    
    for split in ['train', 'val', 'test']:
        img_dir = e4_path / 'images' / split
        lbl_dir = e4_path / 'labels' / split
        
        if not img_dir.exists():
            print(f"  Skipping {split} — not found")
            continue
        
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        for img_name in images:
            stem = img_name.replace('.png', '')
            lbl_name = f"{stem}.txt"
            lbl_path = lbl_dir / lbl_name
            
            # Copy original
            shutil.copy(img_dir / img_name, e5_path / 'images' / split / img_name)
            if lbl_path.exists():
                shutil.copy(lbl_path, e5_path / 'labels' / split / lbl_name)
            else:
                (e5_path / 'labels' / split / lbl_name).touch()
            
            is_bg = 'BG' in img_name
            
            if split == 'train' and is_bg:
                stats['train']['bg_orig'] += 1
                
                # Generate noisy versions
                orig = np.array(Image.open(img_dir / img_name))
                
                for j in range(n_noisy_per_bg):
                    # Vary intensity: some light, some heavy
                    intensity = random.choice(['light', 'medium', 'medium', 'heavy'])
                    noisy = apply_random_sss_noise(orig, intensity)
                    
                    noisy_name = f"{stem}_N{j+1:02d}.png"
                    Image.fromarray(noisy).save(e5_path / 'images' / split / noisy_name)
                    (e5_path / 'labels' / split / f"{stem}_N{j+1:02d}.txt").touch()
                    stats['train']['bg_noisy'] += 1
            elif is_bg:
                stats[split]['bg'] += 1
            else:
                stats[split]['pos'] += 1
    
    # Write data.yaml
    data_yaml = f"""# E5 Dataset — E4 + Realistic SSS Noise
# Generated by scripts/generate_e5_noisy.py
path: {e5_path}
train: images/train
val: images/val
test: images/test

names:
  0: marine_debris
"""
    with open(e5_path / 'data.yaml', 'w') as f:
        f.write(data_yaml)
    
    # Print summary
    print("=" * 50)
    print("E5 DATASET SUMMARY")
    print("=" * 50)
    
    total_imgs = 0
    for split in ['train', 'val', 'test']:
        n = len([f for f in os.listdir(e5_path / 'images' / split) if f.endswith('.png')])
        total_imgs += n
        s = stats[split]
        if split == 'train':
            print(f"  {split:>5}: {n:>4} images ({s['pos']} pos + {s['bg_orig']} bg_orig + {s['bg_noisy']} bg_noisy)")
        else:
            print(f"  {split:>5}: {n:>4} images ({s.get('pos',0)} pos + {s.get('bg',0)} bg)")
    
    print(f"  {'TOTAL':>5}: {total_imgs:>4} images")
    print(f"\n  Key improvement: {stats['train']['bg_noisy']} noisy BG images added")
    print(f"  Model will now see realistic SSS noise patterns!")
    
    return e5_path


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_augmentations(e4_path, n_samples=4):
    """Show augmentation examples."""
    import matplotlib.pyplot as plt
    
    train_dir = Path(e4_path) / 'images' / 'train'
    bg_imgs = sorted([f for f in os.listdir(train_dir) if 'BG' in f])[:n_samples]
    
    fig, axes = plt.subplots(4, n_samples, figsize=(4 * n_samples, 16))
    
    for j, bg_name in enumerate(bg_imgs):
        orig = np.array(Image.open(train_dir / bg_name))
        
        axes[0, j].imshow(orig, cmap='gray')
        axes[0, j].set_title(f'Original\n{bg_name[:20]}', fontsize=9)
        axes[0, j].axis('off')
        
        for i, intensity in enumerate(['light', 'medium', 'heavy']):
            random.seed(42 + i * 100 + j)
            np.random.seed(42 + i * 100 + j)
            noisy = apply_random_sss_noise(orig, intensity)
            axes[i + 1, j].imshow(noisy, cmap='gray')
            axes[i + 1, j].set_title(f'{intensity.capitalize()} noise', fontsize=9)
            axes[i + 1, j].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=11, rotation=0, labelpad=80)
    axes[1, 0].set_ylabel('Light', fontsize=11, rotation=0, labelpad=80)
    axes[2, 0].set_ylabel('Medium', fontsize=11, rotation=0, labelpad=80)
    axes[3, 0].set_ylabel('Heavy', fontsize=11, rotation=0, labelpad=80)
    
    plt.suptitle('SSS Noise Augmentation Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('scripts/e5_augmentation_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Preview saved to scripts/e5_augmentation_preview.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate E5 dataset with SSS noise')
    parser.add_argument('--e4', default='datasets/noaa-debris/e4',
                       help='Path to E4 dataset')
    parser.add_argument('--e5', default='datasets/noaa-debris/e5',
                       help='Output path for E5 dataset')
    parser.add_argument('--noisy-per-bg', type=int, default=3,
                       help='Noisy versions per BG image (default: 3)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show augmentation examples')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_augmentations(args.e4)
    else:
        generate_e5(args.e4, args.e5, args.noisy_per_bg, args.seed)
