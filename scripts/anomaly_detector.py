#!/usr/bin/env python3
"""
SSS Debris Anomaly Detector

Train a convolutional autoencoder on NORMAL seabed patches only.
Detect debris by measuring reconstruction error — debris patches
reconstruct poorly because the model has never seen them.

No positive examples needed for training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import json


class SeabedPatchDataset(Dataset):
    """Dataset of seabed image patches."""
    def __init__(self, image_dir, label_dir=None, mode='train', target_size=128):
        self.target_size = target_size
        self.patches = []
        
        if label_dir:
            # Load images with their labels
            img_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            for img_path in img_files:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(label_dir, f'{basename}.txt')
                
                has_object = False
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    has_object = True
                
                if mode == 'train':
                    # Only train on BACKGROUND (no objects)
                    if not has_object:
                        self.patches.append(img_path)
                else:
                    # Test on everything
                    self.patches.append((img_path, has_object))
        else:
            # Just images, no labels
            self.patches = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        if isinstance(self.patches[idx], tuple):
            img_path, has_object = self.patches[idx]
        else:
            img_path = self.patches[idx]
            has_object = False
        
        img = Image.open(img_path).convert('L')  # grayscale
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) * 2 - 1
        
        if isinstance(self.patches[idx], tuple):
            return img_tensor, has_object, img_path
        return img_tensor


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for SSS anomaly detection.
    
    Architecture:
    - Encoder: 4 conv blocks → compressed latent space
    - Decoder: 4 conv transpose blocks → reconstructed image
    - Reconstruction error = anomaly score
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 128x128x1 → 64x64x32
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 64x64x32 → 32x32x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 32x32x64 → 16x16x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 16x16x128 → 8x8x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Bottleneck
        self.flatten_size = 256 * 8 * 8
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 8x8x256 → 16x16x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16x16x128 → 32x32x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32x32x64 → 64x64x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 64x64x32 → 128x128x1
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_encode(h)
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train_autoencoder(model, train_loader, epochs=100, lr=1e-3, device='cpu'):
    """Train autoencoder on normal seabed patches."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
            else:
                imgs = batch.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(imgs)
            loss = criterion(reconstructed, imgs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    return train_losses


def compute_anomaly_scores(model, dataset, device='cpu'):
    """Compute reconstruction error for each patch."""
    model.eval()
    scores = []
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
                has_objects = batch[1]
                paths = batch[2]
            else:
                imgs = batch.to(device)
                has_objects = [False] * len(batch)
                paths = [''] * len(batch)
            
            reconstructed = model(imgs)
            
            # Per-pixel reconstruction error
            error = (imgs - reconstructed).pow(2).mean(dim=[1, 2, 3])
            
            for i in range(len(imgs)):
                ho = has_objects[i] if isinstance(has_objects, list) else bool(has_objects[i])
                pa = paths[i] if isinstance(paths, list) else paths
                scores.append({
                    'score': error[i].item(),
                    'has_object': ho,
                    'path': pa,
                })
    
    return scores


def main():
    print("=" * 70)
    print("SSS ANOMALY DETECTOR")
    print("Train on normal seabed → detect debris by reconstruction error")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'noaa-debris', 'e4')
    
    # ── Step 1: Train on background patches ──
    print("\n--- Step 1: Loading training data (backgrounds only) ---")
    train_dataset = SeabedPatchDataset(
        os.path.join(base, 'images', 'train'),
        os.path.join(base, 'labels', 'train'),
        mode='train',
        target_size=128
    )
    print(f"  Training patches: {len(train_dataset)} (backgrounds only)")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # ── Step 2: Train autoencoder ──
    print("\n--- Step 2: Training autoencoder ---")
    model = ConvAutoencoder(latent_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")
    
    losses = train_autoencoder(model, train_loader, epochs=100, lr=1e-3, device=device)
    
    # ── Step 3: Compute anomaly scores on all splits ──
    print("\n--- Step 3: Computing anomaly scores ---")
    
    results = {}
    for split in ['train', 'val', 'test']:
        dataset = SeabedPatchDataset(
            os.path.join(base, 'images', split),
            os.path.join(base, 'labels', split),
            mode='test',
            target_size=128
        )
        scores = compute_anomaly_scores(model, dataset, device=device)
        
        bg_scores = [s['score'] for s in scores if not s['has_object']]
        obj_scores = [s['score'] for s in scores if s['has_object']]
        
        results[split] = {
            'n_total': len(scores),
            'n_bg': len(bg_scores),
            'n_obj': len(obj_scores),
            'bg_mean': float(np.mean(bg_scores)) if bg_scores else 0,
            'bg_std': float(np.std(bg_scores)) if bg_scores else 0,
            'obj_mean': float(np.mean(obj_scores)) if obj_scores else 0,
            'obj_std': float(np.std(obj_scores)) if obj_scores else 0,
        }
        
        print(f"\n  {split}:")
        print(f"    Backgrounds: {len(bg_scores)} | mean={results[split]['bg_mean']:.6f} std={results[split]['bg_std']:.6f}")
        if obj_scores:
            print(f"    Objects:     {len(obj_scores)} | mean={results[split]['obj_mean']:.6f} std={results[split]['obj_std']:.6f}")
            
            # Find best threshold
            all_scores = [(s, True) for s in obj_scores] + [(s, False) for s in bg_scores]
            all_scores.sort(key=lambda x: -x[0])
            
            # Sweep thresholds
            best_f1 = 0
            best_thresh = 0
            for thresh in np.linspace(min(s[0] for s in all_scores), max(s[0] for s in all_scores), 100):
                tp = sum(1 for s, is_obj in all_scores if s >= thresh and is_obj)
                fp = sum(1 for s, is_obj in all_scores if s >= thresh and not is_obj)
                fn = sum(1 for s, is_obj in all_scores if s < thresh and is_obj)
                p = tp / max(tp + fp, 1)
                r = tp / max(tp + fn, 1)
                f1 = 2 * p * r / max(p + r, 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_p, best_r = p, r
            
            print(f"    Best threshold: {best_thresh:.6f}")
            print(f"    Best F1: {best_f1:.4f} (P={best_p:.4f}, R={best_r:.4f})")
            results[split]['best_threshold'] = float(best_thresh)
            results[split]['best_f1'] = float(best_f1)
            results[split]['best_precision'] = float(best_p)
            results[split]['best_recall'] = float(best_r)
    
    # ── Step 4: Per-target analysis on test set ──
    print("\n--- Step 4: Per-target detection on test ---")
    test_dataset = SeabedPatchDataset(
        os.path.join(base, 'images', 'test'),
        os.path.join(base, 'labels', 'test'),
        mode='test',
        target_size=128
    )
    test_scores = compute_anomaly_scores(model, test_dataset, device=device)
    
    best_thresh = results['test'].get('best_threshold', 0)
    
    from collections import defaultdict
    target_stats = defaultdict(lambda: {'detected': 0, 'total': 0, 'scores': []})
    
    for s in test_scores:
        path = s['path']
        fname = os.path.basename(path)
        if 'BG' in fname:
            target = 'BG'
        else:
            parts = os.path.splitext(fname)[0].split('_')
            target = parts[1]
        
        target_stats[target]['total'] += 1
        target_stats[target]['scores'].append(s['score'])
        if s['score'] >= best_thresh:
            target_stats[target]['detected'] += 1
    
    print(f"{'Target':<12} {'Images':>6} {'Detected':>8} {'Rate':>8} {'AvgScore':>10}")
    print("-" * 55)
    for target in sorted(target_stats.keys()):
        ts = target_stats[target]
        rate = ts['detected'] / ts['total'] if ts['total'] > 0 else 0
        avg_score = np.mean(ts['scores'])
        flag = " ⚠️" if target == 'BG' and rate > 0.1 else ""
        print(f"{target:<12} {ts['total']:>6} {ts['detected']:>8} {rate:>7.1%} {avg_score:>10.6f}{flag}")
    
    # ── Step 5: Save model and report ──
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, 'sss_anomaly_detector.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': 64,
        'target_size': 128,
        'threshold': best_thresh,
        'train_losses': losses,
        'results': results,
    }, model_path)
    model_size = os.path.getsize(model_path) / 1e6
    print(f"\n✓ Model saved: {model_path} ({model_size:.1f} MB)")
    
    # Save report
    report_path = os.path.join(out_dir, 'anomaly_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'model_params': n_params,
            'model_size_mb': model_size,
            'device': device,
            'results': results,
        }, f, indent=2)
    print(f"✓ Report saved: {report_path}")


if __name__ == "__main__":
    main()
