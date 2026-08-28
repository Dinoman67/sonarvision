"""
SSS Model Comparison Training
==============================
Trains and compares three YOLOv8 variants on SSS marine debris detection:
1. YOLOv8n (baseline) — standard YOLOv8 nano
2. SS-YOLO — GhostConv + FastC2f (47% fewer params)
3. YOLOv8-ESI — YOLOv8n + SE attention (better texture sensitivity)

Usage:
    python scripts/train_sss_comparison.py --dataset datasets/noaa-debris/e4/data.yaml
    python scripts/train_sss_comparison.py --dataset datasets/noaa-debris/e4/data.yaml --epochs 50

For Colab:
    !python scripts/train_sss_comparison.py --dataset /content/e4/data.yaml
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def save_ultralytics_checkpoint(model, filepath, data_yaml=None, args=None):
    """Save a custom model in ultralytics-compatible checkpoint format.

    Key: the checkpoint must contain 'ema' or 'model' as a torch.nn.Module,
    and the module must have .args, .task, .stride attributes.
    """
    # Add required attributes
    model.args = {
        'nc': 1,
        'task': 'detect',
    }
    model.task = 'detect'
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.0])
    if not hasattr(model, 'pt_path'):
        model.pt_path = str(filepath)

    ckpt = {
        'epoch': -1,
        'best_fitness': None,
        'model': None,
        'ema': model,  # ultralytics prefers 'ema' key
        'updates': None,
        'optimizer': None,
        'train_args': {
            'data': data_yaml or '',
            'epochs': args.epochs if args else 100,
            'imgsz': args.imgsz if args else 512,
            'batch': args.batch if args else 16,
        } if args else {},
        'train_metrics': {},
        'train_results': {},
        'date': datetime.now().isoformat(),
        'version': '8.4.129',
    }

    torch.save(ckpt, filepath)
    return filepath


def train_yolov8n(dataset_yaml, args, results_dir):
    """Train standard YOLOv8n baseline."""
    print(f"\n{'='*60}")
    print("Training: YOLOv8n (Baseline)")
    print(f"{'='*60}\n")

    model = YOLO('yolov8n.pt')

    start = time.time()
    model.train(
        data=dataset_yaml,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience,
        lr0=0.005, lrf=0.01, warmup_epochs=3,
        mosaic=0.0, mixup=0.0,
        fliplr=0.0, flipud=0.0, degrees=0.0,
        translate=0.05, scale=0.2,
        name='yolov8n', project=str(results_dir),
        exist_ok=True, plots=True,
    )
    training_time = time.time() - start

    val = model.val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25, verbose=False)
    n_params = sum(p.numel() for p in model.model.parameters())

    try:
        model.export(format='onnx', imgsz=args.imgsz,
                     save_dir=str(results_dir / 'yolov8n' / 'export'))
    except Exception:
        pass

    p, r = val.box.mp, val.box.mr
    return {
        'model': 'YOLOv8n', 'mAP50': val.box.map50, 'mAP50-95': val.box.map,
        'precision': p, 'recall': r,
        'f1': 2 * p * r / max(p + r, 1e-8),
        'training_time': training_time, 'n_params': n_params,
    }


def train_custom_model(model_key, build_fn, model_name, dataset_yaml, args, results_dir):
    """Train a custom SSS model using a monkey-patched trainer.

    SS-YOLO (GhostConv/FastC2f) trains from scratch — its architecture is
    incompatible with YOLOv8n pretrained weights.
    YOLOv8-ESI (C2f + SE) preserves pretrained weights — SE layers are new
    but lightweight and initialized well.
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}\n")

    model_dir = results_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build custom model — SS-YOLO uses pretrained=None (from scratch)
    if model_key == 'ss_yolo':
        custom_model = build_fn(pretrained=None)  # From scratch!
    else:
        custom_model = build_fn(pretrained='yolov8n.pt')  # Transfer learn
    n_params = sum(p.numel() for p in custom_model.parameters())
    print(f"  Built {model_name}: {n_params:,} params ({n_params/1e6:.2f}M)")

    # Monkey-patch get_model to return our custom model
    _orig_get_model = DetectionTrainer.get_model

    def _patched_get_model(self, cfg=None, weights=None, verbose=True):
        """Return our custom model instead of reconstructing from YAML."""
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        
        # Create a minimal model (needed for validator setup)
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        
        # Replace with our custom model
        model.model = custom_model.model
        model.nc = custom_model.nc if hasattr(custom_model, 'nc') else self.data['nc']
        model.names = custom_model.names if hasattr(custom_model, 'names') else {0: 'marine_debris'}
        
        # Only load pretrained weights for ESI (where transfer works)
        if weights and model_key != 'ss_yolo':
            try:
                model.load(weights)
            except Exception as e:
                print(f"  ⚠ Weight loading issue (expected for custom arch): {e}")
        
        return model

    DetectionTrainer.get_model = _patched_get_model

    try:
        yolo = YOLO('yolov8n.pt')

        # SS-YOLO needs more aggressive training (from scratch)
        if model_key == 'ss_yolo':
            train_kwargs = dict(
                lr0=0.01,       # Higher LR for from-scratch training
                lrf=0.05,       # Higher final LR
                warmup_epochs=5, # More warmup for from-scratch
                epochs=args.epochs + 50,  # Extra epochs for from-scratch
            )
        else:
            train_kwargs = dict(
                lr0=0.005, lrf=0.01, warmup_epochs=3,
                epochs=args.epochs,
            )

        start = time.time()
        yolo.train(
            data=dataset_yaml,
            imgsz=args.imgsz, batch=args.batch,
            patience=args.patience,
            mosaic=0.0, mixup=0.0,
            fliplr=0.0, flipud=0.0, degrees=0.0,
            translate=0.05, scale=0.2,
            name=model_key, project=str(results_dir),
            exist_ok=True, plots=True,
            **train_kwargs,
        )
        training_time = time.time() - start

        val = yolo.val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25, verbose=False)

        try:
            yolo.export(format='onnx', imgsz=args.imgsz,
                        save_dir=str(model_dir / 'export'))
        except Exception:
            pass

        p, r = val.box.mp, val.box.mr

        return {
            'model': model_name, 'mAP50': val.box.map50, 'mAP50-95': val.box.map,
            'precision': p, 'recall': r,
            'f1': 2 * p * r / max(p + r, 1e-8),
            'training_time': training_time, 'n_params': n_params,
        }

    except Exception as e:
        print(f"  ⚠ YOLO training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Restore original get_model
        DetectionTrainer.get_model = _orig_get_model


def _train_direct(model, model_name, dataset_yaml, args, model_dir):
    """Direct PyTorch training fallback."""
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.utils.loss import v8DetectionLoss

    import yaml as pyyaml

    # Load data config
    data_cfg = pyyaml.safe_load(open(dataset_yaml))
    if not os.path.isabs(data_cfg['path']):
        data_cfg['path'] = str(PROJECT_ROOT / data_cfg['path'])
    # Derive nc from names if not present
    if 'nc' not in data_cfg:
        names = data_cfg.get('names', {})
        if isinstance(names, dict):
            data_cfg['nc'] = len(names)
        elif isinstance(names, list):
            data_cfg['nc'] = len(names)
        else:
            data_cfg['nc'] = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).float()
    model.nc = data_cfg['nc']
    model.names = {0: 'marine_debris'}

    # Build loaders
    train_ds = build_yolo_dataset(data_cfg, args.imgsz, batch=args.batch,
                                   stride=max(int(model.stride.max()), 32),
                                   rank=-1, prefix='train')
    val_ds = build_yolo_dataset(data_cfg, args.imgsz, batch=args.batch,
                                 stride=max(int(model.stride.max()), 32),
                                 rank=-1, prefix='val')

    train_loader = build_dataloader(train_ds, args.batch, shuffle=True, rank=-1, workers=4)
    val_loader = build_dataloader(val_ds, args.batch, shuffle=False, rank=-1, workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0005)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005,
                                                      total_steps=total_steps)
    criterion = v8DetectionLoss(model)

    weights_dir = model_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_map = 0.0

    print(f"  Training {model_name} for {args.epochs} epochs on {device}...")

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0

        for batch in train_loader:
            imgs = batch['img'].to(device).float() / 255.0

            # Forward + loss
            preds = model(imgs)
            loss = criterion(preds, batch).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n += 1

        avg_loss = epoch_loss / max(n, 1)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch+1:>4}/{args.epochs} — loss: {avg_loss:.4f}")

            # Quick validation (just loss)
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                nv = 0
                for batch in val_loader:
                    imgs = batch['img'].to(device).float() / 255.0
                    preds = model(imgs)
                    loss = criterion(preds, batch).sum()
                    val_loss += loss.item()
                    nv += 1
                val_loss /= max(nv, 1)

            print(f"    val_loss: {val_loss:.4f}")

            # Save checkpoint
            torch.save(model.state_dict(), weights_dir / 'last.pt')
            if val_loss < best_map or best_map == 0:
                best_map = val_loss
                torch.save(model.state_dict(), weights_dir / 'best.pt')

    training_time = time.time() - start
    print(f"  ✓ Training complete in {training_time:.0f}s")

    # Full evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['img'].to(device).float() / 255.0
            preds = model(imgs)
            all_preds.append(preds)

    return {
        'model': model_name, 'mAP50': 0.0, 'mAP50-95': 0.0,
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'training_time': training_time, 'n_params': sum(p.numel() for p in model.parameters()),
    }


def generate_report(results, results_dir, dataset_yaml=None, imgsz=512):
    """Generate comparison report with confidence threshold sweep."""
    df = pd.DataFrame(results)
    df = df.sort_values('f1', ascending=False)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}\n")

    cols = ['model', 'n_params', 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1', 'training_time']
    display_df = df[cols].copy()
    display_df['n_params'] = display_df['n_params'].apply(lambda x: f"{x/1e6:.2f}M")
    display_df['training_time'] = display_df['training_time'].apply(lambda x: f"{x:.0f}s")
    print(display_df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, metric in enumerate(['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']):
        ax = axes[i // 3, i % 3]
        bars = ax.bar(df['model'], df[metric], color=colors[:len(df)])
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=10)

    ax = axes[1, 2]
    ax.bar(range(len(df)), df['n_params'] / 1e6, color=colors[:len(df)])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['model'], rotation=15)
    ax.set_title('PARAMETERS (M)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('SSS Debris Detection: Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    df.to_csv(results_dir / 'comparison.csv', index=False)

    best = df.iloc[0]
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {best['model']}")
    print(f"  F1: {best['f1']:.4f} | mAP50: {best['mAP50']:.4f} | "
          f"Params: {best['n_params']/1e6:.2f}M")
    print(f"{'='*60}")
    return df


def main():
    parser = argparse.ArgumentParser(description='SSS Model Comparison Training')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--output', type=str, default='results/sss_comparison')
    parser.add_argument('--conf-eval', type=float, default=0.1,
                       help='Confidence threshold for evaluation (lower = better recall)')
    parser.add_argument('--models', nargs='+',
                       default=['yolov8n', 'ss_yolo', 'yolov8_esi'],
                       choices=['yolov8n', 'ss_yolo', 'yolov8_esi'])
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU available")

    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    from models.build_sss_models import build_yolov8n_baseline, build_ss_yolo, build_yolov8_esi_full

    configs = {
        'yolov8n': ('YOLOv8n', build_yolov8n_baseline, True),
        'ss_yolo': ('SS-YOLO', build_ss_yolo, False),
        'yolov8_esi': ('YOLOv8-ESI', build_yolov8_esi_full, False),
    }

    all_results = []
    for key in args.models:
        name, builder, is_native = configs[key]
        if is_native:
            result = train_yolov8n(args.dataset, args, results_dir)
        else:
            result = train_custom_model(key, builder, name, args.dataset, args, results_dir)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        generate_report(all_results, results_dir,
                        dataset_yaml=args.dataset, imgsz=args.imgsz)

    print("\n✓ All done!")


if __name__ == '__main__':
    main()
