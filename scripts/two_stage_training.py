#!/usr/bin/env python3
"""
Two-Stage SSS Debris Detection Training
=========================================

Stage 1: COMPETITION (Quick)
  - Train all 3 models for 30 epochs each
  - Evaluate on validation set
  - Select winner based on F1 score (balance of precision + recall)

Stage 2: WINNER TRAINING (Thorough)
  - Train winner for 150+ epochs with optimized hyperparameters
  - Extended patience, learning rate scheduling
  - Final evaluation on test set

Usage (Colab):
    !python scripts/two_stage_training.py --dataset /content/e5_balanced/data.yaml
    
Local testing:
    python scripts/two_stage_training.py --dataset datasets/noaa-debris/e5_balanced/data.yaml --dry-run
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

STAGE1_CONFIG = {
    'epochs': 30,           # Quick competition
    'patience': 15,         # Early stopping
    'lr0': 0.01,            # Higher LR for quick convergence
    'warmup_epochs': 2,
}

STAGE2_CONFIG = {
    'epochs': 150,          # Thorough training
    'patience': 40,         # More patience
    'lr0': 0.005,           # Lower LR for fine-tuning
    'warmup_epochs': 3,
}

# Model-specific configs
MODEL_CONFIGS = {
    'yolov8n': {
        'name': 'YOLOv8n',
        'pretrained': 'yolov8n.pt',
        'from_scratch': False,
        'extra_epochs': 0,
        'lr0': None,  # Use default
    },
    'ss_yolo': {
        'name': 'SS-YOLO',
        'pretrained': None,  # From scratch!
        'from_scratch': True,
        'extra_epochs': 50,  # Needs more epochs from scratch
        'lr0': 0.01,  # Higher LR for from-scratch
        'warmup_epochs': 5,
    },
    'yolov8_esi': {
        'name': 'YOLOv8-ESI',
        'pretrained': 'yolov8n.pt',
        'from_scratch': False,
        'extra_epochs': 0,
        'lr0': None,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: COMPETITION
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_competition(dataset_yaml, args, results_dir):
    """Train all models for a quick competition to find the winner."""
    from ultralytics import YOLO
    from models.build_sss_models import build_ss_yolo, build_yolov8_esi_full
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    
    print("\n" + "="*70)
    print("STAGE 1: MODEL COMPETITION")
    print("="*70)
    print(f"Training each model for {STAGE1_CONFIG['epochs']} epochs...")
    print()
    
    stage1_dir = results_dir / 'stage1_competition'
    stage1_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # ─── Train YOLOv8n ──────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("Training: YOLOv8n (Baseline)")
    print("─"*50)
    
    model_v8n = YOLO('yolov8n.pt')
    
    train_params = {
        'data': dataset_yaml,
        'epochs': STAGE1_CONFIG['epochs'],
        'imgsz': args.imgsz,
        'batch': args.batch,
        'patience': STAGE1_CONFIG['patience'],
        'lr0': STAGE1_CONFIG['lr0'],
        'lrf': 0.01,
        'warmup_epochs': STAGE1_CONFIG['warmup_epochs'],
        'mosaic': 0.0, 'mixup': 0.0,
        'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0,
        'translate': 0.05, 'scale': 0.2,
        'name': 'yolov8n_stage1',
        'project': str(stage1_dir),
        'exist_ok': True,
        'plots': True,
    }
    
    start = time.time()
    model_v8n.train(**train_params)
    time_v8n = time.time() - start
    
    val_v8n = model_v8n.val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25, verbose=False)
    p, r = val_v8n.box.mp, val_v8n.box.mr
    f1 = 2*p*r / max(p+r, 1e-8)
    
    results['yolov8n'] = {
        'model': model_v8n,
        'mAP50': val_v8n.box.map50,
        'precision': p,
        'recall': r,
        'f1': f1,
        'time': time_v8n,
        'name': 'YOLOv8n',
    }
    
    print(f"  Result: mAP50={val_v8n.box.map50:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    
    # ─── Train SS-YOLO ──────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("Training: SS-YOLO (from scratch)")
    print("─"*50)
    
    ss_model = build_ss_yolo(pretrained=None)  # From scratch!
    
    _orig = DetectionTrainer.get_model
    def _patched_ss(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose and RANK == -1)
        )
        model.model = ss_model.model
        model.nc = 1
        model.names = {0: 'marine_debris'}
        return model
    DetectionTrainer.get_model = _patched_ss
    
    try:
        yolo_ss = YOLO('yolov8n.pt')  # Template only
        ss_params = train_params.copy()
        ss_params.update({
            'name': 'ss_yolo_stage1',
            'lr0': MODEL_CONFIGS['ss_yolo']['lr0'],  # Higher LR for from-scratch
            'warmup_epochs': MODEL_CONFIGS['ss_yolo']['warmup_epochs'],
        })
        
        start = time.time()
        yolo_ss.train(**ss_params)
        time_ss = time.time() - start
        
        val_ss = yolo_ss.val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25, verbose=False)
        p, r = val_ss.box.mp, val_ss.box.mr
        f1 = 2*p*r / max(p+r, 1e-8)
        
        results['ss_yolo'] = {
            'model': yolo_ss,
            'mAP50': val_ss.box.map50,
            'precision': p,
            'recall': r,
            'f1': f1,
            'time': time_ss,
            'name': 'SS-YOLO',
        }
        
        print(f"  Result: mAP50={val_ss.box.map50:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        
    except Exception as e:
        print(f"  ✗ SS-YOLO failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        DetectionTrainer.get_model = _orig
    
    # ─── Train YOLOv8-ESI ───────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("Training: YOLOv8-ESI (SE Attention)")
    print("─"*50)
    
    esi_model = build_yolov8_esi_full()
    
    _orig2 = DetectionTrainer.get_model
    def _patched_esi(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose and RANK == -1)
        )
        model.model = esi_model.model
        model.nc = 1
        model.names = {0: 'marine_debris'}
        try:
            model.load(weights)
        except Exception:
            pass
        return model
    DetectionTrainer.get_model = _patched_esi
    
    try:
        yolo_esi = YOLO('yolov8n.pt')
        esi_params = train_params.copy()
        esi_params['name'] = 'yolov8_esi_stage1'
        
        start = time.time()
        yolo_esi.train(**esi_params)
        time_esi = time.time() - start
        
        val_esi = yolo_esi.val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25, verbose=False)
        p, r = val_esi.box.mp, val_esi.box.mr
        f1 = 2*p*r / max(p+r, 1e-8)
        
        results['yolov8_esi'] = {
            'model': yolo_esi,
            'mAP50': val_esi.box.map50,
            'precision': p,
            'recall': r,
            'f1': f1,
            'time': time_esi,
            'name': 'YOLOv8-ESI',
        }
        
        print(f"  Result: mAP50={val_esi.box.map50:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        
    except Exception as e:
        print(f"  ✗ YOLOv8-ESI failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        DetectionTrainer.get_model = _orig2
    
    # ─── Select Winner ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 1 RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<15} {'mAP50':<10} {'P':<10} {'R':<10} {'F1':<10} {'Time':<10}")
    print("-"*65)
    
    for key, r in results.items():
        print(f"{r['name']:<15} {r['mAP50']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1']:<10.4f} {r['time']:<10.0f}s")
    
    # Select winner by F1 score
    winner_key = max(results.keys(), key=lambda k: results[k]['f1'])
    winner = results[winner_key]
    
    print(f"\n{'='*70}")
    print(f"🏆 WINNER: {winner['name']} (F1={winner['f1']:.4f})")
    print(f"{'='*70}")
    
    return results, winner_key, winner


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: WINNER TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def stage2_winner_training(winner_key, winner_model, dataset_yaml, args, results_dir):
    """Train the winner thoroughly with optimized hyperparameters."""
    from ultralytics import YOLO
    
    print("\n" + "="*70)
    print("STAGE 2: WINNER TRAINING")
    print("="*70)
    print(f"Training {winner_model['name']} for {STAGE2_CONFIG['epochs']} epochs...")
    print()
    
    stage2_dir = results_dir / 'stage2_winner'
    stage2_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Stage 2 params based on model type
    config = MODEL_CONFIGS[winner_key]
    
    train_params = {
        'data': dataset_yaml,
        'epochs': STAGE2_CONFIG['epochs'] + config.get('extra_epochs', 0),
        'imgsz': args.imgsz,
        'batch': args.batch,
        'patience': STAGE2_CONFIG['patience'],
        'lr0': config.get('lr0') or STAGE2_CONFIG['lr0'],
        'lrf': 0.01,
        'warmup_epochs': config.get('warmup_epochs', STAGE2_CONFIG['warmup_epochs']),
        'mosaic': 0.0, 'mixup': 0.0,
        'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0,
        'translate': 0.05, 'scale': 0.2,
        'name': f'{winner_key}_stage2_final',
        'project': str(stage2_dir),
        'exist_ok': True,
        'plots': True,
    }
    
    # For from-scratch models (SS-YOLO), use more aggressive training
    if config['from_scratch']:
        print(f"  Training from scratch — using extended schedule")
        train_params['epochs'] += 50  # Extra 50 epochs
        train_params['lr0'] = 0.01   # Higher LR
        train_params['warmup_epochs'] = 5
    
    start = time.time()
    winner_model['model'].train(**train_params)
    total_time = time.time() - start
    
    # Final evaluation
    print("\n" + "─"*50)
    print("FINAL EVALUATION")
    print("─"*50)
    
    val_results = winner_model['model'].val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25)
    test_results = winner_model['model'].val(data=dataset_yaml, imgsz=args.imgsz, conf=0.25, split='test')
    
    # Confidence threshold sweep
    print("\nConfidence threshold sweep (val set):")
    best_conf = 0.25
    best_f1 = 0
    
    for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        r = winner_model['model'].val(data=dataset_yaml, imgsz=args.imgsz, conf=conf, verbose=False)
        p, rv = r.box.mp, r.box.mr
        f1 = 2*p*rv / max(p+rv, 1e-8)
        marker = " ← BEST" if f1 > best_f1 else ""
        print(f"  conf={conf:.2f}: P={p:.4f}, R={rv:.4f}, F1={f1:.4f}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf
    
    print(f"\n  Optimal confidence: {best_conf}")
    
    # Re-evaluate with best confidence
    test_final = winner_model['model'].val(data=dataset_yaml, imgsz=args.imgsz, 
                                           conf=best_conf, split='test')
    
    # Summary
    p, r = test_final.box.mp, test_final.box.mr
    f1 = 2*p*r / max(p+r, 1e-8)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Model:       {winner_model['name']}")
    print(f"  Epochs:      {train_params['epochs']}")
    print(f"  Total time:  {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Best conf:   {best_conf}")
    print(f"  Test mAP50:  {test_final.box.map50:.4f}")
    print(f"  Test P:      {p:.4f}")
    print(f"  Test R:      {r:.4f}")
    print(f"  Test F1:     {f1:.4f}")
    
    # Export
    try:
        export_dir = stage2_dir / winner_key / 'export'
        export_dir.mkdir(exist_ok=True)
        winner_model['model'].export(format='onnx', imgsz=args.imgsz,
                                     save_dir=str(export_dir))
        print(f"\n  ✓ Exported to ONNX: {export_dir}")
    except Exception as e:
        print(f"\n  ⚠ Export failed: {e}")
    
    # Save report
    report = {
        'winner': winner_model['name'],
        'winner_key': winner_key,
        'stage2_epochs': train_params['epochs'],
        'best_confidence': best_conf,
        'test_mAP50': test_final.box.map50,
        'test_precision': p,
        'test_recall': r,
        'test_f1': f1,
        'total_time_seconds': total_time,
    }
    
    report_path = stage2_dir / 'final_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Report saved: {report_path}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Two-Stage SSS Training')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset YAML')
    parser.add_argument('--imgsz', type=int, default=512,
                       help='Image size')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--output', type=str, default='results/two_stage',
                       help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print plan without training')
    args = parser.parse_args()
    
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TWO-STAGE SSS DEBRIS DETECTION TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.dry_run:
        print("DRY RUN — Training plan:")
        print()
        print("Stage 1: COMPETITION")
        print(f"  - Train YOLOv8n for {STAGE1_CONFIG['epochs']} epochs")
        print(f"  - Train SS-YOLO for {STAGE1_CONFIG['epochs']} epochs (from scratch)")
        print(f"  - Train YOLOv8-ESI for {STAGE1_CONFIG['epochs']} epochs")
        print(f"  - Select winner by F1 score")
        print()
        print("Stage 2: WINNER TRAINING")
        print(f"  - Train winner for {STAGE2_CONFIG['epochs']}+ epochs")
        print(f"  - Confidence threshold sweep")
        print(f"  - Final evaluation on test set")
        print(f"  - Export to ONNX")
        return
    
    # Stage 1: Competition
    stage1_results, winner_key, winner = stage1_competition(
        args.dataset, args, results_dir
    )
    
    # Stage 2: Winner Training
    final_report = stage2_winner_training(
        winner_key, winner, args.dataset, args, results_dir
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Winner: {final_report['winner']}")
    print(f"Test F1: {final_report['test_f1']:.4f}")
    print(f"Results: {results_dir}")


if __name__ == '__main__':
    main()
