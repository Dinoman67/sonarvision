"""
SSS Model Training Script
=========================
Compares YOLOv8n (baseline), YOLOv8-ESI, and SS-YOLO on side-scan sonar debris detection.

Usage:
    python scripts/train_sss_models.py --dataset datasets/noaa-debris/e4/data.yaml
    
Requirements:
    pip install ultralytics torch matplotlib pandas
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Import custom modules
from models.sss_custom_modules import (
    WaveletConv, FastC2f, GhostConv, 
    DepthwiseSeparableConv, SEBlock, CBAM
)


def register_custom_modules():
    """Register custom modules with ultralytics for YAML parsing."""
    import ultralytics.nn.modules.block as block_module
    import ultralytics.nn.modules.conv as conv_module
    
    # Register in block module (for C2f-like modules)
    block_module.FastC2f = FastC2f
    block_module.SEBlock = SEBlock
    block_module.CBAM = CBAM
    block_module.DepthwiseSeparableConv = DepthwiseSeparableConv
    
    # Register in conv module (for Conv-like modules)
    conv_module.WaveletConv = WaveletConv
    conv_module.GhostConv = GhostConv
    
    # Also register in ultralytics.nn.tasks
    import ultralytics.nn.tasks as tasks_module
    tasks_module.FastC2f = FastC2f
    tasks_module.WaveletConv = WaveletConv
    tasks_module.GhostConv = GhostConv
    tasks_module.SEBlock = SEBlock
    tasks_module.CBAM = CBAM
    tasks_module.DepthwiseSeparableConv = DepthwiseSeparableConv
    
    print("✓ Custom modules registered with ultralytics")


def get_model_configs():
    """Return model configurations."""
    return {
        'yolov8n': {
            'name': 'YOLOv8n (Baseline)',
            'config': 'yolov8n.yaml',
            'pretrained': 'yolov8n.pt',
            'description': 'Standard YOLOv8 nano - baseline comparison'
        },
        'yolov8_esi': {
            'name': 'YOLOv8-ESI (Wavelet)',
            'config': str(PROJECT_ROOT / 'models' / 'yolov8_esi.yaml'),
            'pretrained': 'yolov8n.pt',  # Transfer learn from YOLOv8n
            'description': 'YOLOv8 + Wavelet convolution for SSS textures'
        },
        'ss_yolo': {
            'name': 'SS-YOLO (Lightweight)',
            'config': str(PROJECT_ROOT / 'models' / 'ss_yolo.yaml'),
            'pretrained': 'yolov8n.pt',  # Transfer learn from YOLOv8n
            'description': 'Lightweight SSS model with Fast-C2f + GhostConv'
        }
    }


def train_model(model_key, dataset_yaml, args, results_dir):
    """Train a single model and return results."""
    
    configs = get_model_configs()
    config = configs[model_key]
    
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")
    
    # Create output directory
    model_results_dir = results_dir / model_key
    model_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    if model_key == 'yolov8n':
        model = YOLO(config['pretrained'])
    else:
        # For custom architectures, we need to load from YAML
        # and then load pretrained weights
        try:
            model = YOLO(config['config'])
            # Try to load pretrained weights (will skip mismatched layers)
            if config['pretrained']:
                print(f"Loading pretrained weights from {config['pretrained']}...")
                pretrained = torch.load(config['pretrained'], map_location='cpu')
                if 'model' in pretrained:
                    pretrained = pretrained['model']
                # Load with strict=False to handle architecture differences
                model.model.load_state_dict(pretrained, strict=False)
                print("✓ Pretrained weights loaded (partial transfer learning)")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Falling back to standard YOLOv8n...")
            model = YOLO('yolov8n.pt')
    
    # Training parameters (SSS-optimized)
    train_params = {
        'data': dataset_yaml,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'patience': args.patience,
        'lr0': 0.005,
        'lrf': 0.01,
        'warmup_epochs': 3,
        # SSS-friendly: no flips, no rotation
        'mosaic': 0.0,
        'mixup': 0.0,
        'fliplr': 0.0,
        'flipud': 0.0,
        'degrees': 0.0,
        'translate': 0.05,
        'scale': 0.2,
        # Output
        'name': model_key,
        'project': str(results_dir),
        'exist_ok': True,
        'plots': True,
        'verbose': True,
    }
    
    # Train
    start_time = datetime.now()
    results = model.train(**train_params)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate on validation set
    print(f"\nEvaluating {config['name']}...")
    val_results = model.val(
        data=dataset_yaml,
        imgsz=args.imgsz,
        conf=0.25,
        verbose=False
    )
    
    # Compile results
    result_data = {
        'model': model_key,
        'name': config['name'],
        'mAP50': val_results.box.map50,
        'mAP50-95': val_results.box.map,
        'precision': val_results.box.mp,
        'recall': val_results.box.mr,
        'f1': 2 * val_results.box.mp * val_results.box.mr / max(val_results.box.mp + val_results.box.mr, 1e-8),
        'training_time_seconds': training_time,
        'epochs': args.epochs,
        'dataset': dataset_yaml,
    }
    
    # Save results
    with open(model_results_dir / 'results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ {config['name']} training complete!")
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall: {val_results.box.mr:.4f}")
    print(f"  F1: {result_data['f1']:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    
    return result_data, model


def compare_models(results, results_dir):
    """Generate comparison report and plots."""
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}\n")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Print comparison table
    print(df[['name', 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1', 'training_time_seconds']].to_string(index=False))
    
    # Find best model for each metric
    print(f"\n{'='*60}")
    print("BEST MODELS:")
    print(f"{'='*60}")
    
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
    for metric in metrics:
        best_idx = df[metric].idxmax()
        print(f"  Best {metric}: {df.loc[best_idx, 'name']} ({df.loc[best_idx, metric]:.4f})")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Metrics comparison
    metrics_to_plot = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        bars = ax.bar(df['name'], df[metric], color=['#2196F3', '#4CAF50', '#FF9800'])
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Training time comparison
    ax = axes[1, 2]
    bars = ax.bar(df['name'], df['training_time_seconds'], color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_title('TRAINING TIME (s)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, df['training_time_seconds']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{val:.0f}s', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('SSS Debris Detection: Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save comparison to CSV
    df.to_csv(results_dir / 'comparison_results.csv', index=False)
    
    print(f"\n✓ Comparison results saved to {results_dir}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Train SSS debris detection models')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (default: 30)')
    parser.add_argument('--models', nargs='+', default=['yolov8n', 'yolov8_esi', 'ss_yolo'],
                       choices=['yolov8n', 'yolov8_esi', 'ss_yolo'],
                       help='Models to train (default: all)')
    parser.add_argument('--output', type=str, default='results/sss_comparison',
                       help='Output directory (default: results/sss_comparison)')
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("⚠ No GPU available, training will be slow!")
    
    # Register custom modules
    register_custom_modules()
    
    # Create output directory
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train all models
    all_results = []
    models_trained = {}
    
    for model_key in args.models:
        try:
            result_data, model = train_model(
                model_key, 
                args.dataset, 
                args, 
                results_dir
            )
            all_results.append(result_data)
            models_trained[model_key] = model
        except Exception as e:
            print(f"\n✗ Error training {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare models
    if len(all_results) > 1:
        comparison_df = compare_models(all_results, results_dir)
        
        # Find overall best
        best_model_idx = comparison_df['f1'].idxmax()
        best_model = comparison_df.loc[best_model_idx]
        
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION")
        print(f"{'='*60}")
        print(f"Best overall model: {best_model['name']}")
        print(f"  F1 Score: {best_model['f1']:.4f}")
        print(f"  mAP50: {best_model['mAP50']:.4f}")
        print(f"  Use this model for deployment!")
    
    # Export best model to ONNX
    if models_trained:
        print(f"\n{'='*60}")
        print("EXPORTING MODELS")
        print(f"{'='*60}")
        
        for model_key, model in models_trained.items():
            try:
                export_path = results_dir / model_key / 'export'
                export_path.mkdir(exist_ok=True)
                
                model.export(format='onnx', imgsz=args.imgsz, save_dir=str(export_path))
                print(f"✓ {model_key} exported to ONNX")
            except Exception as e:
                print(f"✗ Failed to export {model_key}: {e}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"Comparison plot: {results_dir / 'model_comparison.png'}")
    print(f"Comparison CSV: {results_dir / 'comparison_results.csv'}")


if __name__ == '__main__':
    main()
