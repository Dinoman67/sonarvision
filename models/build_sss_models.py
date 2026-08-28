"""
Build SSS-optimized YOLOv8 models — 6 variants for comparison.

Model Variants:
  A: YOLOv8n baseline (standard pretrained)
  B: YOLOv8s baseline (larger pretrained)
  C: SS-YOLO reproduction (from scratch — Yang et al., 2025)
  D: SS-YOLO + pretrained backbone (recommended for small datasets)
  E: SS-YOLO + EIS-inspired modules (WaveletConv + LocalContrast)
  F: SS-YOLO + pretrained backbone + EIS-inspired modules

SS-YOLO Architecture (Yang et al., 2025, JMSE 13(1):66):
  - Backbone: YOLOv8n with GhostConv replacing standard Conv
  - Neck/Head: Fast-C2f replacing standard C2f
  - Fast-C2f: FasterBlock = PConv + PWConv (from FasterNet, CVPR 2023)
  - PConv: Only convolves 1/4 of channels, leaves rest untouched

YOLOv8-ESI:
  - Standard YOLOv8n + SE attention after each C2f
  - Pretrained weights transfer fine (SE blocks are added, not replaced)

Author: Buffy (Codebuff)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF

from models.sss_custom_modules import (
    PConv, FasterBlock, FastC2f, GhostConv,
    SEBlock, CBAM, WaveletConv, LocalContrastEnhance
)


# ══════════════════════════════════════════════════════════════════════════════
# Weight Initialization
# ══════════════════════════════════════════════════════════════════════════════


def _init_weights(m):
    """Kaiming uniform init for Conv/Bn — crucial for from-scratch training."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _count_params(model):
    """Count total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


# ══════════════════════════════════════════════════════════════════════════════
# Model A: YOLOv8n Baseline
# ══════════════════════════════════════════════════════════════════════════════


def build_model_a(pretrained='yolov8n.pt'):
    """Model A: Standard YOLOv8n baseline for comparison.

    Uses COCO pretrained weights. No modifications.
    Training: Fine-tune with frozen backbone recommended.
    """
    base = YOLO(pretrained)
    print(f"Model A (YOLOv8n): {_count_params(base.model):.2f}M params")
    return base.model


# ══════════════════════════════════════════════════════════════════════════════
# Model B: YOLOv8s Baseline
# ══════════════════════════════════════════════════════════════════════════════


def build_model_b(pretrained='yolov8s.pt'):
    """Model B: Larger YOLOv8s baseline for comparison.

    Uses COCO pretrained weights. More parameters but potentially better accuracy.
    """
    base = YOLO(pretrained)
    print(f"Model B (YOLOv8s): {_count_params(base.model):.2f}M params")
    return base.model


# ══════════════════════════════════════════════════════════════════════════════
# Model C: SS-YOLO From Scratch (Original Paper)
# ══════════════════════════════════════════════════════════════════════════════


def _apply_ss_yolo_full(model):
    """Replace all Conv→GhostConv, C2f→FastC2f (full replacement)."""
    layers = list(model.model)

    for i, layer in enumerate(layers):
        orig_f = getattr(layer, 'f', -1)

        if isinstance(layer, Conv):
            conv = layer.conv
            c1, c2 = conv.in_channels, conv.out_channels
            k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
            new_layer = GhostConv(c1, c2, kernel_size=k, stride=s, padding=p)
            new_layer.i, new_layer.f, new_layer.type = i, orig_f, 'GhostConv'
            layers[i] = new_layer

        elif isinstance(layer, C2f):
            c1 = layer.cv1.conv.in_channels
            c2 = layer.cv2.conv.out_channels
            n = len(layer.m)
            shortcut = layer.m[0].add if n > 0 and hasattr(layer.m[0], 'add') else True
            new_layer = FastC2f(c1, c2, n=n, shortcut=shortcut)
            new_layer.i, new_layer.f, new_layer.type = i, orig_f, 'FastC2f'
            layers[i] = new_layer

    model.model = nn.Sequential(*layers)
    return model


def build_model_c():
    """Model C: SS-YOLO from scratch (original paper approach).

    Full replacement: Conv→GhostConv, C2f→FastC2f
    Weights: Random init with Kaiming uniform
    Training: Needs 200+ epochs and large dataset (>5000 images)
    Warning: Will fail with small datasets (<1000 images)
    """
    base = YOLO('yolov8n.pt')
    base.model.apply(_init_weights)
    model = _apply_ss_yolo_full(base)
    print(f"Model C (SS-YOLO scratch): {_count_params(model):.2f}M params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model D: SS-YOLO + Pretrained Backbone
# ══════════════════════════════════════════════════════════════════════════════


def _apply_ss_yolo_neck_only(model):
    """Replace only neck Conv→GhostConv, C2f→FastC2f (keep pretrained backbone)."""
    layers = list(model.model)
    backbone_end = 9  # Last backbone layer index in YOLOv8n

    for i, layer in enumerate(layers):
        if i <= backbone_end:
            continue  # Keep backbone intact

        orig_f = getattr(layer, 'f', -1)

        if isinstance(layer, Conv):
            conv = layer.conv
            c1, c2 = conv.in_channels, conv.out_channels
            k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
            new_layer = GhostConv(c1, c2, kernel_size=k, stride=s, padding=p)
            new_layer.i, new_layer.f, new_layer.type = i, orig_f, 'GhostConv'
            layers[i] = new_layer

        elif isinstance(layer, C2f):
            c1 = layer.cv1.conv.in_channels
            c2 = layer.cv2.conv.out_channels
            n = len(layer.m)
            shortcut = layer.m[0].add if n > 0 and hasattr(layer.m[0], 'add') else True
            new_layer = FastC2f(c1, c2, n=n, shortcut=shortcut)
            new_layer.i, new_layer.f, new_layer.type = i, orig_f, 'FastC2f'
            layers[i] = new_layer

    model.model = nn.Sequential(*layers)
    return model


def build_model_d(pretrained='yolov8n.pt'):
    """Model D: SS-YOLO with pretrained backbone (recommended for small datasets).

    Backbone: YOLOv8n pretrained on COCO (frozen or fine-tuned)
    Neck: GhostConv + FastC2f (SS-YOLO lightweight design)
    Training: 100-150 epochs, lr=0.001, freeze backbone first 10 layers
    """
    base = YOLO(pretrained)
    model = _apply_ss_yolo_neck_only(base)
    print(f"Model D (SS-YOLO pretrained): {_count_params(model):.2f}M params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model E: SS-YOLO + EIS-Inspired Modules
# ══════════════════════════════════════════════════════════════════════════════


def build_model_e():
    """Model E: SS-YOLO from scratch + EIS-inspired WaveletConv.

    Adds frequency-domain feature extraction to SS-YOLO.
    WaveletConv helps distinguish debris (broadband) from seabed (narrowband).
    Warning: Same from-scratch limitations as Model C.
    """
    base = YOLO('yolov8n.pt')
    base.model.apply(_init_weights)
    model = _apply_ss_yolo_full(base)

    # Add WaveletConv after first backbone layer
    layers = list(model.model)
    if isinstance(layers[0], GhostConv):
        wavelet = WaveletConv(3, 16)
        layers.insert(1, wavelet)
        # Update indices
        for i, layer in enumerate(layers):
            layer.i = i
            if hasattr(layer, 'f') and layer.f > 0:
                layer.f += 1
    model.model = nn.Sequential(*layers)

    print(f"Model E (SS-YOLO+EIS scratch): {_count_params(model):.2f}M params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model F: SS-YOLO + Pretrained + EIS-Inspired
# ══════════════════════════════════════════════════════════════════════════════


def build_model_f(pretrained='yolov8n.pt'):
    """Model F: SS-YOLO pretrained backbone + EIS-inspired modules.

    Combines best of both worlds:
    - Pretrained backbone for robust feature extraction
    - GhostConv + FastC2f for lightweight neck
    - WaveletConv + LocalContrast for SSS-specific enhancement
    Training: 100-150 epochs, lr=0.001, freeze backbone first 10 layers
    """
    base = YOLO(pretrained)
    model = _apply_ss_yolo_neck_only(base)

    # Add EIS modules after neck layers
    layers = list(model.model)
    insertions = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'type') and layer.type in ('FastC2f', 'C2f'):
            # Add LocalContrast after C2f/FastC2f in neck
            channels = layer.cv2.conv.out_channels if hasattr(layer, 'cv2') else 64
            contrast = LocalContrastEnhance(channels)
            contrast.i = len(layers) + len(insertions)
            contrast.f = i
            contrast.type = 'LocalContrast'
            insertions.append((i + 1, contrast))

    # Insert in reverse order to maintain correct indices
    for idx, module in reversed(insertions):
        layers.insert(idx, module)

    model.model = nn.Sequential(*layers)
    print(f"Model F (SS-YOLO pretrained+EIS): {_count_params(model):.2f}M params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Utility: Build any model by name
# ══════════════════════════════════════════════════════════════════════════════


MODEL_BUILDERS = {
    'A': ('YOLOv8n baseline', build_model_a),
    'B': ('YOLOv8s baseline', build_model_b),
    'C': ('SS-YOLO scratch', build_model_c),
    'D': ('SS-YOLO pretrained', build_model_d),
    'E': ('SS-YOLO+EIS scratch', build_model_e),
    'F': ('SS-YOLO pretrained+EIS', build_model_f),
}


def build_model(variant, **kwargs):
    """Build a model variant by letter (A-F).

    Args:
        variant: 'A', 'B', 'C', 'D', 'E', or 'F'
        **kwargs: Passed to builder (e.g., pretrained='yolov8n.pt')
    """
    variant = variant.upper()
    if variant not in MODEL_BUILDERS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(MODEL_BUILDERS.keys())}")

    name, builder = MODEL_BUILDERS[variant]
    print(f"\n{'='*60}")
    print(f"Building Model {variant}: {name}")
    print(f"{'='*60}")

    model = builder(**kwargs)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Comparison Table
# ══════════════════════════════════════════════════════════════════════════════


def print_model_comparison():
    """Print comparison table of all model variants."""
    print(f"\n{'='*80}")
    print("SSS DEBRIS DETECTION — MODEL VARIANTS")
    print(f"{'='*80}")
    print(f"{'Variant':<10} {'Name':<30} {'Backbone':<12} {'Train':<10}")
    print(f"{'-'*80}")
    print(f"{'A':<10} {'YOLOv8n baseline':<30} {'COCO':<12} {'Finetune':<10}")
    print(f"{'B':<10} {'YOLOv8s baseline':<30} {'COCO':<12} {'Finetune':<10}")
    print(f"{'C':<10} {'SS-YOLO scratch':<30} {'Random':<12} {'Scratch':<10}")
    print(f"{'D':<10} {'SS-YOLO pretrained':<30} {'COCO':<12} {'Finetune':<10}")
    print(f"{'E':<10} {'SS-YOLO+EIS scratch':<30} {'Random':<12} {'Scratch':<10}")
    print(f"{'F':<10} {'SS-YOLO pretrained+EIS':<30} {'COCO':<12} {'Finetune':<10}")
    print(f"{'='*80}")
    print("\nKey differences:")
    print("  A/B: Standard YOLO — easy to train, good transfer from COCO")
    print("  C/E: From scratch — needs large dataset (>5000 images)")
    print("  D/F: Pretrained backbone + lightweight neck — best for small datasets")
    print("  E/F: Add WaveletConv + LocalContrast for SSS frequency features")


if __name__ == '__main__':
    print_model_comparison()
    print()

    # Build and compare all models
    for variant in ['A', 'B', 'C', 'D', 'E', 'F']:
        try:
            model = build_model(variant)
            params = _count_params(model)
            print(f"  Model {variant}: {params:.2f}M params ✓\n")
        except Exception as e:
            print(f"  Model {variant}: FAILED — {e}\n")
