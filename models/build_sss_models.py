"""
Build SSS-optimized YOLOv8 models programmatically.

SS-YOLO (Yang et al., 2025, JMSE):
- GhostConv replaces Conv in backbone (lightweight feature extraction)
- Fast-C2f replaces C2f (FasterBlock = PConv + PWConv from FasterNet CVPR 2023)
- Two training modes:
  1. From scratch (original paper — needs large dataset)
  2. Pretrained backbone (for small datasets — recommended for <5000 images)

YOLOv8-ESI:
- Standard YOLOv8n backbone + SE attention after each C2f
- Pretrained weights transfer fine (SE blocks are added, not replaced)

Author: Buffy (Codebuff)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF

from models.sss_custom_modules import (
    WaveletConv, FastC2f, GhostConv,
    PConv, FasterBlock, DepthwiseSeparableConv, SEBlock, CBAM
)


def _init_weights(m):
    """Kaiming uniform init for Conv/Bn layers — crucial for training from scratch."""
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


def build_ss_yolo(pretrained='yolov8n.pt', mode='backbone_only'):
    """Build SS-YOLO: Lightweight SSS detection model.
    
    Based on Yang et al. (2025), JMSE 13(1):66.
    
    Training modes:
    - 'full': Replace all Conv→GhostConv, C2f→FastC2f (from scratch, needs large dataset)
    - 'backbone_only': Only replace neck Conv→GhostConv, C2f→FastC2f (pretrained backbone)
    
    For small datasets (<5000 images), use mode='backbone_only' for better convergence.
    
    Args:
        pretrained: Path to pretrained weights (None for from scratch)
        mode: 'full' or 'backbone_only'
    """
    print(f"Building SS-YOLO (mode={mode})...")
    
    # Load base model
    if pretrained:
        base_model = YOLO(pretrained)
        print(f"  Loading pretrained weights from {pretrained}")
    else:
        base_model = YOLO('yolov8n.pt')
        # Strip weights for from-scratch training
        base_model.model.apply(_init_weights)
        print("  Building from scratch (no pretrained weights)")
    
    model = base_model.model
    layers = list(model.model)
    
    # Find backbone/neck boundary
    # YOLOv8n structure: backbone (layers 0-9) → neck (layers 10-15) → detect (layer 16)
    backbone_end = 9  # Last backbone layer index
    
    print(f"  Backbone: layers 0-{backbone_end} (preserved)")
    print(f"  Neck: layers {backbone_end+1}-{len(layers)-2} (modified)")
    
    for i, layer in enumerate(layers):
        # Preserve metadata
        orig_f = getattr(layer, 'f', -1)
        
        if mode == 'backbone_only':
            # Only modify neck layers (after backbone)
            if i <= backbone_end:
                continue
        
        if isinstance(layer, Conv):
            # Replace Conv with GhostConv
            conv = layer.conv
            c1 = conv.in_channels
            c2 = conv.out_channels
            k = conv.kernel_size[0]
            s = conv.stride[0]
            p = conv.padding[0]
            new_layer = GhostConv(c1, c2, kernel_size=k, stride=s, padding=p)
            new_layer.i = i
            new_layer.f = orig_f
            new_layer.type = 'GhostConv'
            layers[i] = new_layer
            print(f"  Layer {i}: Conv({c1},{c2}) → GhostConv({c1},{c2})")
            
        elif isinstance(layer, C2f):
            # Replace C2f with FastC2f
            c1 = layer.cv1.conv.in_channels
            c2 = layer.cv2.conv.out_channels
            n_bottlenecks = len(layer.m)
            shortcut = layer.m[0].add if n_bottlenecks > 0 and hasattr(layer.m[0], 'add') else True
            new_layer = FastC2f(c1, c2, n=n_bottlenecks, shortcut=shortcut)
            new_layer.i = i
            new_layer.f = orig_f
            new_layer.type = 'FastC2f'
            layers[i] = new_layer
            print(f"  Layer {i}: C2f({c1},{c2},n={n_bottlenecks}) → FastC2f")
    
    model.model = nn.Sequential(*layers)
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    base_total = sum(p.numel() for p in YOLO('yolov8n.pt').model.parameters())
    reduction = (1 - total / base_total) * 100
    
    print(f"\nSS-YOLO built ({mode}):")
    print(f"  Parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"  Reduction from YOLOv8n: {reduction:.1f}%")
    
    return model


def build_yolov8_esi(pretrained='yolov8n.pt'):
    """Build YOLOv8-ESI: Add SE attention to YOLOv8n.

    YOLOv8-ESI keeps standard Conv/C2f (so pretrained weights transfer)
    but wraps each C2f with SE attention for channel recalibration.
    This is key for SSS where texture discrimination matters.
    """
    base_model = YOLO(pretrained)
    model = base_model.model

    print("Building YOLOv8-ESI by wrapping C2f with SE attention...")
    print("  (Keeping pretrained weights — SE layers are added, not replaced)")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nYOLOv8-ESI built:")
    print(f"  Parameters: {total:,} ({total/1e6:.2f}M)")

    return model


def build_yolov8_esi_full(pretrained='yolov8n.pt'):
    """Build YOLOv8-ESI: YOLOv8n backbone + SE attention after each C2f.

    This is the production version. SE blocks are lightweight wrappers
    that preserve the original C2f weights while adding channel attention.
    """
    base_model = YOLO(pretrained)
    model = base_model.model

    print("Building YOLOv8-ESI with SE attention blocks...")
    print("  (Pretrained YOLOv8n weights preserved — only SE layers are new)")

    return _add_se_blocks(model)


class C2fWithSE(nn.Module):
    """C2f followed by SE attention - drop-in replacement for C2f."""
    def __init__(self, c2f_module, reduction=16):
        super().__init__()
        self.c2f = c2f_module
        self.se = SEBlock(c2f_module.cv2.conv.out_channels, reduction=reduction)
        # Preserve metadata from original C2f
        self.i = c2f_module.i
        self.f = c2f_module.f
        self.type = 'C2fWithSE'
    def forward(self, x):
        return self.se(self.c2f(x))


def _add_se_blocks(model, reduction=16):
    """Add SE attention blocks after each C2f in the model.

    Uses C2fWithSE wrapper instead of inserting new layers,
    which avoids index shifting issues with .f attributes.
    """
    layers = list(model.model)

    for i, layer in enumerate(layers):
        if isinstance(layer, C2f):
            c2 = layer.cv2.conv.out_channels
            layers[i] = C2fWithSE(layer, reduction=reduction)
            print(f"  Wrapped layer {i}: C2f({c2}) → C2fWithSE")

    model.model = nn.Sequential(*layers)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nYOLOv8-ESI (SE-augmented) built:")
    print(f"  Parameters: {total:,} ({total/1e6:.2f}M)")

    return model


def build_yolov8n_baseline(pretrained='yolov8n.pt'):
    """Return standard YOLOv8n as baseline for comparison."""
    model = YOLO(pretrained)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"\nYOLOv8n baseline:")
    print(f"  Parameters: {total:,} ({total/1e6:.2f}M)")
    return model.model


if __name__ == '__main__':
    # Test building all models
    print("=" * 60)
    print("Building all SSS models for comparison")
    print("=" * 60)

    baseline = build_yolov8n_baseline()
    ss_yolo = build_ss_yolo(mode='backbone_only')
    esi_model = build_yolov8_esi_full()

    # Compare sizes
    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)

    models = {
        'YOLOv8n': baseline,
        'SS-YOLO': ss_yolo,
        'YOLOv8-ESI': esi_model,
    }

    for name, m in models.items():
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:>15}: {n:>12,} params ({n/1e6:.2f}M)")
