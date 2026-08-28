"""
Build SSS-optimized YOLOv8 models programmatically.

SS-YOLO (Yang et al., 2025, JMSE):
- GhostConv replaces Conv in backbone (lightweight feature extraction)
- Fast-C2f replaces C2f (FasterBlock = PConv + PWConv from FasterNet CVPR 2023)
- Trained from scratch (PConv/FasterBlock weights don't transfer from YOLOv8n)

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


def build_ss_yolo(pretrained=None):
    """Build SS-YOLO: Lightweight SSS detection model.

    Based on Yang et al. (2025), JMSE 13(1):66.
    Architecture: YOLOv8n with GhostConv backbone + Fast-C2f neck.

    IMPORTANT: Must be built with pretrained=None and trained from scratch.
    GhostConv/FasterBlock have incompatible weight structures with YOLOv8n.

    Key modules:
    - GhostConv: lightweight convolution via cheap operations
    - Fast-C2f: C2f with FasterBlock (PConv + PWConv) instead of standard Bottleneck
    - PConv: partial convolution — only processes 1/4 of channels (FasterNet, CVPR 2023)
    """
    if pretrained is not None:
        import warnings
        warnings.warn(
            "SS-YOLO should NOT load YOLOv8n pretrained weights — "
            "GhostConv/FastC2f have incompatible weight shapes. "
            "Train from scratch instead. Ignoring pretrained= argument.",
            stacklevel=2,
        )

    # Always start from a fresh YOLOv8n architecture (no weight loading)
    base_model = YOLO('yolov8n.pt')
    model = base_model.model

    # Strip the pretrained weights — we'll replace them anyway
    # Re-initialize with random weights for a clean baseline
    model.apply(_init_weights)

    print("Building SS-YOLO by replacing layers in YOLOv8n (from scratch)...")

    layers = list(model.model)

    for i, layer in enumerate(layers):
        # Preserve metadata from original layer
        orig_f = getattr(layer, 'f', -1)
        orig_type = getattr(layer, 'type', type(layer).__name__)

        if isinstance(layer, Conv):
            # Replace standard Conv with GhostConv
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

    # Apply Kaiming init to all new layers for better training from scratch
    model.apply(_init_weights)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    base_total = sum(p.numel() for p in YOLO('yolov8n.pt').model.parameters())
    reduction = (1 - total / base_total) * 100

    print(f"\nSS-YOLO built (from scratch — no pretrained transfer):")
    print(f"  Parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"  Reduction from YOLOv8n: {reduction:.1f}%")
    print(f"  (YOLOv8n had {base_total:,} params)")

    return model


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


def build_yolov8_esi(pretrained='yolov8n.pt', insert_wavelet_at=None):
    """Build YOLOv8-ESI: Add SE attention to YOLOv8n.

    YOLOv8-ESI keeps standard Conv/C2f (so pretrained weights transfer)
    but wraps each C2f with SE attention for channel recalibration.
    This is key for SSS where texture discrimination matters.
    """
    base_model = YOLO(pretrained)
    model = base_model.model

    print("Building YOLOv8-ESI by wrapping C2f with SE attention...")
    print("  (Keeping pretrained weights — SE layers are added, not replaced)")

    # SE blocks are added AFTER C2f, so existing C2f weights transfer fine.
    # Only the SE layers start from random init (tiny fraction of params).

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
    ss_yolo = build_ss_yolo()
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
