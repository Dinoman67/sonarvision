#!/usr/bin/env python3
"""
YOLOv8-ESI v6 ONNX Export Script
=================================
Exports trained v6 model weights (s2_best.pt) to FP32 and FP16 ONNX.
Pre-registers SEBlock and C2fWithSE to ensure seamless unpickling of custom layers.

Usage:
    python scripts/export_v6_onnx.py --weights ~/Downloads/s2_best.pt
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 1. Architecture definitions needed to unpickle custom YOLOv8-ESI weights
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class C2fWithSE(nn.Module):
    """C2f module augmented with Squeeze-and-Excitation."""
    def __init__(self, c2f_module, reduction=16):
        super().__init__()
        self.c2f = c2f_module
        self.se = SEBlock(c2f_module.cv2.conv.out_channels, reduction=reduction)
        self.i = c2f_module.i
        self.f = c2f_module.f
        self.type = 'C2fWithSE'

    def forward(self, x):
        return self.se(self.c2f(x))


# Inject into __main__ and ultralytics namespaces
sys.modules['__main__'].SEBlock = SEBlock
sys.modules['__main__'].C2fWithSE = C2fWithSE

try:
    import ultralytics.nn.modules.block as block_mod
    import ultralytics.nn.modules as mod_mod
    setattr(block_mod, 'SEBlock', SEBlock)
    setattr(block_mod, 'C2fWithSE', C2fWithSE)
    setattr(mod_mod, 'SEBlock', SEBlock)
    setattr(mod_mod, 'C2fWithSE', C2fWithSE)
except ImportError:
    pass

from ultralytics import YOLO


def export_v6(weights_path: Path, output_dir: Path, imgsz: int = 256):
    weights_path = Path(weights_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    print("=" * 60)
    print(f"Loading weights: {weights_path}")
    print(f"Target image size: {imgsz}x{imgsz}")
    print(f"Export directory: {output_dir}")
    print("=" * 60)

    model = YOLO(str(weights_path))

    exported_files = []
    for fmt, half in [('fp32', False), ('fp16', True)]:
        print(f"\nExporting ONNX ({fmt.upper()}, half={half})...")
        try:
            exported_path = model.export(
                format='onnx',
                imgsz=imgsz,
                simplify=True,
                dynamic=False,
                half=half
            )
            src = Path(exported_path)
            dest = output_dir / f"yolo_esi_v6_{fmt}.onnx"
            shutil.copy2(src, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ Successfully created: {dest} ({size_mb:.2f} MB)")
            exported_files.append(dest)
        except Exception as e:
            print(f"  ✗ Export failed for {fmt}: {e}")

    # Also copy to yolo_esi_fp16.onnx for backend drop-in
    fp16_target = output_dir / "yolo_esi_fp16.onnx"
    fp16_src = output_dir / "yolo_esi_v6_fp16.onnx"
    if fp16_src.exists():
        shutil.copy2(fp16_src, fp16_target)
        print(f"  ✓ Prepared backend drop-in: {fp16_target}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETED")
    for f in exported_files:
        print(f"  - {f.name} ({f.stat().st_size / (1024 * 1024):.2f} MB)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export YOLOv8-ESI v6 to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path.home() / "Downloads" / "s2_best.pt"),
        help="Path to trained PyTorch weights (e.g. s2_best.pt)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path.home() / "sonar-vision" / "models"),
        help="Directory to save exported ONNX files"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Image size (default: 256)"
    )
    args = parser.parse_args()
    export_v6(args.weights, args.out, args.imgsz)
