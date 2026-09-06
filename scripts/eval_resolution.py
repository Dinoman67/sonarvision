#!/usr/bin/env python3
"""Eval trained mock models at multiple INFERENCE resolutions.

Isolates the resolution question from training dynamics: the same weights,
evaluated at 256/384/512 px. If debris stays ~0 everywhere, more pixels
were never the missing ingredient; if it jumps with imgsz, resolution is
the lever.

Models:
  mock_v5_s1   = trained @ imgsz 256, batch 16
  mock_v5_s512 = trained @ imgsz 512, batch 8  (LR/opt confounded vs s1)
"""
import argparse
from pathlib import Path
import torch, torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f


# Required so the ESI checkpoint can be unpickled (classes must exist in
# __main__). Identical definitions to the training script.
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class C2fWithSE(nn.Module):
    def __init__(self, c2f_module, reduction=16):
        super().__init__()
        self.c2f = c2f_module
        self.se = SEBlock(c2f_module.cv2.conv.out_channels, reduction=reduction)
        self.i = c2f_module.i
        self.f = c2f_module.f
        self.type = 'C2fWithSE'
    def forward(self, x):
        return self.se(self.c2f(x))

ROOT = Path('/home/ashish/sonar-vision')
DATA_YAML = ROOT / 'datasets/sonarvision_multisource_v5/dataset.yaml'
UNSEEN = ROOT / 'datasets/noaa-debris/h8_unseen_test'
OUT = ROOT / 'mock_runs'

_p = argparse.ArgumentParser()
_p.add_argument('--name', default='mock_v5_s1')
_p.add_argument('--sizes', default='256,384,512')
_a = _p.parse_args()
sizes = [int(x) for x in _a.sizes.split(',')]

unseen_yaml = OUT / 'h8_unseen_v5.yaml'
unseen_yaml.write_text(
    f"path: {UNSEEN}\ntrain: images/test\nval: images/test\ntest: images/test\n"
    f"nc: 4\nnames:\n  0: unknown_debris\n  1: airplane\n  2: mine\n  3: wreck\n")

best = OUT / f'{_a.name}/weights/best.pt'
print(f'MODEL: {_a.name}  ({best})')
m = YOLO(str(best))

for imgsz in sizes:
    print(f'\n=== inference imgsz {imgsz} ===')
    r = m.val(data=str(DATA_YAML), imgsz=imgsz, conf=0.25, split='test', verbose=False, workers=0)
    print(f'  HONEST TEST   mAP50={r.box.map50:.4f}  P={r.box.mp:.4f}  R={r.box.mr:.4f}'
          f'  | debris={r.box.ap50[0]:.4f} airplane={r.box.ap50[1]:.4f}'
          f' mine={r.box.ap50[2]:.4f} wreck={r.box.ap50[3]:.4f}')
    r2 = m.val(data=str(unseen_yaml), imgsz=imgsz, conf=0.05, verbose=False, workers=0)
    print(f'  h8_UNSEEN     debris={r2.box.ap50[0]:.4f}  P={r2.box.mp:.4f}  R={r2.box.mr:.4f}')
print('\nDONE')