#!/usr/bin/env python3
"""Debris-feature diagnostic: does each model HAVE a debris feature on
genuinely unseen passes, or is it absent?

Method: v5 NOAA test (99 debris-positive frames, whole passes unseen by
the v5 model's train). Measure debris recall/AP at conf 0.001 -> 0.25.
  - If recall rises sharply at low conf  -> feature EXISTS but low-confidence
    (a calibration/deployment lever can recover it).
  - If recall stays ~0 even at conf 0.001 -> feature ABSENT for unseen passes.
Same probes on the Core_model (pass-SEEN for it) as the reference.
"""
from pathlib import Path
from ultralytics import YOLO

# --- ESI classes must exist in __main__ before unpickling the ESI checkpoint ---
import torch.nn as nn


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
OUT = ROOT / 'mock_runs'
NOAA_YAML = OUT / 'noaa_test_nc1/data.yaml'   # nc=1, v5 NOAA test (built read-only earlier)
UNSEEN_YAML = ROOT / 'datasets/noaa-debris/h8_unseen_test/data.yaml'

models = {
    'v5_4class_10ep':  str(OUT / 'mock_v5_s1/weights/best.pt'),
    'core_fp32':       str(ROOT / 'hf_yolo_esi_model/yolo_esi_fp32.onnx'),
}

confs = [0.001, 0.005, 0.01, 0.05, 0.25]

for tag, path in models.items():
    print(f'\n{"=" * 66}\nMODEL: {tag}\n{"=" * 66}')
    m = YOLO(path)
    for split_name, yaml_path in [('v5 NOAA test (UNSEEN passes for v5, SEEN for core)',
                                   NOAA_YAML),
                                  ('h8_unseen_test (SEEN passes for both)',
                                   UNSEEN_YAML)]:
        print(f'\n  --- {split_name} ---')
        print(f'  {"conf":>6} {"debrisAP":>9} {"P":>7} {"R":>7}  (recall = frac of GT debris found)')
        for conf in confs:
            r = m.val(data=str(yaml_path), imgsz=256, conf=conf, verbose=False, workers=0)
            # nc=1 yamls -> class 0 is debris in both
            print(f'  {conf:>6.3f} {r.box.ap50[0]:>9.4f} {r.box.mp:>7.4f} {r.box.mr:>7.4f}')
print('\nDONE')