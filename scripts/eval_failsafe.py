#!/usr/bin/env python3
"""READ-ONLY test of the user's failsafe Core_model exports.

Only reads hf_yolo_esi_model/*.onnx. All outputs go to mock_runs/.
Evaluates fp32 / fp16 / int8 at imgsz 256 on:
  [A] h8_unseen_test (the model's own "unseen" test — pass-SEEN)
  [B] v5 NOAA test   (whole passes; pass-SEEN for this model too)
"""
from pathlib import Path
import os, csv, shutil

from ultralytics import YOLO

ROOT = Path('/home/ashish/sonar-vision')
V5 = ROOT / 'datasets/sonarvision_multisource_v5'
UNSEEN = ROOT / 'datasets/noaa-debris/h8_unseen_test'
OUT = ROOT / 'mock_runs'
HF = ROOT / 'hf_yolo_esi_model'

# --- build v5-NOAA nc=1 test subset once (scratch dir only) ---
d = OUT / 'noaa_test_nc1'
if not (d / 'data.yaml').exists():
    if d.exists():
        shutil.rmtree(d)
    (d / 'images' / 'test').mkdir(parents=True)
    (d / 'labels' / 'test').mkdir(parents=True)
    rows = list(csv.DictReader(open(str(V5 / 'metadata.csv'))))
    for r_ in rows:
        if r_['split'] == 'test' and r_['image_path'].split('/')[-1].startswith('NOAA_'):
            ip = V5 / r_['image_path']
            lp = Path(str(ip).replace('/images/', '/labels/').replace('.png', '.txt'))
            (d / 'images' / 'test' / ip.name).symlink_to(ip)
            (d / 'labels' / 'test' / lp.name).symlink_to(lp)
    n_pos = sum(1 for f in os.listdir(d / 'labels' / 'test')
                if os.path.getsize(d / 'labels' / 'test' / f) > 0)
    (d / 'data.yaml').write_text(
        f"path: {d}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: 1\nnames:\n  0: unknown_debris\n")
    print(f'built {d}: {len(os.listdir(d / "images" / "test"))} imgs, {n_pos} positive')

tests = {
    'h8_unseen_test': str(UNSEEN / 'data.yaml'),
    'v5_NOAA_test':   str(d / 'data.yaml'),
}

print(f'\nMODEL FILES (read-only):')
for f in sorted(os.listdir(HF)):
    if f.endswith('.onnx'):
        print(f'  {f}  {os.path.getsize(HF / f) / 1e6:.2f} MB')

for fmt in ['fp32', 'fp16', 'int8']:
    path = HF / f'yolo_esi_{fmt}.onnx'
    print(f'\n{"=" * 60}\nEXPORT: {fmt}\n{"=" * 60}')
    try:
        m = YOLO(str(path))
    except Exception as e:
        print(f'  ✗ load failed: {e}')
        continue
    for name, yaml_path in tests.items():
        try:
            r = m.val(data=yaml_path, imgsz=256, conf=0.05, verbose=False, workers=0)
            f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
            print(f'  [{name:15s}] mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
                  f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
        except Exception as e:
            print(f'  [{name:15s}] ✗ eval failed: {str(e)[:120]}')

print('\nDONE — no model files were written or modified.')