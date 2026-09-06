#!/usr/bin/env python3
"""Evaluate the deployed Core_model (single-class debris ONNX) on:

1. h8_unseen_test  — its own claimed "unseen" test (pass-overlapping:
   every positive frame comes from a pass present in h8_data, which the
   Core_model trained on). Expect ~0.82 (reproduces the notebook).
2. v5 NOAA test    — whole passes held out of v5's train. NOTE: those
   passes WERE in the Core_model's training set (h8_data), so this is
   still a pass-SEEN eval for the Core_model. Contrasts with the v5
   4-class model, which scored 0.0005 here (passes truly unseen).
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path('/home/ashish/sonar-vision')
V5 = ROOT / 'datasets/sonarvision_multisource_v5'
UNSEEN = ROOT / 'datasets/noaa-debris/h8_unseen_test'
OUT = ROOT / 'mock_runs'
ONNX = '/home/ashish/sonar-vision/hf_yolo_esi_model/yolo_esi_fp32.onnx'

# 1) h8_unseen_test as-is (nc=1, names marine_debris)
print('MODEL: Core_model single-class fp32 ONNX')
m = YOLO(ONNX)

r = m.val(data=str(UNSEEN / 'data.yaml'), imgsz=256, conf=0.05, verbose=False, workers=0)
f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
print(f'\n[1] h8_unseen_test   (pass-SEEN for Core_model)')
print(f'    mAP50={r.box.map50:.4f}  P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')

# 2) v5 NOAA test subset (whole passes; labels are class 0 debris)
import shutil, os, csv
d = OUT / 'core_eval_noaa'
if d.exists():
    shutil.rmtree(d)
(d / 'images' / 'test').mkdir(parents=True)
(d / 'labels' / 'test').mkdir(parents=True)
rows = list(csv.DictReader(open(str(V5 / 'metadata.csv'))))
n_pos = 0
for r_ in rows:
    if r_['split'] == 'test' and r_['image_path'].split('/')[-1].startswith('NOAA_'):
        ip = V5 / r_['image_path']
        lp = Path(str(ip).replace('/images/', '/labels/').replace('.png', '.txt'))
        (d / 'images' / 'test' / ip.name).symlink_to(ip)
        (d / 'labels' / 'test' / lp.name).symlink_to(lp)
        with open(lp) as fh:
            if any(l.strip() for l in fh):
                n_pos += 1
(d / 'data.yaml').write_text(
    f"path: {d}\ntrain: images/test\nval: images/test\ntest: images/test\n"
    f"nc: 1\nnames:\n  0: unknown_debris\n")
r2 = m.val(data=str(d / 'data.yaml'), imgsz=256, conf=0.05, verbose=False, workers=0)
f12 = 2 * r2.box.mp * r2.box.mr / max(r2.box.mp + r2.box.mr, 1e-8)
n_imgs = len(os.listdir(d / 'images' / 'test'))
print(f'\n[2] v5 NOAA test     ({n_imgs} imgs, {n_pos} positive)')
print(f'    (pass-SEEN for Core_model: its train covered all h8_data passes)')
print(f'    mAP50={r2.box.map50:.4f}  P={r2.box.mp:.4f}  R={r2.box.mr:.4f}  F1={f12:.4f}')

print('\nCONTRAST — v5 4-class model (10-epoch mock):')
print('  same v5 NOAA test      debris AP50 = 0.0005  (passes truly unseen by it)')
print('  h8_unseen_test         debris AP50 = 0.3301  (passes seen in its train)')
print('\nDONE')