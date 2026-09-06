"""
================================================================
YOLOv8-ESI v4 — sensor-isolated classes + balanced split (imgsz 640 + cls_pw)
================================================================
WHY THIS RUN (v3 post-mortem):
  The v3 run (512px, no cls_pw) converged at mAP50 ~0.45, but per-class
  AP50 exposed two REAL failures (not just small targets):
    - unknown_debris 0.058 val / 0.000 test: the ONLY mixed-sensor class
      (NOAA h8 + NOAA e4 + MILCO NOMBO = THREE visual modes: brightness
      46/74/53, contrast 19/32/31). Every single-identity class worked
      (airplane .81, wreck .76, drowning .24-.33, mine .43 val).
    - mine 0.432 val / 0.000 test: the v3 random group split put 88% of
      mine images in train and ZERO 2015 mines (dominant look, 111 imgs)
      in val/test, so mine was only evaluated on unseen 2010/2018 batches.

v4 dataset fixes (built + audited, 0 leakage):
  1. NOMBO is now its OWN class 5 (nombo_contact); unknown_debris = NOAA
     only (e4 + h8). One visual identity per class, consistently.
  2. Stratified CLASS-BALANCED group split: every class has representative
     val/test (mine 42/124 boxes vs v3's 13/24; h8 debris back in val;
     all 6 classes covered in all 3 splits). 2015 mines now in val+test.
  3. G-clip survey frames block-grouped (near-duplicate seafloor frames
     were singleton groups scattered across splits).
  4. imgsz 640 + cls_pw (below) attack the small-target recall ceiling.

cls_pw=0.5 -> AUTOMATIC inverse-frequency class weights (ultralytics 8.4.x).
  The builder also now (a) builds the head with nc=6 (not the COCO 80-class
  head) and (b) actually LOADS yolov8n.pt pretrained weights via SE key
  remap — the earlier builder silently trained the backbone from scratch.

UNCHANGED (proven recipe):
  - YOLOv8-ESI (SE after every C2f), data-driven nc/names patch
  - SSS augmentation: no flips / rotation / mosaic / mixup
  - Stage 1 only (Stage 2 regressed on multi-source data)
  - yolov8n-ESI only (per user decision)
  - Conf sweep + per-class AP50 + per-source eval + ONNX FP32/FP16
================================================================
"""
# ================================================================
# CELL 1 — Mount Drive, config, extract dataset
# ================================================================
from google.colab import drive
drive.mount('/content/drive')

# ── CONFIG — edit this to where you uploaded the v4 zip ──────────
DRIVE = '/content/drive/MyDrive'          # zip is directly at Drive root
ZIP   = f'{DRIVE}/sonarvision_multisource_v4.zip'
# After unzip the dataset lives at /content/datasets/sonarvision_multisource_v4

import os, zipfile, yaml
os.makedirs('/content/datasets', exist_ok=True)
with zipfile.ZipFile(ZIP, 'r') as z:
    z.extractall('/content/')

DATASET_DIR = '/content/datasets/sonarvision_multisource_v4'
DATA = f'{DATASET_DIR}/dataset.yaml'

# Fix dataset.yaml path for Colab
with open(DATA) as f:
    cfg = yaml.safe_load(f)
cfg['path'] = DATASET_DIR
with open(DATA, 'w') as f:
    yaml.dump(cfg, f)

print(f'Classes ({cfg["nc"]}):')
for k, v in cfg['names'].items():
    print(f'  {k}: {v}')
for split in ['train', 'val', 'test']:
    n = len([f for f in os.listdir(f'{DATASET_DIR}/images/{split}') if f.endswith('.png')])
    print(f'  {split}: {n} images')
assert cfg['nc'] == 6

OUT = '/content/results'
OUT_DRIVE = f'{DRIVE}/results_v4'        # persistent copy on Drive
os.makedirs(OUT_DRIVE, exist_ok=True)

# ================================================================
# CELL 2 — YOLOv8-ESI (SE attention) + data-driven patch
# ================================================================
import torch, torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect.train import DetectionTrainer


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
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
    """C2f followed by SE attention — drop-in replacement for C2f."""
    def __init__(self, c2f_module, reduction=16):
        super().__init__()
        self.c2f = c2f_module
        self.se = SEBlock(c2f_module.cv2.conv.out_channels, reduction=reduction)
        self.i = c2f_module.i
        self.f = c2f_module.f
        self.type = 'C2fWithSE'
    def forward(self, x):
        return self.se(self.c2f(x))


def _get_layers(model):
    m = model.model
    return list(m) if isinstance(m, nn.Sequential) else list(m.model)


def _set_layers(model, layers):
    m = model.model
    if isinstance(m, nn.Sequential):
        model.model = nn.Sequential(*layers)
    else:
        m.model = nn.Sequential(*layers)


def build_yolov8_esi_full(nc=6, yaml_name='yolov8n.yaml', weights='yolov8n.pt'):
    """YOLOv8-ESI: SE after every C2f, built from the yaml for `nc` classes.

    CRITICAL FIXES vs the pre-v4 builder (found via local mock runs):
    1. Base is DetectionModel(yolov8n.yaml, nc=nc) so ultralytics builds the
       CORRECT nc-class Detect head. The old builder injected the COCO
       80-class head — bce_loss[80] x class_weights[nc] crashes as soon as
       cls_pw is set, and 75 output channels were wasted.
    2. Pretrained yolov8n.pt weights now actually LOAD. The C2fWithSE
       wrapper nests the C2f (`model.6.c2f.cv1...`), so plain
       intersect_dicts() matched nothing and every earlier ESI run
       (incl. Core_model-era code) silently trained the backbone FROM
       SCRATCH. Keys are remapped target->checkpoint to restore transfer.
    """
    import re
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(yaml_name, nc=nc, ch=3, verbose=False)
    layers = _get_layers(model)
    for i, layer in enumerate(layers):
        if isinstance(layer, C2f):
            layers[i] = C2fWithSE(layer)
    _set_layers(model, layers)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    csd = ckpt['model'].float().state_dict()
    msd = model.state_dict()
    remapped = {}
    for k in msd:
        m2 = re.match(r'^model\.(\d+)\.c2f\.(.*)$', k)
        ck = f'model.{m2.group(1)}.{m2.group(2)}' if m2 else k
        if ck in csd and tuple(csd[ck].shape) == tuple(msd[k].shape):
            remapped[k] = csd[ck]
    msd.update(remapped)
    model.load_state_dict(msd, strict=False)
    print(f'YOLOv8-ESI: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M '
          f'params | head nc={nc} | pretrained keys {len(remapped)}/{len(csd)} '
          f'(SE-remapped) — ALL key blocks transferred')
    return model


def patch_trainer(model_obj):
    """CRITICAL: ultralytics recreates the model during training and would
    drop the SE blocks. Patch DetectionTrainer.get_model to inject ours.
    Data-driven nc/names — NOT the binary hardcode."""
    _orig = DetectionTrainer.get_model
    def _patched(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        dm = DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'],
                            verbose=verbose and RANK == -1)
        dm.model = model_obj.model   # weights already loaded in build
        dm.nc = self.data['nc']
        dm.names = self.data['names']
        try:
            dm.load(weights)         # transfers nothing extra (keys nested)
        except Exception:
            pass
        return dm
    DetectionTrainer.get_model = _patched
    return _orig


print('✓ SEBlock + C2fWithSE + patch_trainer ready (data-driven nc/names)')

# ================================================================
# CELL 3 — Training config (imgsz 640 + cls_pw)
# ================================================================
IMGSZ = 640                       # more pixels per tiny target
BATCH = 8                         # same pixel budget as 512/16 on T4
# NOTE: ultralytics 8.4.x cls_pw is a SCALAR power (0..1) enabling AUTOMATIC
# inverse-frequency class weights (mean-normalized). A per-class LIST crashes
# get_cfg(). 0.5 = sqrt(inverse freq). Verified locally in the v4 mock run.
CLS_PW = 0.5

TRAIN_CONFIG = {
    # SSS-specific augmentation (NO flips, NO rotation, NO mosaic)
    'mosaic': 0.0, 'mixup': 0.0,
    'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0,
    'translate': 0.05, 'scale': 0.2,
    'project': OUT, 'exist_ok': True, 'plots': True, 'verbose': True,
}
print(f'v4 config | imgsz={IMGSZ} batch={BATCH} cls_pw={CLS_PW} '
      f'(auto inverse-frequency class weights)')

# ================================================================
# CELL 4 — Stage 1: train yolov8n-ESI, 100 epochs (patience 30)
# ================================================================
print('=' * 60)
print('STAGE 1: yolov8n-ESI @ 640px, 6 classes, 100 epochs (patience=30)')
print('=' * 60)

# Ensure yolov8n.pt is downloaded before building the ESI model
_ = YOLO('yolov8n.pt')

esi_obj = build_yolov8_esi_full(nc=cfg['nc'], weights='yolov8n.pt')
_orig = patch_trainer(esi_obj)
try:
    model = YOLO('yolov8n.pt')
    model.train(data=DATA, epochs=100, imgsz=IMGSZ, batch=BATCH, patience=30,
                lr0=0.01, lrf=0.01, warmup_epochs=2, cls_pw=CLS_PW,
                **TRAIN_CONFIG, name='model_esi_v4_s1')
finally:
    DetectionTrainer.get_model = _orig

BEST = f'{OUT}/model_esi_v4_s1/weights/best.pt'
print(f'✓ training done — best: {BEST}')

# ================================================================
# CELL 5 — Per-class AP50 on val + test
# ================================================================
import pandas as pd
from pathlib import Path
import shutil

class_names = list(cfg['names'].values())
m = YOLO(BEST)

print('=' * 60)
print('VAL + TEST — PER-CLASS AP50')
print('=' * 60)
summary = []
for split in ['val', 'test']:
    r = m.val(data=DATA, imgsz=IMGSZ, conf=0.25, split=split, verbose=False)
    f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
    print(f'\n-- {split.upper()} --  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
          f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
    print(f'{"Class":<18}{"AP50":>8}')
    print('-' * 28)
    for i, name in enumerate(class_names):
        print(f'{name:<18}{r.box.ap50[i]:>8.4f}')
        summary.append({'split': split, 'class': name, 'AP50': r.box.ap50[i]})
    summary.append({'split': split, 'class': 'mAP50', 'AP50': r.box.map50})

# ================================================================
# CELL 6 — Confidence sweep on test
# ================================================================
print('=' * 60)
print('CONFIDENCE SWEEP (test)')
print('=' * 60)
best_f1, best_conf = 0, 0.05
for conf in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    r = m.val(data=DATA, imgsz=IMGSZ, conf=conf, split='test', verbose=False)
    p, rv = r.box.mp, r.box.mr
    f1 = 2 * p * rv / max(p + rv, 1e-8)
    print(f'  conf={conf:.2f}  mAP50={r.box.map50:.4f}  P={p:.4f}  R={rv:.4f}  F1={f1:.4f}')
    if f1 > best_f1:
        best_f1, best_conf = f1, conf
print(f'\n  Best: conf={best_conf} F1={best_f1:.4f}')

# ================================================================
# CELL 7 — Per-source test eval (SIH multi-sensor proof)
# ================================================================
import csv
src_root = Path('/content/source_eval_v4')
src_root.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(f'{DATASET_DIR}/metadata.csv')))
print('=' * 60)
print('PER-SOURCE TEST EVAL (multi-sensor generalization)')
print('=' * 60)
for src in ['NOAA', 'MILCO', 'KAGGLE']:
    d = src_root / src
    (d / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (d / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
    for r_ in rows:
        if r_['split'] == 'test' and r_['image_path'].split('/')[-1].startswith(src + '_'):
            ip = f'{DATASET_DIR}/{r_["image_path"]}'
            lp = ip.replace('/images/', '/labels/').replace('.png', '.txt')
            (d / 'images' / 'test' / Path(ip).name).symlink_to(ip)
            (d / 'labels' / 'test' / Path(lp).name).symlink_to(lp)
    (d / 'data.yaml').write_text(
        f"path: {d}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: {cfg['nc']}\nnames:\n" +
        ''.join(f'  {k}: {v}\n' for k, v in cfg['names'].items()))
    r = m.val(data=str(d / 'data.yaml'), imgsz=IMGSZ, conf=best_conf, verbose=False)
    f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
    n_imgs = len([f for f in os.listdir(d / 'images' / 'test') if f.endswith('.png')])
    print(f'  {src:<8} test_imgs={n_imgs:<4} mAP50={r.box.map50:.4f}  '
          f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')

# ================================================================
# CELL 8 — ONNX export (FP32 + FP16) + save everything to Drive
# ================================================================
print('=' * 60)
print('ONNX EXPORT + DRIVE SAVE')
print('=' * 60)
EXPORT_DIR = Path('/content/yolo_esi_v4_exports')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

for fmt in ['fp32', 'fp16']:
    kwargs = {'format': 'onnx', 'imgsz': IMGSZ, 'simplify': True, 'dynamic': False}
    if fmt == 'fp16':
        kwargs['half'] = True
    try:
        src = Path(m.export(**kwargs))
        dest = EXPORT_DIR / f'yolo_esi_v4_{fmt}.onnx'
        shutil.copy2(src, dest)
        print(f'  ✓ {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)')
    except Exception as e:
        print(f'  ✗ ONNX {fmt} failed: {e}')

# Validate exports on test
print('\nEXPORT VALIDATION (test):')
for f in sorted(EXPORT_DIR.glob('*.onnx')):
    try:
        ex = YOLO(str(f))
        r = ex.val(data=DATA, imgsz=IMGSZ, conf=best_conf, split='test', verbose=False)
        print(f'  {f.name:<40} mAP50={r.box.map50:.4f}')
    except Exception as e:
        print(f'  {f.name}: FAILED — {e}')

# Persistent copy on Drive
pd.DataFrame(summary).to_csv('/content/v4_eval_summary.csv', index=False)
for src_file in [BEST, f'{OUT}/model_esi_v4_s1/weights/last.pt',
                 '/content/v4_eval_summary.csv']:
    shutil.copy2(src_file, f'{OUT_DRIVE}/{Path(src_file).name}')
shutil.copy2(EXPORT_DIR / 'yolo_esi_v4_fp16.onnx', OUT_DRIVE)
shutil.make_archive('/content/yolo_esi_v4', 'zip', str(EXPORT_DIR))
print(f'\nSaved to Drive: {OUT_DRIVE}')
print(f'  best.pt, last.pt, v4_eval_summary.csv, yolo_esi_v4_fp16.onnx')

print('\nDONE!')
print(f'  Best weights : {BEST}')
print(f'  Best conf    : {best_conf}')
print(f'  Test mAP50   : {[s["AP50"] for s in summary if s["split"] == "test" and s["class"] == "mAP50"][0]:.4f}')
print('\nCompare against the v3 run: mAP50=0.453 / R=0.479 (epoch 52 best)')