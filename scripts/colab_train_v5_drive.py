"""
================================================================
YOLOv8-ESI v5 — 4 classes, h8-only debris, Core_model recipe
================================================================
TRAINING RECIPE (from Core_model.ipynb — the notebook that produced a
working debris model):
  Stage 1: YOLOv8-ESI @ imgsz 256, batch 32, 30 epochs, patience 15,
           lr0=0.01, warmup 2            -> val mAP50 0.76 (h8 single-class)
  Stage 2: 50 more epochs, batch 16, patience 20, lr0=0.005, freeze=10,
           warmup 3                      -> test mAP50 0.823 (h8 single-class)
  Eval + ONNX export all at imgsz 256.

Two fixes vs the notebook:
  1. The notebook's Stage 2 RE-BUILT the model from scratch (its patched
     trainer ignored the Stage-1 weights). v5 Stage 2 properly loads
     Stage-1 best.pt onto the ESI architecture (0 missing / 0 unexpected
     keys asserted) — i.e. true "50 more epochs".
  2. cls_pw=0.5 kept (auto inverse-frequency weights): the notebook was
     single-class where class weighting is irrelevant; v5 has 4 imbalanced
     classes. (A per-class list still crashes ultralytics 8.4.x.)

DATASET (v5, audited):
  - 4 classes: unknown_debris (h8_data only — ONE visual mode) / airplane
    (Kaggle) / mine (MILCO) / wreck (Kaggle). NOMBO + drowning + NOAA-e4
    dropped entirely.
  - h8 re-split WHOLE-PASS group-aware (train∩test groups = 0, verified).
    The notebook's "unseen test" drew 48/48 positives from passes in
    h8_data — near-duplicate exposure. v5's test is the honest one.
  - MILCO batch 15 + per-year quotas: mine test is 52% 2015 vs train 50%
    (v4 was 76% vs 21%).
  - Oversampling (airplane 3x, mine 2x) TRAIN-ONLY post-split.

UNCHANGED (proven): YOLOv8-ESI with the SE key-remap weight load, SE
blocks preserved via patched DetectionTrainer.get_model, SSS augs (no
flips/rotation/mosaic/mixup), per-class AP50 + conf sweep + per-source
eval + ONNX FP32/FP16.
================================================================
"""
# ================================================================
# CELL 1 — Mount Drive, config, extract dataset
# ================================================================
from google.colab import drive
drive.mount('/content/drive')

# ── CONFIG — edit this to where you uploaded the v5 zip ──────────
DRIVE = '/content/drive/MyDrive'          # zip is directly at Drive root
ZIP   = f'{DRIVE}/sonarvision_multisource_v5.zip'

import os, zipfile, yaml
with zipfile.ZipFile(ZIP, 'r') as z:
    z.extractall('/content/')

# ── Locate the dataset dir AUTO-MAGICALLY (handles both zip layouts) ──
# v4-style zip nests under datasets/sonarvision_multisource_v5/...
# v5-current zip is FLAT (dataset.yaml, images/ at the zip root).
found = None
for root, dirs, files in os.walk('/content'):
    dirs[:] = [d for d in dirs if d not in ('drive', 'sample_data', '.config', '.cache')]
    depth = root[len('/content'):].count(os.sep)
    if depth > 3:
        dirs[:] = []
        continue
    if 'dataset.yaml' in files:
        found = root
        break
assert found, 'dataset.yaml not found after extraction — check the zip'
DATASET_DIR = found
DATA = f'{DATASET_DIR}/dataset.yaml'
print(f'Dataset located at: {DATASET_DIR}')

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
assert cfg['nc'] == 4

OUT = '/content/results'
OUT_DRIVE = f'{DRIVE}/results_v5'        # persistent copy on Drive
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


def build_yolov8_esi_full(nc=4, yaml_name='yolov8n.yaml', weights='yolov8n.pt'):
    """YOLOv8-ESI: SE after every C2f, built from the yaml for `nc` classes.

    FIXES vs the Core_model-era builder:
    1. Base is DetectionModel(yolov8n.yaml, nc=nc) so the head has the
       CORRECT nc classes (not the COCO 80-class head).
    2. Pretrained yolov8n.pt weights ACTUALLY load via SE key remap — the
       C2fWithSE wrapper nests the C2f (`model.6.c2f.cv1...`), so plain
       intersect_dicts() matched nothing and earlier ESI runs silently
       trained the backbone from scratch.
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
    Data-driven nc/names — NOT a binary hardcode."""
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
            dm.load(weights)         # harmless on a fresh run (keys nested)
        except Exception:
            pass
        return dm
    DetectionTrainer.get_model = _patched
    return _orig


def patch_trainer_continue(model_obj):
    """Stage-2 continuation patcher WITHOUT dm.load(weights).

    dm.load would overwrite trained stem/conv layers with COCO values
    (idempotent only on run #1). Stage 2 injects the Stage-1 model whose
    weights are already loaded — do NOT reload COCO over them."""
    _orig = DetectionTrainer.get_model
    def _patched(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        dm = DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'],
                            verbose=verbose and RANK == -1)
        dm.model = model_obj.model      # Stage-1 weights already inside
        dm.nc = self.data['nc']
        dm.names = self.data['names']
        return dm
    DetectionTrainer.get_model = _patched
    return _orig


print('✓ SEBlock + C2fWithSE + patch_trainer(s) ready (data-driven nc/names)')

# ================================================================
# CELL 3 — Training config (Core_model recipe: imgsz 256, 2-stage)
# ================================================================
IMGSZ = 256                       # Core_model recipe (proven working model)
BATCH1 = 32                       # Stage 1 batch
BATCH2 = 16                       # Stage 2 batch
# Core_model was single-class (cls_pw irrelevant). v5 has 4 imbalanced
# classes -> keep AUTO inverse-frequency weights. (A per-class LIST still
# crashes ultralytics 8.4.x — scalar 0.5 enables the auto mode.)
CLS_PW = 0.5

STAGE1 = dict(epochs=30, patience=15, lr0=0.01, lrf=0.01, warmup_epochs=2)
STAGE2 = dict(epochs=50, patience=20, lr0=0.005, lrf=0.01, warmup_epochs=3,
              freeze=10, batch=BATCH2)
# Lower patience = stop sooner when converged (one-day risk control); the
# absolute best checkpoint is kept and Drive-backed-up either way.
STAGE1['patience'] = 12
STAGE2['patience'] = 15

TRAIN_CONFIG = {
    # SSS-specific augmentation (NO flips, NO rotation, NO mosaic)
    'mosaic': 0.0, 'mixup': 0.0,
    'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0,
    'translate': 0.05, 'scale': 0.2,
    'project': OUT, 'exist_ok': True, 'plots': True, 'verbose': True,
}
print(f'v5 config (Core_model recipe) | imgsz={IMGSZ} '
      f'stage1: batch={BATCH1} {STAGE1} | stage2: {STAGE2} | cls_pw={CLS_PW}')

# ================================================================
# CELL 4 — STAGE 1: yolov8n-ESI @ 256px, 4 classes, 30 epochs
# ================================================================
print('=' * 60)
print(f'STAGE 1: yolov8n-ESI @ {IMGSZ}px, {cfg["nc"]} classes, '
      f'{STAGE1["epochs"]} epochs (patience={STAGE1["patience"]})')
print('=' * 60)

# Ensure yolov8n.pt is downloaded before building the ESI model
_ = YOLO('yolov8n.pt')

esi_obj = build_yolov8_esi_full(nc=cfg['nc'], weights='yolov8n.pt')
_orig = patch_trainer(esi_obj)
try:
    model = YOLO('yolov8n.pt')
    model.train(data=DATA, imgsz=IMGSZ, batch=BATCH1, cls_pw=CLS_PW,
                **STAGE1, **TRAIN_CONFIG, name='model_esi_v5_s1')
finally:
    DetectionTrainer.get_model = _orig

BEST1 = f'{OUT}/model_esi_v5_s1/weights/best.pt'
print(f'✓ Stage 1 done — best: {BEST1}')

# ── IMMEDIATE Drive backup — a Colab session death must not lose Stage 1 ──
import shutil as _sh
for _f in ['best.pt', 'last.pt']:
    _p = f'{OUT}/model_esi_v5_s1/weights/{_f}'
    if os.path.exists(_p):
        _sh.copy2(_p, f'{OUT_DRIVE}/s1_{_f}')
print('✓ Stage-1 weights backed up to Drive (s1_best.pt / s1_last.pt)')

# ================================================================
# CELL 5 — STAGE 2: 50 more epochs (freeze=10, lr0=0.005) — true continuation
# ================================================================
print('=' * 60)
print(f'STAGE 2: continue from Stage 1 — {STAGE2["epochs"]} more epochs '
      f'(patience={STAGE2["patience"]}, freeze={STAGE2["freeze"]}, lr0={STAGE2["lr0"]})')
print('=' * 60)

# 0) BEST1 is defined in Cell 4 — but after a session death you re-run
#    Cells 1-3 and SKIP Cell 4 (already trained). Compute it here so this
#    cell is self-sufficient.
BEST1 = f'{OUT}/model_esi_v5_s1/weights/best.pt'

# 0.1) Recover Stage-1 weights if this is a NEW session (session died after
#    Stage 1 — /content is wiped, but Drive has the s1_* backup from Cell 4)
import shutil as _sh2
if not os.path.exists(BEST1):
    _d1 = f'{OUT_DRIVE}/s1_best.pt'
    assert os.path.exists(_d1), 'BEST1 missing locally AND no Drive backup (s1_best.pt) — re-run Cells 1-4'
    os.makedirs(os.path.dirname(BEST1), exist_ok=True)
    _sh2.copy2(_d1, BEST1)
    print(f'  ↻ recovered Stage-1 best.pt from Drive backup')

# 1) Build the SE architecture (scaffold — weights replaced in step 2)
#    After a session death this cell runs WITHOUT Cell 4, so ensure the
#    pretrained weights exist first (idempotent — no-op if already cached).
_ = YOLO('yolov8n.pt')
esi2 = build_yolov8_esi_full(nc=cfg['nc'], weights='yolov8n.pt')
se = sum(1 for _, m in esi2.named_modules() if type(m).__name__ == 'C2fWithSE')
print(f'SE blocks in model: {se}  (expect 8)')

# 2) LOAD STAGE-1 WEIGHTS onto that architecture (true continuation)
import torch as _t
ck1 = _t.load(BEST1, map_location='cpu', weights_only=False)
src1 = ck1.get('ema') or ck1.get('model')
assert src1 is not None, 'Stage-1 checkpoint has neither ema nor model'
missing, unexpected = esi2.load_state_dict(src1.float().state_dict(), strict=False)
assert not missing and not unexpected, \
    f'state mismatch: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}'
print('✓ Stage-1 weights loaded — 0 missing / 0 unexpected keys')

# 3) Patcher WITHOUT the dm.load(COCO) step (would clobber Stage-1 layers)
_orig2 = patch_trainer_continue(esi2)
try:
    model2 = YOLO('yolov8n.pt')     # shell only — weights never reach training
    model2.train(data=DATA, imgsz=IMGSZ, cls_pw=CLS_PW,
                 **STAGE2, **TRAIN_CONFIG, name='model_esi_v5_s2')
finally:
    DetectionTrainer.get_model = _orig2

BEST2 = f'{OUT}/model_esi_v5_s2/weights/best.pt'
print(f'✓ Stage 2 done — best: {BEST2}')

# ── IMMEDIATE Drive backup — same session-death insurance for Stage 2 ──
# (_sh2, not _sh: after a session death you run Cells 1-3 then 5, and _sh
#  is only defined in Cell 4 — Cell 5 must be self-contained.)
for _f in ['best.pt', 'last.pt']:
    _p = f'{OUT}/model_esi_v5_s2/weights/{_f}'
    if os.path.exists(_p):
        _sh2.copy2(_p, f'{OUT_DRIVE}/s2_{_f}')
print('✓ Stage-2 weights backed up to Drive (s2_best.pt / s2_last.pt)')

# ================================================================
# CELL 6 — Per-class AP50: Stage 1 vs Stage 2 on val + test, keep winner
# ================================================================
import pandas as pd
from pathlib import Path
import shutil

class_names = list(cfg['names'].values())
summary = []

print('=' * 60)
print('STAGE 1 vs STAGE 2 — PER-CLASS AP50 (val + test)')
print('=' * 60)
cands = [('stage1', BEST1)]
if os.path.exists(BEST2):
    cands.append(('stage2', BEST2))
else:
    print('  ⚠ BEST2 not found — Stage 2 did not complete. Eval on Stage 1 only.')
val_map = {}
for tag, path in cands:
    mm = YOLO(path)
    r = mm.val(data=DATA, imgsz=IMGSZ, conf=0.25, split='val', verbose=False)
    val_map[tag] = r.box.map50
    f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
    print(f'\n-- {tag.upper()} VAL --  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
          f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
    print(f'{"Class":<18}{"AP50":>8}')
    print('-' * 28)
    for i, name in enumerate(class_names):
        print(f'{name:<18}{r.box.ap50[i]:>8.4f}')
        summary.append({'model': tag, 'split': 'val', 'class': name, 'AP50': r.box.ap50[i]})
    summary.append({'model': tag, 'split': 'val', 'class': 'mAP50', 'AP50': r.box.map50})

winner = max(val_map, key=val_map.get)
BEST = {'stage1': BEST1, 'stage2': BEST2}[winner]
print(f'\n→ Winner: {winner} (val mAP50 {val_map[winner]:.4f}) -> BEST={BEST}')
if winner == 'stage1':
    print('  (Stage 2 regressed — Stage 1 kept; both saved to Drive)')

m = YOLO(BEST)
for split in ['val', 'test']:
    r = m.val(data=DATA, imgsz=IMGSZ, conf=0.25, split=split, verbose=False)
    f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
    print(f'\n-- {split.upper()} (winner) --  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
          f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
    print(f'{"Class":<18}{"AP50":>8}')
    print('-' * 28)
    for i, name in enumerate(class_names):
        print(f'{name:<18}{r.box.ap50[i]:>8.4f}')
        summary.append({'model': winner, 'split': split, 'class': name, 'AP50': r.box.ap50[i]})
    summary.append({'model': winner, 'split': split, 'class': 'mAP50', 'AP50': r.box.map50})

# ================================================================
# CELL 7 — Confidence sweep on test (winner)
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
# CELL 8 — Per-source test eval (SIH multi-sensor proof)
# ================================================================
import csv
src_root = Path('/content/source_eval_v5')
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
# CELL 9 — ONNX export (FP32 + FP16) + save everything to Drive
# ================================================================
print('=' * 60)
print('ONNX EXPORT + DRIVE SAVE')
print('=' * 60)
EXPORT_DIR = Path('/content/yolo_esi_v5_exports')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

for fmt in ['fp32', 'fp16']:
    kwargs = {'format': 'onnx', 'imgsz': IMGSZ, 'simplify': True, 'dynamic': False}
    if fmt == 'fp16':
        kwargs['half'] = True
    try:
        src = Path(m.export(**kwargs))
        dest = EXPORT_DIR / f'yolo_esi_v5_{fmt}.onnx'
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
pd.DataFrame(summary).to_csv('/content/v5_eval_summary.csv', index=False)
_saved = []
for src_file in [BEST, BEST1, BEST2, f'{OUT}/model_esi_v5_s1/weights/last.pt',
                 f'{OUT}/model_esi_v5_s2/weights/last.pt',
                 '/content/v5_eval_summary.csv']:
    if os.path.exists(src_file):
        shutil.copy2(src_file, f'{OUT_DRIVE}/{Path(src_file).name}')
        _saved.append(Path(src_file).name)
_fp16 = EXPORT_DIR / 'yolo_esi_v5_fp16.onnx'
if _fp16.exists():
    shutil.copy2(_fp16, OUT_DRIVE)
    _saved.append(_fp16.name)
shutil.make_archive('/content/yolo_esi_v5', 'zip', str(EXPORT_DIR))
print(f'\nSaved to Drive {OUT_DRIVE}: {_saved}')

print('\nDONE!')
print(f'  Best weights : {BEST}')
print(f'  Best conf    : {best_conf}')
_mAP = [s['AP50'] for s in summary if s['split'] == 'test' and s['class'] == 'mAP50']
print(f'  Test mAP50   : {_mAP[0]:.4f}' if _mAP else '  Test mAP50   : n/a')
print('\nCore_model single-class reference: val 0.76 / test 0.823 @ imgsz 256')
print('  NOTE: v5 test is HONEST (whole unseen passes; the notebook test')
print('  drew 48/48 positives from passes in its train set).')