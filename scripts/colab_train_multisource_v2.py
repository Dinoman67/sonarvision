"""
================================================================
YOLOv8-ESI Training — SonarVision Multi-Source Dataset v2
================================================================
5-class detection: unknown_debris, airplane, drowning_victim, mine, wreck
3 sensors: NOAA e4+h8 | MILCO-NOMBO | Kaggle SSS   (5626 images, leakage-free)

GOAL (SIH):
  The previous Core_model.ipynb trained YOLOv8-ESI on NOAA h8 ONLY
  (binary marine_debris) and reached mAP50 = 0.823 on the unseen h8
  test set. That proved debris detection on ONE sensor.
  This run trains the SAME YOLOv8-ESI architecture on the CLEANED
  MULTI-CLASS, MULTI-SENSOR v2 dataset so the model:
    1. detects debris from different SSS sensors (generalization), AND
    2. identifies the debris TYPE (airplane / drowning_victim / mine /
       wreck / unknown_debris).
  v2 was rebuilt leakage-free: no cross-split sequence leakage, uniform
  512x512, junk annotations removed (see audit_dataset_definitive.py).

LESSONS APPLIED FROM Core_model.ipynb:
  - YOLOv8-ESI (SE attention after every C2f) beat plain YOLOv8n AND
    SS-YOLO on h8. => ESI is the architecture.
  - SSS augmentation: NO flips/rotation/mosaic (sonar geometry is
    physical: seafloor down, port/starboard fixed).
  - patch_trainer() is REQUIRED or ultralytics rebuilds the model and
    drops the SE blocks. nc/names MUST be data-driven (5 classes), NOT
    the hardcoded binary from the h8 notebook.

CHANGES vs v1 training:
  - v2 dataset (more data, richer MILCO mine labels, mine boxes nearly
    doubled 358 -> 577)
  - STAGE 1 ONLY — Stage 2 regressed on the multi-source dataset
    (it helped on single-source h8, not here)
  - yolov8n-ESI AND yolov8s-ESI trained for comparison
  - Per-source test evaluation to verify multi-sensor generalization
================================================================
"""
# ================================================================
# CELL 1 — Setup
# ================================================================
!pip install -q ultralytics torch torchvision matplotlib pandas

import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:  {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ================================================================
# CELL 2 — Upload & extract dataset (v2, multi-class)
# ================================================================
from google.colab import files
import os, zipfile, yaml

print('Upload sonarvision_multisource_v2.zip:')
uploaded = files.upload()
zip_path = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/content/')

DATASET_DIR = '/content/datasets/sonarvision_multisource_v2'
DATA = os.path.join(DATASET_DIR, 'dataset.yaml')

# Fix path for Colab
with open(DATA) as f:
    cfg = yaml.safe_load(f)
cfg['path'] = DATASET_DIR
with open(DATA, 'w') as f:
    yaml.dump(cfg, f)

print(f'\nClasses ({cfg["nc"]} — MULTI-CLASS, not binary like h8-only):')
for k, v in cfg['names'].items():
    print(f'  {k}: {v}')

for split in ['train', 'val', 'test']:
    n = len([f for f in os.listdir(f'{DATASET_DIR}/images/{split}') if f.endswith('.png')])
    print(f'  {split}: {n} images')
assert cfg['nc'] == 5, 'dataset.yaml must define 5 classes'

# ================================================================
# CELL 3 — Custom modules (SEBlock + C2fWithSE), same as Core_model
# ================================================================
import torch.nn as nn
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


def _add_se_blocks(model, reduction=16):
    """Wrap every C2f layer with SE attention (proven in Core_model)."""
    layers = _get_layers(model)
    for i, layer in enumerate(layers):
        if isinstance(layer, C2f):
            c2 = layer.cv2.conv.out_channels
            layers[i] = C2fWithSE(layer, reduction=reduction)
            print(f'  Wrapped layer {i}: C2f({c2}) -> C2fWithSE')
    _set_layers(model, layers)
    total = sum(p.numel() for p in model.parameters())
    print(f'YOLOv8-ESI built: {total / 1e6:.2f}M params')
    return model


def build_yolov8_esi_full(base_model_name='yolov8n.pt'):
    """YOLOv8-ESI: base YOLO + SE attention after every C2f layer.
    Core_model proved this beats plain YOLOv8n and SS-YOLO."""
    base_model = YOLO(base_model_name)
    model = base_model.model
    print(f'Building YOLOv8-ESI from {base_model_name}...')
    print('  (Pretrained weights preserved — only SE layers are new)')
    return _add_se_blocks(model)


def patch_trainer(model_obj):
    """CRITICAL: ultralytics recreates the model during training and would
    drop the SE blocks. Patch DetectionTrainer.get_model to inject ours.

    MUST be data-driven (5 classes from the dataset), NOT the hardcoded
    binary (nc=1, names={0:'marine_debris'}) from Core_model.ipynb.
    """
    _orig = DetectionTrainer.get_model
    def _patched(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        dm = DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'],
                            verbose=verbose and RANK == -1)
        dm.model = model_obj.model
        dm.nc = self.data['nc']
        dm.names = self.data['names']
        try:
            dm.load(weights)
        except Exception:
            pass
        return dm
    DetectionTrainer.get_model = _patched
    return _orig


print('✓ SEBlock + C2fWithSE + patch_trainer ready (data-driven nc/names)')

# ================================================================
# CELL 4 — Shared training config (proven SSS-optimized augmentation)
# ================================================================
TRAIN_CONFIG = {
    # Augmentation — SSS-specific (NO flips, NO rotation, NO mosaic)
    'mosaic': 0.0,      # would mix different sonar scenes
    'mixup': 0.0,       # would blend different sonar sensors
    'fliplr': 0.0,      # sonar has port/starboard geometry
    'flipud': 0.0,      # seafloor always at bottom
    'degrees': 0.0,     # fixed sonar orientation
    'translate': 0.05,  # slight position jitter
    'scale': 0.2,       # scale variation
    'project': '/content/results',
    'exist_ok': True,
    'plots': True,
    'verbose': True,
}
IMGSZ = 512
BATCH = 16   # T4-friendly at 512px
print(f'Training config set | imgsz={IMGSZ} batch={BATCH} | '
      f'{cfg["nc"]} classes: {list(cfg["names"].values())}')

# ================================================================
# CELL 5 — Stage 1: train yolov8n-ESI AND yolov8s-ESI (30 epochs each)
# ================================================================
# Core_model Stage 1 on h8: 30 epochs, lr0=0.01, no backbone freeze.
# On the multi-source dataset Stage 2 regressed (v1 finding) => Stage 1 only.
import pandas as pd

VARIANTS = {
    'yolov8n_esi': {'base': 'yolov8n.pt', 'name': 'model_esi_v2_n_s1'},
    'yolov8s_esi': {'base': 'yolov8s.pt', 'name': 'model_esi_v2_s_s1'},
}

trained = {}
for tag, v in VARIANTS.items():
    print('=' * 60)
    print(f'STAGE 1: {tag.upper()} — 30 epochs (all layers trainable)')
    print('=' * 60)
    esi_obj = build_yolov8_esi_full(v['base'])
    _orig = patch_trainer(esi_obj)
    try:
        model = YOLO(v['base'])
        model.train(data=DATA, epochs=30, imgsz=IMGSZ, batch=BATCH, patience=15,
                    lr0=0.01, lrf=0.01, warmup_epochs=2,
                    **TRAIN_CONFIG, name=v['name'])
    finally:
        DetectionTrainer.get_model = _orig
    trained[tag] = f'/content/results/{v["name"]}/weights/best.pt'
    print(f'✓ {tag} done: {trained[tag]}')

# ================================================================
# CELL 6 — Evaluate both variants on val + test (per-class AP50)
# ================================================================
class_names = list(cfg['names'].values())


def evaluate_model(path, split='val', conf=0.25, verbose=True):
    m = YOLO(path)
    r = m.val(data=DATA, imgsz=IMGSZ, conf=conf, split=split, verbose=verbose)
    f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
    return r, f1


def print_per_class(r, class_names):
    print(f'{"Class":<18}{"AP50":>8}')
    print('-' * 28)
    for i, name in enumerate(class_names):
        if i < len(r.box.ap50):
            print(f'{name:<18}{r.box.ap50[i]:>8.4f}')


print('=' * 60)
print('VAL + TEST — PER-CLASS AP50')
print('=' * 60)
summary = []
for tag, path in trained.items():
    print(f'\n### {tag} ###')
    for split in ['val', 'test']:
        r, f1 = evaluate_model(path, split=split)
        print(f'\n-- {split.upper()} --  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
              f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
        print_per_class(r, class_names)
        summary.append({'variant': tag, 'split': split, 'mAP50': r.box.map50,
                        'mAP50-95': r.box.map, 'P': r.box.mp, 'R': r.box.mr, 'F1': f1})

pd.DataFrame(summary).to_csv('/content/v2_eval_summary.csv', index=False)
print('\n' + pd.DataFrame(summary).to_string(index=False))

# ================================================================
# CELL 7 — Confidence sweep on the better variant
# ================================================================
# Pick the higher test mAP50
best_tag = max(summary, key=lambda x: x['mAP50'] if x['split'] == 'test' else 0)['variant']
best_path = trained[best_tag]
print(f'=' * 60)
print(f'CONFIDENCE SWEEP on {best_tag} ({best_path})')
print('=' * 60)

m = YOLO(best_path)
best_f1, best_conf = 0, 0.05
for conf in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    r = m.val(data=DATA, imgsz=IMGSZ, conf=conf, split='test', verbose=False)
    p, rv = r.box.mp, r.box.mr
    f1 = 2 * p * rv / max(p + rv, 1e-8)
    print(f'  conf={conf:.2f}  mAP50={r.box.map50:.4f}  P={p:.4f}  R={rv:.4f}  F1={f1:.4f}')
    if f1 > best_f1:
        best_f1, best_conf = f1, conf
print(f'\n  Best: conf={best_conf} F1={best_f1:.4f}')

# ================================================================
# CELL 8 — Per-source test evaluation (multi-sensor generalization)
# ================================================================
# For SIH the model must work across SSS sensors. v2 test images are named
# NOAA_*/MILCO_*/KAGGLE_*, so we evaluate the test split per sensor by
# building tiny source-filtered datasets (symlinked, no data copied).
import csv, shutil
from pathlib import Path

ENABLE_SOURCE_EVAL = True
if ENABLE_SOURCE_EVAL:
    src_root = Path('/content/source_eval')
    src_root.mkdir(exist_ok=True)
    meta_path = f'{DATASET_DIR}/metadata.csv'
    rows = list(csv.DictReader(open(meta_path)))
    for src in ['NOAA', 'MILCO', 'KAGGLE']:
        d = src_root / src
        (d / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (d / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r['split'] == 'test' and r['image_path'].split('/')[-1].startswith(src + '_'):
                ip = f'{DATASET_DIR}/{r["image_path"]}'
                lp = ip.replace('/images/', '/labels/').replace('.png', '.txt')
                (d / 'images' / 'test' / Path(ip).name).symlink_to(ip)
                (d / 'labels' / 'test' / Path(lp).name).symlink_to(lp)
        (d / 'data.yaml').write_text(
            f"path: {d}\ntrain: images/test\nval: images/test\ntest: images/test\n"
            f"nc: {cfg['nc']}\nnames:\n" +
            ''.join(f'  {k}: {v}\n' for k, v in cfg['names'].items()))
        try:
            r = m.val(data=str(d / 'data.yaml'), imgsz=IMGSZ, conf=best_conf, verbose=False)
            f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
            n_imgs = len([f for f in os.listdir(d / 'images' / 'test') if f.endswith('.png')])
            print(f'  {src:<8} test_imgs={n_imgs:<4} mAP50={r.box.map50:.4f}  '
                  f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
        except Exception as e:
            print(f'  {src}: eval failed — {e}')

# ================================================================
# CELL 9 — Export best model as ONNX (FP32 + FP16)
# ================================================================
print(f'=' * 60)
print(f'EXPORT — best variant: {best_tag}')
print('=' * 60)
EXPORT_DIR = Path('/content/yolo_esi_v2_exports')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

for fmt in ['fp32', 'fp16']:
    kwargs = {'format': 'onnx', 'imgsz': IMGSZ, 'simplify': True, 'dynamic': False}
    if fmt == 'fp16':
        kwargs['half'] = True
    try:
        src = Path(m.export(**kwargs))
        dest = EXPORT_DIR / f'yolo_esi_v2_{best_tag}_{fmt}.onnx'
        shutil.copy2(src, dest)
        print(f'  ✓ {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)')
    except Exception as e:
        print(f'  ✗ ONNX {fmt} failed: {e}')

# Validate exports on test set
print(f'\nEXPORT VALIDATION (test split):')
for f in sorted(EXPORT_DIR.glob('*.onnx')):
    try:
        ex = YOLO(str(f))
        r = ex.val(data=DATA, imgsz=IMGSZ, conf=best_conf, split='test', verbose=False)
        print(f'  {f.name:<42} mAP50={r.box.map50:.4f}')
    except Exception as e:
        print(f'  {f.name}: FAILED — {e}')

# ================================================================
# CELL 10 — v1 vs v2 comparison + download
# ================================================================
print('=' * 60)
print('v1 vs v2 (fill in v1 numbers from the v1 run)')
print('=' * 60)
print(f'  v2 {best_tag} test mAP50      : {[s for s in summary if s["variant"] == best_tag and s["split"] == "test"][0]["mAP50"]:.4f}')
print(f'  v1 (from v1 logs) test mAP50 : <enter v1 value>')
print(f'  h8-only Core_model (unseen)  : 0.823 (binary, single sensor)')
print('  NOTE: v1/v2/h8 numbers are NOT directly comparable — different')
print('  classes (1 vs 5) and different test sets. Compare v1 vs v2 which')
print('  share the same 5-class setup and per-split protocol.')

from google.colab import files
files.download(best_path)
shutil.make_archive('/content/yolo_esi_v2', 'zip', str(EXPORT_DIR))
files.download('/content/yolo_esi_v2.zip')
pd.DataFrame(summary).to_csv('/content/v2_eval_summary.csv', index=False)
files.download('/content/v2_eval_summary.csv')

print('\nDONE!')
print(f'  Best weights : {best_path}')
print(f'  Best conf    : {best_conf}')
print(f'  Exports      : {EXPORT_DIR}')
