"""
================================================================
YOLOv8-ESI Training — SonarVision Multi-Source Dataset v1
================================================================

5-class detection: unknown_debris, airplane, drowning_victim, mine, wreck
3 sources: NOAA + MILCO-NOMBO + Kaggle SSS (2030 images, leakage-free)

Previous training was BINARY (1 class: marine_debris) on NOAA only.
This training is MULTI-CLASS (5 classes) on 3 mixed sources.

Previous training used a noise pipeline for augmentation.
This training starts CLEAN — noise can be added as v3 experiment.

Based on the proven Core_model.ipynb approach.
================================================================
"""
# Output: /content/results/
# ================================================================
# CELL 1 — Setup
# ================================================================
!pip install -q ultralytics torch torchvision matplotlib pandas

import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:  {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
# ================================================================
# CELL 2 — Upload & extract dataset
# ================================================================
from google.colab import files
import os, zipfile, yaml

print('Upload sonarvision_multisource_v1.zip:')
uploaded = files.upload()
zip_path = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/content/')

DATASET_DIR = '/content/datasets/sonarvision_multisource_v1'
DATA = os.path.join(DATASET_DIR, 'dataset.yaml')

# Fix path for Colab
with open(DATA) as f:
    cfg = yaml.safe_load(f)
cfg['path'] = DATASET_DIR
with open(DATA, 'w') as f:
    yaml.dump(cfg, f)

# Show classes
print(f'\nClasses ({cfg["nc"]} classes — NOT binary like previous training):')
for k, v in cfg['names'].items():
    print(f'  {k}: {v}')

for split in ['train', 'val', 'test']:
    n = len([f for f in os.listdir(f'{DATASET_DIR}/images/{split}') if f.endswith('.png')])
    print(f'  {split}: {n} images')
# ================================================================
# CELL 3 — Define custom modules (SE + C2fWithSE)
# ================================================================
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect.train import DetectionTrainer

# --- SE Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w

# --- C2fWithSE: wraps C2f + adds SE ---
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

print('✓ SEBlock + C2fWithSE defined')
# ================================================================
# CELL 4 — Build YOLOv8-ESI & patch trainer
# ================================================================
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
    layers = _get_layers(model)
    for i, layer in enumerate(layers):
        if isinstance(layer, C2f):
            c2 = layer.cv2.conv.out_channels
            layers[i] = C2fWithSE(layer, reduction=reduction)
            print(f'  Wrapped layer {i}: C2f({c2}) → C2fWithSE')
    _set_layers(model, layers)
    total = sum(p.numel() for p in model.parameters())
    print(f'\nYOLOv8-ESI built: {total/1e6:.2f}M params')
    return model

def build_yolov8_esi_full(pretrained='yolov8n.pt'):
    """Build YOLOv8-ESI: YOLOv8n backbone + SE attention after each C2f."""
    base_model = YOLO(pretrained)
    model = base_model.model
    print('Building YOLOv8-ESI with SE attention...')
    print('  (Pretrained YOLOv8n weights preserved — only SE layers are new)')
    return _add_se_blocks(model)

def patch_trainer(model_obj):
    """CRITICAL: ultralytics recreates model during training.
    Without this patch, custom SE blocks are lost.

    KEY DIFFERENCE from previous binary training:
      Previous: dm.nc = 1, dm.names = {0: 'marine_debris'}
      This:     dm.nc = self.data['nc'], dm.names = self.data['names']
      Because we now have 5 classes, not 1.
    """
    _orig = DetectionTrainer.get_model
    def _patched(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        dm = DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'],
                            verbose=verbose and RANK == -1)
        dm.model = model_obj.model
        # === USE DATA-DRIVEN CLASS INFO (5 classes) ===
        # Previous binary training hardcoded: dm.nc = 1, dm.names = {0: 'marine_debris'}
        # We have 5 classes now, so we read from the dataset config
        dm.nc = self.data['nc']
        dm.names = self.data['names']
        try:
            dm.load(weights)
        except Exception:
            pass
        return dm
    DetectionTrainer.get_model = _patched
    return _orig

# Build ESI model
esi_obj = build_yolov8_esi_full('yolov8n.pt')
print('✓ Ready for training')
# ================================================================
# CELL 5 — Training hyperparameters (shared config)
# ================================================================
# Previous binary training used imgsz=256, batch=32 (small dataset, 1 class)
# We use imgsz=512, batch=16 (larger images, 5 classes, 3 sources)

TRAIN_CONFIG = {
    # Augmentation — SSS-specific (NO flips, NO rotation, NO mosaic)
    'mosaic': 0.0,      # Would mix different sonar scenes
    'mixup': 0.0,       # Would blend different sonar sources
    'fliplr': 0.0,      # Sonar has port/starboard geometry
    'flipud': 0.0,      # Seafloor always at bottom
    'degrees': 0.0,     # Fixed sonar orientation
    'translate': 0.05,  # Slight position jitter
    'scale': 0.2,       # Scale variation

    # Output
    'project': '/content/results',
    'exist_ok': True,
    'plots': True,
    'verbose': True,
}

print('Training config set (SSS-optimized augmentation)')
print(f'  Classes: {cfg["nc"]} ({list(cfg["names"].values())})')
print(f'  Image size: 512px (previous training used 256px)')
# ================================================================
# CELL 6 — Stage 1: Train 30 epochs (full model)
# ================================================================
print('='*60)
print('STAGE 1: YOLOv8-ESI — 30 epochs (all layers trainable)')
print('='*60)

esi_obj = build_yolov8_esi_full('yolov8n.pt')
_orig = patch_trainer(esi_obj)

try:
    model = YOLO('yolov8n.pt')
    model.train(
        data=DATA,
        epochs=30,
        imgsz=512,
        batch=16,
        patience=15,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=2,
        **TRAIN_CONFIG,
        name='model_esi_v1_s1',
    )
finally:
    DetectionTrainer.get_model = _orig

print('✓ Stage 1 done')
# ================================================================
# CELL 7 — Evaluate Stage 1
# ================================================================
best_st1 = '/content/results/model_esi_v1_s1/weights/best.pt'
m = YOLO(best_st1)

# Validation
r = m.val(data=DATA, imgsz=512, conf=0.25, verbose=True)
f1 = 2*r.box.mp*r.box.mr / max(r.box.mp+r.box.mr, 1e-8)
print(f'\n--- Stage 1 Validation ---')
print(f'  mAP50:     {r.box.map50:.4f}')
print(f'  Precision: {r.box.mp:.4f}')
print(f'  Recall:    {r.box.mr:.4f}')
print(f'  F1:        {f1:.4f}')
# ================================================================
# CELL 8 — Confidence sweep (find optimal threshold for 5 classes)
# ================================================================
print('='*60)
print('CONFIDENCE SWEEP (5 classes)')
print('='*60)

best_f1, best_conf = 0, 0.05
for conf in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    r = m.val(data=DATA, imgsz=512, conf=conf, verbose=False)
    p, rv = r.box.mp, r.box.mr
    f1 = 2*p*rv / max(p+rv, 1e-8)
    print(f'  conf={conf:.2f}  mAP50={r.box.map50:.4f}  P={p:.4f}  R={rv:.4f}  F1={f1:.4f}')
    if f1 > best_f1:
        best_f1, best_conf = f1, conf

print(f'\n  Best: conf={best_conf} F1={best_f1:.4f}')
# ================================================================
# CELL 9 — Stage 2: 50 more epochs (freeze backbone)
# ================================================================
print('='*60)
print('STAGE 2: YOLOv8-ESI — 50 more epochs (freeze backbone)')
print('='*60)

esi_obj = build_yolov8_esi_full(best_st1)
_orig = patch_trainer(esi_obj)

try:
    model = YOLO(best_st1)
    model.train(
        data=DATA,
        epochs=50,
        imgsz=512,
        batch=16,
        patience=20,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=3,
        freeze=10,
        **TRAIN_CONFIG,
        name='model_esi_v1_s2',
    )
finally:
    DetectionTrainer.get_model = _orig

print('✓ Stage 2 done')
# ================================================================
# CELL 10 — Final evaluation (val + test)
# ================================================================
best_st2 = '/content/results/model_esi_v1_s2/weights/best.pt'
m = YOLO(best_st2)

# Validation
v = m.val(data=DATA, imgsz=512, conf=best_conf, verbose=True)
f1_v = 2*v.box.mp*v.box.mr / max(v.box.mp+v.box.mr, 1e-8)

print(f'\n{"="*60}')
print(f'VALIDATION RESULTS')
print(f'{"="*60}')
print(f'  mAP50:     {v.box.map50:.4f}')
print(f'  mAP50-95:  {v.box.map:.4f}')
print(f'  Precision: {v.box.mp:.4f}')
print(f'  Recall:    {v.box.mr:.4f}')
print(f'  F1:        {f1_v:.4f}')
print(f'  Conf:      {best_conf}')

# Test set
t = m.val(data=DATA, imgsz=512, conf=best_conf, split='test', verbose=True)
f1_t = 2*t.box.mp*t.box.mr / max(t.box.mp+t.box.mr, 1e-8)

print(f'\n{"="*60}')
print(f'TEST SET RESULTS (unseen)')
print(f'{"="*60}')
print(f'  mAP50:     {t.box.map50:.4f}')
print(f'  mAP50-95:  {t.box.map:.4f}')
print(f'  Precision: {t.box.mp:.4f}')
print(f'  Recall:    {t.box.mr:.4f}')
print(f'  F1:        {f1_t:.4f}')
# ================================================================
# CELL 11 — Per-class results
# ================================================================
class_names = list(cfg['names'].values())

print(f'\n{"="*60}')
print(f'PER-CLASS RESULTS')
print(f'{"="*60}')

# Per-class AP from validation
if hasattr(v, 'ap_per_class') and v.ap_per_class is not None:
    print(f'{"Class":<20} {"AP50":>8}')
    print(f'{"-"*30}')
    for i, name in enumerate(class_names):
        if i < len(v.box.ap50):
            print(f'{name:<20} {v.box.ap50[i]:>8.4f}')
else:
    print('  (per-class AP not available — check confusion matrix)')
    print(f'  Confusion matrix: /content/results/model_esi_v1_s2/confusion_matrix.png')
# ================================================================
# CELL 12 — Visualize predictions on test set
# ================================================================
import matplotlib.pyplot as plt
from PIL import Image
import random

test_imgs = sorted([f for f in os.listdir(f'{DATASET_DIR}/images/test') if f.endswith('.png')])
samples = random.sample(test_imgs, min(8, len(test_imgs)))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for idx, name in enumerate(samples):
    if idx >= 8: break
    ax = axes[idx//4, idx%4]
    r = m.predict(f'{DATASET_DIR}/images/test/{name}', conf=best_conf, verbose=False)
    img = Image.open(f'{DATASET_DIR}/images/test/{name}')
    ax.imshow(img, cmap='gray')
    for box in r[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        c = int(box.cls[0]); cf = float(box.conf[0])
        ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,
                                   fill=False, edgecolor='lime', lw=2))
        ax.text(x1, y1-5, f'{class_names[c]} {cf:.2f}', color='lime', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    ax.set_title(name[:25], fontsize=9)
    ax.axis('off')
plt.suptitle('YOLOv8-ESI Predictions — 5-Class Multi-Source', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/predictions.png', dpi=150, bbox_inches='tight')
plt.show()
# ================================================================
# CELL 13 — Export FP32, FP16, INT8 ONNX
# ================================================================
import shutil
from pathlib import Path

EXPORT_DIR = Path('/content/yolo_esi_exports')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def export_onnx(model, fmt, calib_yaml=None):
    print(f'\n  Exporting: ONNX {fmt.upper()}')
    try:
        kwargs = {'format': 'onnx', 'imgsz': 512, 'simplify': True, 'dynamic': False}
        if fmt == 'fp16':
            kwargs['half'] = True
        elif fmt == 'int8':
            kwargs['int8'] = True
            kwargs['data'] = str(calib_yaml)
        src = Path(model.export(**kwargs))
        dest = EXPORT_DIR / f'yolo_esi_{fmt}.onnx'
        shutil.copy2(src, dest)
        size = dest.stat().st_size / (1024**2)
        print(f'    ✓ {dest} ({size:.2f} MB)')
        return dest
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        return None

fp32_path = export_onnx(m, 'fp32')
fp16_path = export_onnx(m, 'fp16')
int8_path = export_onnx(m, 'int8', calib_yaml=DATA)
# ================================================================
# CELL 14 — Validate all exports
# ================================================================
print(f'\n{"="*60}')
print('EXPORT VALIDATION (on test set)')
print(f'{"="*60}')

all_results = []
for name, path in [('Master .pt', best_st2), ('ONNX FP32', fp32_path),
                    ('ONNX FP16', fp16_path), ('ONNX INT8', int8_path)]:
    if not path or not Path(path).exists():
        continue
    try:
        ex = YOLO(str(path))
        r = ex.val(data=DATA, imgsz=512, conf=best_conf, split='test', verbose=False)
        p, rv = r.box.mp, r.box.mr
        f1 = 2*p*rv / max(p+rv, 1e-8)
        size = Path(path).stat().st_size / (1024**2)
        print(f'  {name:<16} mAP50={r.box.map50:.4f}  F1={f1:.4f}  Size={size:.2f}MB')
        all_results.append({'Model': name, 'mAP50': r.box.map50, 'F1': f1, 'Size_MB': size})
    except Exception as e:
        print(f'  {name:<16} FAILED: {e}')
# ================================================================
# CELL 15 — Download results
# ================================================================
from google.colab import files

# Download best weights
files.download(best_st2)

# Download all exports
shutil.make_archive('/content/yolo_esi_multisource', 'zip', '/content/results/model_esi_v1_s2')
files.download('/content/yolo_esi_multisource.zip')

if fp16_path and Path(fp16_path).exists():
    files.download(str(fp16_path))

print(f'\n{"="*60}')
print('DONE!')
print(f'{"="*60}')
print(f'Best weights:  {best_st2}')
print(f'Best conf:     {best_conf}')
print(f'Val mAP50:     {v.box.map50:.4f}')
print(f'Test mAP50:    {t.box.map50:.4f}')
print(f'{"="*60}')
