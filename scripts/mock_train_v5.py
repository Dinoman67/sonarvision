#!/usr/bin/env python3
"""
Local mock run of the v5 pipeline — validate + early-read the 4-class ESI
model before the real Colab run.

Recipe: Core_model (imgsz 256), capped at 10 epochs (patience 5).
Eval after training:
  - v5 val / v5 test (per-class AP50)        <- HONEST whole-pass split
  - per-source test (NOAA/MILCO/KAGGLE)
  - h8_unseen_test (debris class-0 AP50)     <- user's "unseen" reference
    (DISCLOSURE: 48/48 positives there come from passes present in
    h8_data, so it is a same-pass holdout — optimistic, not truly unseen.)
"""
import os, sys, csv, shutil, json, argparse
from pathlib import Path

import torch, torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect.train import DetectionTrainer

ROOT = Path('/home/ashish/sonar-vision')
DATA_YAML = ROOT / 'datasets/sonarvision_multisource_v5/dataset.yaml'
UNSEEN = ROOT / 'datasets/noaa-debris/h8_unseen_test'
OUT = ROOT / 'mock_runs'
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# YOLOv8-ESI machinery (identical to colab_train_v5_drive.py CELL 2)
# ---------------------------------------------------------------------------
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
          f'(SE-remapped)')
    return model


def patch_trainer(model_obj):
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


# ---------------------------------------------------------------------------
# Config (Core_model recipe, capped for the mock)
# ---------------------------------------------------------------------------
_p = argparse.ArgumentParser()
_p.add_argument('--imgsz', type=int, default=256)
_p.add_argument('--batch', type=int, default=16)
_p.add_argument('--epochs', type=int, default=10)
_p.add_argument('--name', type=str, default='mock_v5_s1')
_a = _p.parse_args()

IMGSZ = _a.imgsz
BATCH = _a.batch
EPOCHS = _a.epochs
NAME = _a.name
PATIENCE = 5
CLS_PW = 0.5

TRAIN_CONFIG = {
    'mosaic': 0.0, 'mixup': 0.0,
    'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0,
    'translate': 0.05, 'scale': 0.2,
    'project': str(OUT), 'name': NAME,
    'exist_ok': True, 'plots': True, 'verbose': True,
    'workers': 2, 'seed': 0,
}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    print('=' * 60)
    print(f'MOCK: yolov8n-ESI @ {IMGSZ}px, nc=4, {EPOCHS} epochs (patience={PATIENCE}) name={NAME}')
    print('=' * 60)

    _ = YOLO('yolov8n.pt')  # ensure weights cached
    esi_obj = build_yolov8_esi_full(nc=4, weights='yolov8n.pt')
    _orig = patch_trainer(esi_obj)
    try:
        model = YOLO('yolov8n.pt')
        model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                    patience=PATIENCE, lr0=0.01, lrf=0.01, warmup_epochs=2,
                    cls_pw=CLS_PW, **TRAIN_CONFIG)
    finally:
        DetectionTrainer.get_model = _orig

    best = OUT / f'{NAME}/weights/best.pt'
    print(f'✓ trained — best: {best}')
    m = YOLO(str(best))

    # ------------------------------------------------------------------
    # Eval 1: v5 val + test (HONEST whole-pass split)
    # ------------------------------------------------------------------
    import yaml as _y
    with open(DATA_YAML) as f:
        cfg = _y.safe_load(f)
    class_names = list(cfg['names'].values())

    print('\n' + '=' * 60)
    print('EVAL: v5 val + test (per-class AP50)')
    print('=' * 60)
    for split in ['val', 'test']:
        r = m.val(data=str(DATA_YAML), imgsz=IMGSZ, conf=0.25, split=split, verbose=False)
        f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
        print(f'\n-- {split.upper()} --  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
              f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')
        print(f'{"Class":<18}{"AP50":>8}')
        print('-' * 28)
        for i, name in enumerate(class_names):
            print(f'{name:<18}{r.box.ap50[i]:>8.4f}')

    # ------------------------------------------------------------------
    # Eval 2: per-source test (SIH multi-sensor proof)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('PER-SOURCE TEST EVAL')
    print('=' * 60)
    rows = list(csv.DictReader(open(str(ROOT / 'datasets/sonarvision_multisource_v5/metadata.csv'))))
    src_root = OUT / 'src_eval'
    if src_root.exists():
        shutil.rmtree(src_root)
    src_root.mkdir(exist_ok=True)
    for src in ['NOAA', 'MILCO', 'KAGGLE']:
        d = src_root / src
        (d / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (d / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        for r_ in rows:
            if r_['split'] == 'test' and r_['image_path'].split('/')[-1].startswith(src + '_'):
                ip = ROOT / 'datasets/sonarvision_multisource_v5' / r_['image_path']
                lp = Path(str(ip).replace('/images/', '/labels/').replace('.png', '.txt'))
                (d / 'images' / 'test' / ip.name).symlink_to(ip)
                (d / 'labels' / 'test' / lp.name).symlink_to(lp)
        (d / 'data.yaml').write_text(
            f"path: {d}\ntrain: images/test\nval: images/test\ntest: images/test\n"
            f"nc: 4\nnames:\n" +
            ''.join(f'  {k}: {v}\n' for k, v in cfg['names'].items()))
        r = m.val(data=str(d / 'data.yaml'), imgsz=IMGSZ, conf=0.10, verbose=False)
        f1 = 2 * r.box.mp * r.box.mr / max(r.box.mp + r.box.mr, 1e-8)
        n_imgs = len([f for f in os.listdir(d / 'images' / 'test') if f.endswith('.png')])
        print(f'  {src:<8} test_imgs={n_imgs:<4} mAP50={r.box.map50:.4f}  '
              f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={f1:.4f}')

    # ------------------------------------------------------------------
    # Eval 3: h8_unseen_test (user's "unseen" reference, DISCLOSED caveat)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('EVAL: h8_unseen_test (debris = class 0)')
    print('=' * 60)
    unseen_yaml = OUT / 'h8_unseen_v5.yaml'
    unseen_yaml.write_text(
        f"path: {UNSEEN}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: 4\nnames:\n" +
        ''.join(f'  {k}: {v}\n' for k, v in cfg['names'].items()))
    r = m.val(data=str(unseen_yaml), imgsz=IMGSZ, conf=0.05, verbose=False)
    print(f'  (labels are class 0 = unknown_debris)')
    print(f'  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  '
          f'P={r.box.mp:.4f}  R={r.box.mr:.4f}  F1={2*r.box.mp*r.box.mr/max(r.box.mp+r.box.mr,1e-8):.4f}')
    print(f'  debris AP50 = {r.box.ap50[0]:.4f}')
    print('  ⚠ DISCLOSURE: 48/48 positives in this set come from passes')
    print('    present in h8_data — same-pass holdout, optimistic.')

    print('\nDONE. best.pt at:', best)
    print(f'total time budget: {EPOCHS} epochs @ {IMGSZ}px on MX550')
    print(f'NAME={NAME} IMGSZ={IMGSZ} BATCH={BATCH}')


if __name__ == '__main__':
    main()