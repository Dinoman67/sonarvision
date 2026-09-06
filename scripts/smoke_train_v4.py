#!/usr/bin/env python3
"""
Smoke test: YOLOv8-ESI (SE attention) 6-class training on a mock subset of
the v4 dataset. Validates end-to-end BEFORE the long Colab run:
  - custom SEBlock/C2fWithSE wrapping survives ultralytics training
  - patch_trainer injects the custom model with DATA-DRIVEN nc/names (6)
  - all 6 classes train (per-class AP50 checked EVERY epoch via callback)
  - per-class eval on unseen v4 TEST images after the run
  - config parity with colab_train_v4_drive.py: cls_pw, SSS aug, lr0=0.01

Run:  .venv/bin/python scripts/smoke_train_v4.py
"""
import os
import csv
import glob
import random
import shutil
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect.train import DetectionTrainer

V4 = os.path.abspath("datasets/sonarvision_multisource_v4")
MOCK = os.path.abspath("datasets/mock_train_v4")
MOCK_TEST = os.path.abspath("datasets/mock_test_v4")
DATA = os.path.join(MOCK, "data.yaml")
BASE_WEIGHTS = os.path.abspath("yolov8n.pt")
PROJECT = os.path.abspath("runs/smoke")
IMGSZ = 320
BATCH = 6
EPOCHS = int(os.environ.get("SMOKE_EPOCHS", "6"))
# NOTE: ultralytics 8.4.x replaced the per-class list with automatic
# inverse-frequency weighting: cls_pw is a SCALAR power (0..1). 0.5 = sqrt
# inverse frequency, computed from the training set and mean-normalized.
# A per-class list crashes get_cfg() — do NOT use a list here or in Colab.
CLS_PW = 0.5
CLASSES = {0: "unknown_debris", 1: "airplane", 2: "drowning_victim",
           3: "mine", 4: "wreck", 5: "nombo_contact"}
TRAIN_PER_CLASS = 18
VAL_PER_CLASS = 8
TEST_PER_CLASS = 10


class SEBlock(nn.Module):
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


def build_yolov8_esi_full(nc=6, yaml_name="yolov8n.yaml", weights=BASE_WEIGHTS):
    """YOLOv8-ESI: SE attention after every C2f, built for `nc` classes.

    Two fixes vs the old builder (both matter for real training):
    1. Base is DetectionModel(yolov8n.yaml, nc=nc) — ultralytics builds the
       CORRECT nc-class Detect head. The old builder injected a COCO
       80-class head (bce_loss[80] x class_weights[nc] crashed, 75 wasted
       channels).
    2. Pretrained yolov8n.pt weights are loaded with SE key remapping: the
       C2fWithSE wrapper nests the C2f as `c2f.`, so plain intersect_dicts()
       matches NOTHING and every earlier ESI run silently trained the
       backbone FROM SCRATCH. Stripping the `c2f.` prefix restores transfer.
    """
    import re
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(yaml_name, nc=nc, ch=3)
    layers = _get_layers(model)
    for i, layer in enumerate(layers):
        if isinstance(layer, C2f):
            layers[i] = C2fWithSE(layer)
    _set_layers(model, layers)

    # load pretrained weights (head skipped: nc differs). The C2fWithSE
    # wrapper NESTS the C2f (`model.6.c2f.cv1...`), so target keys are
    # nested while checkpoint keys are flat — map TARGET -> CHECKPOINT.
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    csd = ckpt["model"].float().state_dict()
    msd = model.state_dict()
    remapped = {}
    for k in msd:
        m2 = re.match(r"^model\.(\d+)\.c2f\.(.*)$", k)
        ck = f"model.{m2.group(1)}.{m2.group(2)}" if m2 else k
        if ck in csd and tuple(csd[ck].shape) == tuple(msd[k].shape):
            remapped[k] = csd[ck]
    msd.update(remapped)
    model.load_state_dict(msd, strict=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"YOLOv8-ESI built: {total / 1e6:.2f}M params | head nc={nc} | "
          f"pretrained keys loaded: {len(remapped)}/{len(csd)} (SE-remapped)")
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


def boxes_of(lbl):
    if not os.path.exists(lbl):
        return []
    out = []
    for line in open(lbl).read().strip().splitlines():
        parts = line.split()
        if parts:
            out.append(int(parts[0]))
    return out


def build_subset():
    """Copy a small balanced subset of v4 (train/val from v4 train/val, plus
    an UNSEEN per-class test sample from v4 test)."""
    for root in [MOCK, MOCK_TEST]:
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(root, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(root, "labels", sp), exist_ok=True)
    rng = random.Random(7)

    counts = {"train": {}, "val": {}}
    for split in ["train", "val"]:
        meta = os.path.join(V4, f"images/{split}")
        # collect candidate images per class
        per_class = {c: [] for c in CLASSES}
        for img in sorted(os.listdir(meta)):
            stem = os.path.splitext(img)[0]
            lbl = os.path.join(V4, f"labels/{split}/{stem}.txt")
            cs = boxes_of(lbl)
            for c in set(cs):
                if c in per_class:
                    per_class[c].append(img)
        chosen = {}
        for c, imgs in per_class.items():
            rng.shuffle(imgs)
            n = TRAIN_PER_CLASS if split == "train" else VAL_PER_CLASS
            for img in imgs[:n]:
                chosen.setdefault(img, set()).add(c)
        for img, cs in chosen.items():
            stem = os.path.splitext(img)[0]
            shutil.copy2(os.path.join(V4, f"images/{split}/{img}"),
                         os.path.join(MOCK, f"images/{split}/{img}"))
            shutil.copy2(os.path.join(V4, f"labels/{split}/{stem}.txt"),
                         os.path.join(MOCK, f"labels/{split}/{stem}.txt"))
        counts[split] = {c: sum(1 for v in chosen.values() if c in v)
                         for c in CLASSES}

    # unseen test sample: 10 images per class from v4 TEST
    tcount = {}
    per_class = {c: [] for c in CLASSES}
    for img in sorted(os.listdir(os.path.join(V4, "images/test"))):
        stem = os.path.splitext(img)[0]
        cs = boxes_of(os.path.join(V4, f"labels/test/{stem}.txt"))
        for c in set(cs):
            if c in per_class:
                per_class[c].append(img)
    chosen = {}
    for c, imgs in per_class.items():
        rng.shuffle(imgs)
        for img in imgs[:TEST_PER_CLASS]:
            chosen.setdefault(img, set()).add(c)
    for img in chosen:
        stem = os.path.splitext(img)[0]
        shutil.copy2(os.path.join(V4, f"images/test/{img}"),
                     os.path.join(MOCK_TEST, f"images/test/{img}"))
        shutil.copy2(os.path.join(V4, f"labels/test/{stem}.txt"),
                     os.path.join(MOCK_TEST, f"labels/test/{stem}.txt"))
    tcount = {c: sum(1 for v in chosen.values() if c in v) for c in CLASSES}

    os.makedirs(MOCK, exist_ok=True)
    with open(DATA, "w") as f:
        f.write(f"path: {MOCK}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnc: {len(CLASSES)}\nnames:\n")
        for cid, cname in CLASSES.items():
            f.write(f"  {cid}: {cname}\n")

    print("Mock subset (images with >=1 box of each class):")
    print(f"  {'class':<18}{'train':>7}{'val':>7}{'test(unseen)':>14}")
    for c, name in CLASSES.items():
        print(f"  {name:<18}{counts['train'][c]:>7}{counts['val'][c]:>7}{tcount[c]:>14}")
    for sp in ["train", "val"]:
        n_img = len(glob.glob(f"{MOCK}/images/{sp}/*.png"))
        n_pos = sum(1 for f in glob.glob(f"{MOCK}/labels/{sp}/*.txt")
                    if os.path.getsize(f) > 0)
        print(f"  {sp}: {n_img} images ({n_pos} labeled)")


def per_class_eval(weights, data, tag, conf=0.001, imgsz=IMGSZ, count_dets=False):
    import yaml
    cfg = yaml.safe_load(open(data))
    m = YOLO(weights)
    r = m.val(data=data, imgsz=imgsz, conf=conf, verbose=False,
              project=PROJECT, name=f"eval_{tag.replace(' ', '_')}", exist_ok=True)
    ap = {cfg['names'][i]: float(r.box.ap50[i]) for i in range(len(cfg['names']))}
    print(f"[{tag}] mAP50={r.box.map50:.4f} P={r.box.mp:.4f} R={r.box.mr:.4f}")
    for name in cfg['names'].values():
        print(f"    {name:<18} AP50={ap[name]:.4f}")
    if count_dets:
        # count predicted boxes per class (raw detection activity)
        from collections import Counter
        import glob as _g
        d = Counter()
        for img_path in _g.glob(os.path.join(os.path.dirname(data), "images", "*", "*.png")):
            res = m.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
            for c in res.boxes.cls.tolist():
                d[cfg['names'][int(c)]] += 1
        print(f"    detections by class (conf>={conf}): {dict(d) if d else 'NONE'}")
    return ap


def main():
    import yaml
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)
    build_subset()

    cfg = yaml.safe_load(open(DATA))
    print(f"\nClasses ({cfg['nc']}): {cfg['names']}")
    assert cfg['nc'] == 6

    esi_obj = build_yolov8_esi_full()
    _orig = patch_trainer(esi_obj)

    # per-epoch per-class check: hook on_fit_epoch_end, evaluate current
    # best.pt on mock val so every class is monitored DURING the run
    def check_epoch(trainer):
        ep = trainer.epoch
        csv_path = os.path.join(trainer.save_dir, "results.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                print(f"\n--- EPOCH {ep} done: train/box={last.get('train/box_loss')} "
                      f"train/cls={last.get('train/cls_loss')} "
                      f"val/mAP50={last.get('metrics/mAP50(B)')} "
                      f"val/mAP50-95={last.get('metrics/mAP50-95(B)')} "
                      f"P={last.get('metrics/precision(B)')} R={last.get('metrics/recall(B)')}")
        best = os.path.join(trainer.save_dir, "weights", "best.pt")
        if os.path.exists(best):
            per_class_eval(best, DATA, f"val per-class @ep{ep}")

    model = YOLO(BASE_WEIGHTS)
    model.add_callback("on_fit_epoch_end", check_epoch)
    try:
        model.train(
            data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=99,
            lr0=0.01, lrf=0.1, warmup_epochs=1, cls_pw=CLS_PW,
            mosaic=0.0, mixup=0.0, fliplr=0.0, flipud=0.0, degrees=0.0,
            translate=0.05, scale=0.2,
            project=PROJECT, name='esi_v4_smoke', exist_ok=True,
            plots=False, verbose=False, workers=2, device=0,
        )
    finally:
        DetectionTrainer.get_model = _orig

    print("\n" + "=" * 70)
    print("AFTER RUN — per-class AP50 on mock val (best.pt)")
    print("=" * 70)
    best = os.path.join(PROJECT, 'esi_v4_smoke', 'weights', 'best.pt')
    val_ap = per_class_eval(best, DATA, "mock VAL", count_dets=True)

    print("\n" + "=" * 70)
    print("AFTER RUN — per-class AP50 on mock TRAIN (memorization check)")
    print("train AP50 > 0 for a class => the class IS being learned;")
    print("low VAL/TEST on tiny 6-epoch data is expected for small targets)")
    print("=" * 70)
    train_data = os.path.join(MOCK, "data_train_eval.yaml")
    with open(train_data, "w") as f:
        f.write(f"path: {MOCK}\ntrain: images/train\nval: images/train\n"
                f"test: images/train\nnc: {len(CLASSES)}\nnames:\n")
        for cid, cname in CLASSES.items():
            f.write(f"  {cid}: {cname}\n")
    train_ap = per_class_eval(best, train_data, "mock TRAIN", count_dets=True)

    print("\n" + "=" * 70)
    print("AFTER RUN — per-class AP50 on UNSEEN v4 TEST images")
    print("(images never seen by the mock run — real v4 test split)")
    print("=" * 70)
    test_data = os.path.join(MOCK_TEST, "data.yaml")
    with open(test_data, "w") as f:
        f.write(f"path: {MOCK_TEST}\ntrain: images/test\nval: images/test\n"
                f"test: images/test\nnc: {len(CLASSES)}\nnames:\n")
        for cid, cname in CLASSES.items():
            f.write(f"  {cid}: {cname}\n")
    test_ap = per_class_eval(best, test_data, "unseen TEST", count_dets=True)

    print("\n" + "=" * 70)
    print("CLASS SUMMARY (train mem / mock val / unseen test):")
    for name in cfg['names'].values():
        print(f"  {name:<18} train={train_ap[name]:.3f}  val={val_ap[name]:.3f}  "
              f"test={test_ap[name]:.3f}")
    zero = [n for n in cfg['names'].values() if test_ap[n] == 0 and val_ap[n] == 0]
    print(f"\nClasses with AP50=0 on BOTH val and test: {zero if zero else 'NONE'}")
    print("SMOKE COMPLETE")


if __name__ == "__main__":
    main()