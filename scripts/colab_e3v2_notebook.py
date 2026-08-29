#!/usr/bin/env python3
"""
Copy-paste ready Colab cells for E3 V2 training.
Each section is meant to be a separate Colab cell.
"""

# ============================================================
# CELL 1: Setup & Upload
# ============================================================
# === Run this first ===

# Install dependencies
!pip install ultralytics pandas matplotlib -q

# Upload dataset
from google.colab import files
print("Upload e3_v2.zip from your machine:")
uploaded = files.upload()

# Unzip
!unzip -q e3_v2.zip -d /content/
!ls -la /content/e3_v2/

# Verify dataset
import os
for split in ['train', 'val', 'test']:
    img_dir = f'/content/e3_v2/images/{split}'
    lbl_dir = f'/content/e3_v2/labels/{split}'
    n_img = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    n_pos = sum(1 for f in os.listdir(lbl_dir) if f.endswith('.txt') and os.path.getsize(os.path.join(lbl_dir, f)) > 0)
    print(f'{split}: {n_img} images, {n_pos} positive, {n_img - n_pos} negative')


# ============================================================
# CELL 2: Quick Sanity Check
# ============================================================
# === Verify labels are correct before training ===

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Show a few training examples with bboxes
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
train_imgs = sorted([f for f in os.listdir('/content/e3_v2/images/train') if f.endswith('.png')])

# Show mix of positive and negative
pos_imgs = [f for f in train_imgs if f.startswith('E3v2_TGT')]
neg_imgs = [f for f in train_imgs if f.startswith('E3v2_BG')]
show_imgs = pos_imgs[:4] + neg_imgs[:4]

for i, img_name in enumerate(show_imgs):
    ax = axes[i // 4, i % 4]
    img = np.array(Image.open(f'/content/e3_v2/images/train/{img_name}'))
    ax.imshow(img, cmap='gray')

    lbl_path = f'/content/e3_v2/labels/train/{os.path.splitext(img_name)[0]}.txt'
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, w, h = map(float, parts)
                    x1 = (xc - w/2) * 512
                    y1 = (yc - h/2) * 512
                    rect = plt.Rectangle((x1, y1), w*512, h*512,
                                        fill=False, edgecolor='lime', linewidth=2)
                    ax.add_patch(rect)
        ax.set_title(os.path.splitext(img_name)[0][:20], color='green', fontsize=8)
    else:
        ax.set_title('BG (no label)', color='red', fontsize=8)
    ax.axis('off')

plt.suptitle('Training Examples: Green=bbox, Red=background', fontsize=12)
plt.tight_layout()
plt.savefig('/content/sanity_check.png', dpi=150)
plt.show()
print("Check: do the green boxes align with bright spots in the images?")


# ============================================================
# CELL 3: Train YOLOv8s
# ============================================================
# === Main training cell ===

from ultralytics import YOLO
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Load pretrained model
model = YOLO('yolov8s.pt')

# Train
results = model.train(
    data='/content/e3_v2/data_colab.yaml',
    epochs=100,
    imgsz=512,
    batch=16,
    patience=20,         # early stopping
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    # Augmentation
    mosaic=1.0,
    mixup=0.0,           # keep 0 for small datasets
    fliplr=0.5,
    flipud=0.0,          # sonar has orientation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,         # no rotation
    translate=0.1,
    scale=0.5,
    # Output
    name='e3_v2_yolov8s',
    project='/content/runs',
    exist_ok=True,
    plots=True,
)

print(f"\n✓ Training complete!")
print(f"Best weights: /content/runs/e3_v2_yolov8s/weights/best.pt")


# ============================================================
# CELL 4: Plot Training Curves
# ============================================================
# === Visualize training progress ===

import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv('/content/runs/e3_v2_yolov8s/results.csv')
results_df.columns = results_df.columns.str.strip()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loss curves
axes[0].plot(results_df['epoch'], results_df['train/box_loss'], 'b-', label='Box Loss')
axes[0].plot(results_df['epoch'], results_df['train/cls_loss'], 'r-', label='Cls Loss')
axes[0].plot(results_df['epoch'], results_df['train/dfl_loss'], 'g-', label='DFL Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# mAP curves
axes[1].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], 'b-', label='mAP50')
axes[1].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], 'r-', label='mAP50-95')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('mAP')
axes[1].set_title('Validation mAP')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Precision/Recall
axes[2].plot(results_df['epoch'], results_df['metrics/precision(B)'], 'b-', label='Precision')
axes[2].plot(results_df['epoch'], results_df['metrics/recall(B)'], 'r-', label='Recall')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Score')
axes[2].set_title('Precision & Recall')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/training_curves.png', dpi=150)
plt.show()

# Print best epoch
best_epoch = results_df['metrics/mAP50(B)'].idxmax()
print(f"Best epoch: {int(results_df.iloc[best_epoch]['epoch'])}")
print(f"Best mAP50: {results_df.iloc[best_epoch]['metrics/mAP50(B)']:.4f}")
print(f"Best mAP50-95: {results_df.iloc[best_epoch]['metrics/mAP50-95(B)']:.4f}")


# ============================================================
# CELL 5: Evaluate on Validation Set
# ============================================================
# === Detailed validation evaluation ===

model = YOLO('/content/runs/e3_v2_yolov8s/weights/best.pt')

results = model.val(
    data='/content/e3_v2/data_colab.yaml',
    split='val',
    imgsz=512,
    conf=0.25,
    save_json=True,
    workers=2,
)

print(f"\n=== VALIDATION SET ===")
print(f"mAP50:    {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")
f1 = 2 * results.box.mp * results.box.mr / max(results.box.mp + results.box.mr, 1e-8)
print(f"F1:        {f1:.4f}")


# ============================================================
# CELL 6: Evaluate on UNSEEN Test Set
# ============================================================
# === This is the real test — completely held-out data ===

results_test = model.val(
    data='/content/e3_v2/data_colab.yaml',
    split='test',
    imgsz=512,
    conf=0.25,
    save_json=True,
    workers=2,
)

print(f"\n=== TEST SET (UNSEEN DATA) ===")
print(f"mAP50:    {results_test.box.map50:.4f}")
print(f"mAP50-95: {results_test.box.map:.4f}")
print(f"Precision: {results_test.box.mp:.4f}")
print(f"Recall:    {results_test.box.mr:.4f}")
f1_test = 2 * results_test.box.mp * results_test.box.mr / max(results_test.box.mp + results_test.box.mr, 1e-8)
print(f"F1:        {f1_test:.4f}")


# ============================================================
# CELL 7: Confidence Threshold Sweep
# ============================================================
# === Find optimal confidence threshold ===

sweep_results = []
for conf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    r = model.val(data='/content/e3_v2/data_colab.yaml', split='val',
                  imgsz=512, conf=conf, verbose=False, workers=2)
    p, r_val, m = r.box.mp, r.box.mr, r.box.map50
    f1 = 2 * p * r_val / max(p + r_val, 1e-8)
    sweep_results.append({'conf': conf, 'P': p, 'R': r_val, 'mAP50': m, 'F1': f1})

import pandas as pd
sweep_df = pd.DataFrame(sweep_results)
print(sweep_df.to_string(index=False))

best = sweep_df.loc[sweep_df['F1'].idxmax()]
print(f"\nBest F1: conf={best['conf']:.2f}, F1={best['F1']:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep_df['conf'], sweep_df['P'], 'b-o', label='Precision')
ax.plot(sweep_df['conf'], sweep_df['R'], 'r-o', label='Recall')
ax.plot(sweep_df['conf'], sweep_df['F1'], 'g-o', label='F1', linewidth=2)
ax.axvline(best['conf'], color='gray', linestyle='--', alpha=0.5, label=f"Best ({best['conf']:.2f})")
ax.set_xlabel('Confidence Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision/Recall/F1 vs Confidence Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/confidence_sweep.png', dpi=150)
plt.show()


# ============================================================
# CELL 8: Per-Target Detection Analysis
# ============================================================
# === Which targets does the model detect well? ===

test_dir = '/content/e3_v2/images/test'
test_lbl_dir = '/content/e3_v2/labels/test'

# Group by target
target_files = {}
for f in sorted(os.listdir(test_dir)):
    if not f.endswith('.png'):
        continue
    parts = f.replace('.png', '').split('_')
    tid = 'BG'
    for p in parts:
        if p.startswith('TGT'):
            tid = p
            break
    if tid not in target_files:
        target_files[tid] = []
    target_files[tid].append(f)

# Run inference
pred_results = model.predict(source=test_dir, imgsz=512, conf=best['conf'],
                             save=False, verbose=False)

# Map predictions to targets
target_dets = {}
for r in pred_results:
    img_name = os.path.basename(str(r.path))
    parts = img_name.replace('.png', '').split('_')
    tid = 'BG'
    for p in parts:
        if p.startswith('TGT'):
            tid = p
            break
    if tid not in target_dets:
        target_dets[tid] = {'detected': 0, 'total': 0, 'confs': []}
    target_dets[tid]['total'] += 1
    if len(r.boxes) > 0:
        target_dets[tid]['detected'] += 1
        target_dets[tid]['confs'].extend([float(c) for c in r.boxes.conf])

print(f"{'Target':>10} {'Images':>7} {'Detected':>9} {'Rate':>8} {'Avg Conf':>9}")
print("-" * 50)
for tid in sorted(target_dets.keys()):
    s = target_dets[tid]
    rate = s['detected'] / max(s['total'], 1)
    avg_c = np.mean(s['confs']) if s['confs'] else 0
    bar = '█' * int(rate * 20) + '░' * (20 - int(rate * 20))
    print(f"{tid:>10} {s['total']:>7} {s['detected']:>9} {bar} {rate:>7.0%} {avg_c:>8.3f}")


# ============================================================
# CELL 9: Visual Predictions on Test Set
# ============================================================
# === Show model predictions on test images ===

pred_results = model.predict(source=test_dir, imgsz=512, conf=best['conf'],
                             save=True, project='/content/runs', name='test_predictions',
                             exist_ok=True, workers=2)

# Show a grid of predictions
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
pred_dir = '/content/runs/test_predictions'
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.jpg')])[:12]

for i, fname in enumerate(pred_files):
    ax = axes[i // 4, i % 4]
    img = np.array(Image.open(os.path.join(pred_dir, fname)))
    ax.imshow(img)
    ax.set_title(fname[:25], fontsize=8)
    ax.axis('off')

plt.suptitle('Test Set Predictions', fontsize=14)
plt.tight_layout()
plt.savefig('/content/test_predictions_grid.png', dpi=150)
plt.show()


# ============================================================
# CELL 10: Export Model
# ============================================================
# === Export to ONNX for deployment ===

onnx_path = model.export(format='onnx', imgsz=512, simplify=True)
print(f"✓ ONNX exported: {onnx_path}")

# Download the model
files.download('/content/runs/e3_v2_yolov8s/weights/best.pt')
files.download(onnx_path)


# ============================================================
# CELL 11: Save Everything
# ============================================================
# === Package results for download ===

import json

report = {
    "model": "yolov8s",
    "dataset": "E3 V2 (NOAA H11833 SSS Marine Debris)",
    "epochs": 100,
    "val_results": {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    },
    "test_results": {
        "mAP50": float(results_test.box.map50),
        "mAP50-95": float(results_test.box.map),
        "precision": float(results_test.box.mp),
        "recall": float(results_test.box.mr),
    },
    "best_confidence": float(best['conf']),
    "per_target": {k: {'detected': v['detected'], 'total': v['total']}
                   for k, v in target_dets.items()},
}

with open('/content/training_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Download everything
!zip -r /content/e3_v2_results.zip /content/runs/e3_v2_yolov8s/ /content/training_report.json /content/training_curves.png /content/confidence_sweep.png
files.download('/content/e3_v2_results.zip')

print("\n✓ All results saved and ready for download!")
print(f"\n=== FINAL SUMMARY ===")
print(f"Model: yolov8s (trained 100 epochs)")
print(f"Val mAP50: {results.box.map50:.4f}")
print(f"Test mAP50: {results_test.box.map50:.4f} (UNSEEN)")
print(f"Best conf: {best['conf']:.2f}")
