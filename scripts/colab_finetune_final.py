"""
Sonar Vision — E3 Inference (FINAL CORRECTED)
================================================
Problem: COCO model detects debris as "sports ball" (class 37), but ground
truth is class 0 (marine_debris). model.val() shows 0 mAP due to class mismatch.

Fix: Train for 2 epochs to remap the head to class 0, while keeping backbone intact.
The COCO backbone already detects the objects — we just need the right class label.
"""

# =============================================================================
# ==== CELL 1: Install ====
# =============================================================================
# !pip install -q -U ultralytics
# import torch
# print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")

# =============================================================================
# ==== CELL 2: Upload + Fix paths ====
# =============================================================================
# from google.colab import files
# print("Upload: noaa_debris_e3_colab.zip")
# uploaded = files.upload()
# !unzip -qo noaa_debris_e3_colab.zip
#
# from pathlib import Path
# import yaml
# e3 = Path("e3")
# yaml_path = e3 / "data.yaml"
# data = yaml.safe_load(yaml_path.read_text())
# data["path"] = str(e3.resolve())
# yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
#
# for split in ["train", "val", "test"]:
#     imgs = list((e3 / "images" / split).glob("*.png"))
#     pos = sum(1 for f in (e3 / "labels" / split).glob("*.txt") if f.stat().st_size > 0)
#     neg = sum(1 for f in (e3 / "labels" / split).glob("*.txt") if f.stat().st_size == 0)
#     print(f"  {split:>5}: {len(imgs):3d} imgs | {pos:3d} pos + {neg:3d} neg")

# =============================================================================
# ==== CELL 3: Quick train — 2 epochs to remap class head ====
# =============================================================================
# WHY: The COCO backbone already detects debris. We just need to change the
# output from "sports ball" (class 37) to "marine_debris" (class 0).
# 2 epochs is enough to remap the head without destroying backbone features.
#
# from ultralytics import YOLO
#
# model = YOLO("yolo26s.pt")
#
# results = model.train(
#     data="e3/data.yaml",
#     imgsz=640,
#     epochs=2,              # Just 2 epochs — only remap the head
#     batch=16,
#     lr0=0.001,             # Moderate LR — enough to update head, not enough to break backbone
#     lrf=0.1,
#     optimizer="AdamW",
#     augment=True,
#     cache=True,
#     device=0,
#     project="runs",
#     name="E3_Remapped",
#     exist_ok=True,
#     plots=True,
#     seed=42,
#     workers=4,
# )
#
# print("✅ Done! Model now predicts class 0 (marine_debris)")

# =============================================================================
# ==== CELL 4: Benchmark the remapped model ====
# =============================================================================
# from ultralytics import YOLO
#
# model = YOLO("runs/E3_Remapped/weights/best.pt")
#
# for conf in [0.10, 0.05, 0.03, 0.01]:
#     for iou in [0.5, 0.3, 0.1]:
#         test = model.val(
#             data="e3/data.yaml", split="test",
#             imgsz=640, batch=16, device=0,
#             conf=conf, iou=iou, verbose=False,
#         )
#         if test.box.map50 > 0 or test.box.mr > 0:
#             print(f"  conf={conf} iou={iou} → P={test.box.mp:.4f} R={test.box.mr:.4f} mAP50={test.box.map50:.4f}")

# =============================================================================
# ==== CELL 5: Per-image analysis ====
# =============================================================================
# from ultralytics import YOLO
# from pathlib import Path
#
# model = YOLO("runs/E3_Remapped/weights/best.pt")
# root = Path("e3")
#
# for BEST_CONF in [0.05, 0.03, 0.01]:
#     print(f"\n{'='*70}")
#     print(f"conf={BEST_CONF}")
#     print(f"{'='*70}")
#     tp = fp = fn = 0
#     for img_path in sorted((root / "images" / "test").glob("*.png")):
#         lbl = root / "labels" / "test" / f"{img_path.stem}.txt"
#         has_gt = lbl.exists() and lbl.stat().st_size > 0
#         r = model.predict(str(img_path), imgsz=640, conf=BEST_CONF, verbose=False)[0]
#         n = len(r.boxes)
#         if has_gt and n > 0: tp += 1; s = "TP ✓"
#         elif has_gt: fn += 1; s = "FN ✗"
#         elif n > 0: fp += 1; s = "FP ⚠"
#         else: s = "TN ✓"
#         print(f"  {img_path.stem}: {s} dets={n}")
#     total = tp + fn
#     print(f"\nTP={tp} FP={fp} FN={fn} | Detection: {tp}/{total} = {tp/max(total,1)*100:.1f}%")

# =============================================================================
# ==== CELL 6: Save visual predictions ====
# =============================================================================
# from ultralytics import YOLO
# from pathlib import Path
#
# model = YOLO("runs/E3_Remapped/weights/best.pt")
# root = Path("e3")
#
# results = model.predict(
#     source=str(root / "images" / "test"),
#     imgsz=640, conf=0.03,
#     save=True, project="runs", name="E3_final_predictions", exist_ok=True,
# )
# print(f"✅ Saved {len(results)} images to runs/E3_final_predictions/")
