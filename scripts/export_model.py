#!/usr/bin/env python3
"""
YOLOv8-ESI — Production Export Pipeline
========================================

Exports trained model to ONNX in FP32/FP16/INT8 formats.
Validates each export on unseen test data.
Recommends best model for deployment.

Usage (in Colab):
    winner_path = '/content/runs/model_esi_stage2/weights/best.pt'
    H8_UNSEEN = '/content/h8_unseen_test/data.yaml'
    exec(open('scripts/export_model.py').read())

Usage (standalone):
    python scripts/export_model.py --model best.pt --data data.yaml
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

IMG_SIZE = 256
EXPORT_FORMATS = ['fp32', 'fp16', 'int8']


def get_size_mb(path):
    """Get file size in MB."""
    if path and Path(path).exists():
        return Path(path).stat().st_size / (1024 ** 2)
    return 0.0


def validate_model(model_path, data_yaml, img_size, name):
    """Validate a model and return metrics."""
    if not model_path or not Path(model_path).exists():
        return None

    print(f"\n  {'─'*60}")
    print(f"  VALIDATING: {name}")
    print(f"  {'─'*60}")

    try:
        m = YOLO(str(model_path))
        metrics = m.val(
            data=str(data_yaml),
            imgsz=img_size,
            split="test",
            verbose=False
        )

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        result = {
            "Model": name,
            "Size_MB": get_size_mb(model_path),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

        print(f"    mAP50      : {result['mAP50']:.4f}")
        print(f"    mAP50-95   : {result['mAP50-95']:.4f}")
        print(f"    Precision  : {result['Precision']:.4f}")
        print(f"    Recall     : {result['Recall']:.4f}")
        print(f"    F1         : {result['F1']:.4f}")
        print(f"    Size       : {result['Size_MB']:.2f} MB")

        return result

    except Exception as e:
        print(f"    ❌ Validation failed: {e}")
        return None


def export_onnx(model, img_size, export_dir, fmt):
    """Export model to ONNX format."""
    print(f"\n  {'─'*60}")
    print(f"  EXPORTING: ONNX {fmt.upper()}")
    print(f"  {'─'*60}")

    try:
        kwargs = {
            'format': 'onnx',
            'imgsz': img_size,
            'simplify': True,
            'dynamic': False
        }

        if fmt == 'fp16':
            kwargs['half'] = True
        elif fmt == 'int8':
            kwargs['int8'] = True

        result = model.export(**kwargs)

        # Copy to export directory
        source = Path(result)
        dest = export_dir / f'yolo_esi_{fmt}.onnx'
        shutil.copy2(source, dest)

        size = get_size_mb(dest)
        print(f"    Created: {dest}")
        print(f"    Size: {size:.2f} MB")

        return dest

    except Exception as e:
        print(f"    ❌ Export failed: {e}")
        print(f"       Error: {str(e)[:100]}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8-ESI model')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=256, help='Image size')
    parser.add_argument('--export-dir', type=str, default='/content/yolo_esi_exports',
                        help='Export directory')
    args = parser.parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)
    img_size = args.imgsz
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    print("=" * 70)
    print("YOLOv8-ESI PRODUCTION EXPORT PIPELINE")
    print("=" * 70)

    print(f"\n  Master model : {model_path}")
    print(f"  Test dataset : {data_yaml}")
    print(f"  Image size   : {img_size}")
    print(f"  Export dir   : {export_dir}")

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Load model
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: LOAD MASTER MODEL")
    print("=" * 70)

    model = YOLO(str(model_path))
    original_size = get_size_mb(model_path)

    print(f"  Loaded: {model_path}")
    print(f"  Size: {original_size:.2f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Validate master .pt model
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATE MASTER .PT MODEL")
    print("=" * 70)

    baseline = validate_model(model_path, data_yaml, img_size, "YOLOv8-ESI .pt")

    if not baseline:
        print("\n❌ Cannot validate master model. Aborting.")
        return

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Export all formats
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: EXPORT ONNX MODELS")
    print("=" * 70)

    exports = {}
    for fmt in EXPORT_FORMATS:
        path = export_onnx(model, img_size, export_dir, fmt)
        if path:
            exports[fmt] = path

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Validate all exports
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: VALIDATE ALL EXPORTS")
    print("=" * 70)

    results = [baseline]

    for fmt, path in exports.items():
        name = f"ONNX {fmt.upper()}"
        result = validate_model(path, data_yaml, img_size, name)
        if result:
            results.append(result)

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Comparison table
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(df.round(4).to_string(index=False))

    # ═══════════════════════════════════════════════════════════
    # STEP 6: Accuracy change from baseline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ACCURACY CHANGE FROM MASTER .PT")
    print("=" * 70)

    base_map50 = baseline['mAP50']
    base_f1 = baseline['F1']
    base_recall = baseline['Recall']

    for _, row in df.iloc[1:].iterrows():
        map50_change = (row['mAP50'] - base_map50) * 100
        f1_change = (row['F1'] - base_f1) * 100
        recall_change = (row['Recall'] - base_recall) * 100

        print(f"  {row['Model']:<20} "
              f"mAP50: {map50_change:+.2f}pp | "
              f"F1: {f1_change:+.2f}pp | "
              f"Recall: {recall_change:+.2f}pp")

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Recommendation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    exported = df[df['Model'] != 'YOLOv8-ESI .pt']

    if len(exported) > 0:
        best_accuracy = exported.loc[exported['mAP50'].idxmax()]
        best_f1 = exported.loc[exported['F1'].idxmax()]
        smallest = exported.loc[exported['Size_MB'].idxmin()]

        print(f"\n  🏆 Best Accuracy : {best_accuracy['Model']} "
              f"(mAP50={best_accuracy['mAP50']:.4f})")
        print(f"  🎯 Best F1       : {best_f1['Model']} "
              f"(F1={best_f1['F1']:.4f})")
        print(f"  📦 Smallest      : {smallest['Model']} "
              f"({smallest['Size_MB']:.2f} MB)")

        # Size reduction
        fp16_row = exported[exported['Model'] == 'ONNX FP16']
        if len(fp16_row) > 0:
            fp16_size = fp16_row['Size_MB'].values[0]
            reduction = (1 - fp16_size / original_size) * 100
            print(f"\n  📉 FP16 Size Reduction: {reduction:.0f}% smaller")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: File listing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPORTED FILES")
    print("=" * 70)

    print(f"\n  Master (DO NOT MODIFY):")
    print(f"    {model_path}")

    print(f"\n  Exported models:")
    for fmt, path in exports.items():
        size = get_size_mb(path)
        print(f"    {path} ({size:.2f} MB)")

    # ═══════════════════════════════════════════════════════════
    # STEP 9: Deployment instructions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DEPLOYMENT INSTRUCTIONS")
    print("=" * 70)

    print("""
  ONNX RUNTIME (Laptop / Raspberry Pi):
  ─────────────────────────────────────
    pip install onnxruntime opencv-python numpy

    import onnxruntime as ort
    import cv2
    import numpy as np

    session = ort.InferenceSession('yolo_esi_fp16.onnx')
    # ... (see detect.py for full code)

  ULTRALYTICS (Laptop):
  ─────────────────────
    from ultralytics import YOLO
    model = YOLO('best.pt')
    results = model.predict(source='test_image.png')
""")

    print("=" * 70)
    print("✅ EXPORT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
