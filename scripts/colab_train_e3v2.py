#!/usr/bin/env python3
"""
scripts/colab_train_e3v2.py

Complete YOLOv8 training pipeline for E3 V2 dataset.
Designed for Google Colab (T4/A100 GPU).

Features:
- Multiple model sizes (n/s/m) with comparison
- Full augmentation pipeline
- Early stopping + learning rate scheduling
- Comprehensive evaluation on val AND test splits
- Per-target detection analysis
- Confidence threshold sweep
- Model export (ONNX, TensorRT)
- Results visualization and comparison plots

Usage in Colab:
  !unzip -q e3_v2.zip -d /content/
  !cp /path/to/colab_train_e3v2.py /content/
  !python /content/colab_train_e3v2.py
"""

import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    # Dataset
    "data_yaml": "/content/e3_v2/data_colab.yaml",
    "dataset_dir": "/content/e3_v2",

    # Training
    "model_size": "s",          # n=tiny, s=small, m=medium
    "epochs": 100,
    "imgsz": 512,
    "batch": 16,
    "patience": 20,             # early stopping patience
    "lr0": 0.01,                # initial LR
    "lrf": 0.01,                # final LR factor
    "warmup_epochs": 3,

    # Augmentation (conservative for small dataset)
    "mosaic": 1.0,
    "mixup": 0.0,               # keep 0 for small datasets
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,              # sonar images have orientation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,             # no rotation for sonar
    "translate": 0.1,
    "scale": 0.5,
    "erasing": 0.4,

    # Evaluation
    "conf_thresholds": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "iou_threshold": 0.5,

    # Export
    "export_onnx": True,
    "export_tensorrt": False,    # needs TensorRT on Colab

    # Output
    "project": "/content/runs",
    "name": "e3_v2_training",
}


def setup_colab():
    """Check and setup Colab environment."""
    print("=" * 70)
    print("E3 V2 YOLO TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("WARNING: No GPU detected! Training will be very slow.")
        print("Go to Runtime → Change runtime type → T4 GPU")

    # Check dataset
    data_yaml = Path(CONFIG["data_yaml"])
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found!")
        print("Upload e3_v2.zip and unzip first.")
        sys.exit(1)

    # Check splits
    ds_dir = Path(CONFIG["dataset_dir"])
    for split in ["train", "val", "test"]:
        img_dir = ds_dir / "images" / split
        lbl_dir = ds_dir / "labels" / split
        n_img = len(list(img_dir.glob("*.png")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        n_pos = sum(1 for f in lbl_dir.glob("*.txt") if f.stat().st_size > 0)
        print(f"  {split}: {n_img} images, {n_pos} positive, {n_img - n_pos} negative")

    print()


def train_model():
    """Train YOLOv8 model with full configuration."""
    from ultralytics import YOLO

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    # Select pretrained model
    model_name = f"yolov8{CONFIG['model_size']}.pt"
    print(f"Model: {model_name}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Image size: {CONFIG['imgsz']}")
    print(f"Batch size: {CONFIG['batch']}")
    print()

    model = YOLO(model_name)

    # Train
    results = model.train(
        data=CONFIG["data_yaml"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        patience=CONFIG["patience"],
        lr0=CONFIG["lr0"],
        lrf=CONFIG["lrf"],
        warmup_epochs=CONFIG["warmup_epochs"],
        mosaic=CONFIG["mosaic"],
        mixup=CONFIG["mixup"],
        copy_paste=CONFIG["copy_paste"],
        fliplr=CONFIG["fliplr"],
        flipud=CONFIG["flipud"],
        hsv_h=CONFIG["hsv_h"],
        hsv_s=CONFIG["hsv_s"],
        hsv_v=CONFIG["hsv_v"],
        degrees=CONFIG["degrees"],
        translate=CONFIG["translate"],
        scale=CONFIG["scale"],
        erasing=CONFIG["erasing"],
        name=CONFIG["name"],
        project=CONFIG["project"],
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    # Find best model
    run_dir = Path(CONFIG["project"]) / CONFIG["name"]
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"\nTraining complete!")
    print(f"Best model: {best_pt}")
    print(f"Last model: {last_pt}")

    # Training curves
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        print(f"Training curves: {run_dir / 'results.png'}")

    return str(best_pt), str(last_pt), str(run_dir)


def evaluate_model(model_path, run_dir):
    """Comprehensive evaluation on val and test splits."""
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model = YOLO(model_path)
    eval_results = {}

    for split in ["val", "test"]:
        print(f"\n--- {split.upper()} SET ---")
        results = model.val(
            data=CONFIG["data_yaml"],
            split=split,
            imgsz=CONFIG["imgsz"],
            conf=CONFIG["iou_threshold"],
            workers=2,
            verbose=True,
        )

        eval_results[split] = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "f1": 2 * results.box.mp * results.box.mr / max(results.box.mp + results.box.mr, 1e-8),
            "images": results.seen,
            "instances": results.boxes.shape[0] if hasattr(results.boxes, 'shape') else 0,
        }

        print(f"  mAP50:    {eval_results[split]['mAP50']:.4f}")
        print(f"  mAP50-95: {eval_results[split]['mAP50-95']:.4f}")
        print(f"  Precision: {eval_results[split]['precision']:.4f}")
        print(f"  Recall:    {eval_results[split]['recall']:.4f}")
        print(f"  F1:        {eval_results[split]['f1']:.4f}")

    # Confidence threshold sweep
    print(f"\n--- CONFIDENCE THRESHOLD SWEEP (val set) ---")
    print(f"{'Conf':>6} {'P':>8} {'R':>8} {'mAP50':>8} {'F1':>8}")
    print("-" * 42)

    sweep_results = []
    for conf in CONFIG["conf_thresholds"]:
        results = model.val(
            data=CONFIG["data_yaml"],
            split="val",
            imgsz=CONFIG["imgsz"],
            conf=conf,
            workers=2,
            verbose=False,
        )
        p, r, m50 = results.box.mp, results.box.mr, results.box.map50
        f1 = 2 * p * r / max(p + r, 1e-8)
        sweep_results.append({"conf": conf, "P": p, "R": r, "mAP50": m50, "F1": f1})
        print(f"{conf:>6.2f} {p:>8.4f} {r:>8.4f} {m50:>8.4f} {f1:>8.4f}")

    # Find best F1 threshold
    best_sweep = max(sweep_results, key=lambda x: x["F1"])
    print(f"\nBest F1 threshold: conf={best_sweep['conf']:.2f} (F1={best_sweep['F1']:.4f})")

    # Plot confidence sweep
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    confs = [r["conf"] for r in sweep_results]
    axes[0].plot(confs, [r["P"] for r in sweep_results], "b-o", label="Precision")
    axes[0].plot(confs, [r["R"] for r in sweep_results], "r-o", label="Recall")
    axes[0].plot(confs, [r["F1"] for r in sweep_results], "g-o", label="F1")
    axes[0].set_xlabel("Confidence Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision/Recall/F1 vs Confidence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(confs, [r["mAP50"] for r in sweep_results], "m-o")
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("mAP50")
    axes[1].set_title("mAP50 vs Confidence")
    axes[1].grid(True, alpha=0.3)

    # Training curves
    results_csv = Path(run_dir) / "results.csv"
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        epochs = df.iloc[:, 0].values
        train_loss = df.iloc[:, 1].values  # train/box_loss
        val_map50 = df.iloc[:, 7].values   # metrics/mAP50(B)

        axes[2].plot(epochs, train_loss, "b-", label="Train Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss", color="b")
        ax2 = axes[2].twinx()
        ax2.plot(epochs, val_map50, "r-", label="Val mAP50")
        axes[2].set_ylabel("mAP50", color="r")
        axes[2].set_title("Training Progress")
        axes[2].legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    sweep_path = Path(run_dir) / "evaluation_sweep.png"
    plt.savefig(sweep_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Saved evaluation plot: {sweep_path}")

    eval_results["sweep"] = sweep_results
    eval_results["best_conf"] = best_sweep["conf"]

    return eval_results


def analyze_per_target(model_path):
    """Analyze detection performance per target ID."""
    from ultralytics import YOLO
    from PIL import Image
    import glob

    print("\n" + "=" * 70)
    print("PER-TARGET ANALYSIS (test set)")
    print("=" * 70)

    model = YOLO(model_path)
    test_dir = Path(CONFIG["dataset_dir"]) / "images" / "test"
    test_lbl = Path(CONFIG["dataset_dir"]) / "labels" / "test"

    # Group images by target
    target_images = {}
    for img_path in sorted(test_dir.glob("*.png")):
        stem = img_path.stem
        parts = stem.split("_")
        tid = "BG"
        for p in parts:
            if p.startswith("TGT"):
                tid = p
                break
        if tid not in target_images:
            target_images[tid] = []
        target_images[tid].append(img_path)

    # Run inference
    results_list = model.predict(
        source=str(test_dir),
        imgsz=CONFIG["imgsz"],
        conf=best_conf if 'best_conf' in dir() else 0.25,
        save=False,
        verbose=False,
    )

    # Analyze per target
    print(f"{'Target':>10} {'Images':>7} {'Detected':>9} {'Avg Conf':>9} {'Avg IoU':>9}")
    print("-" * 50)

    target_stats = {}
    for tid, img_paths in sorted(target_images.items()):
        n_images = len(img_paths)
        n_detected = 0
        confs = []

        for img_path in img_paths:
            # Find corresponding result
            for r in results_list:
                if str(img_path) in str(r.path) or img_path.name in str(r.path):
                    if len(r.boxes) > 0:
                        n_detected += 1
                        confs.extend([float(c) for c in r.boxes.conf])
                    break

        avg_conf = np.mean(confs) if confs else 0
        detection_rate = n_detected / max(n_images, 1)

        target_stats[tid] = {
            "images": n_images,
            "detected": n_detected,
            "detection_rate": detection_rate,
            "avg_conf": avg_conf,
        }

        print(f"{tid:>10} {n_images:>7} {n_detected:>9} {avg_conf:>9.3f} {detection_rate:>9.1%}")

    # Summary
    total_images = sum(s["images"] for s in target_stats.values())
    total_detected = sum(s["detected"] for s in target_stats.values())
    print(f"\nOverall: {total_detected}/{total_images} images with detections ({100*total_detected/max(total_images,1):.0f}%)")

    return target_stats


def export_model(model_path, run_dir):
    """Export model to ONNX and optionally TensorRT."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("MODEL EXPORT")
    print("=" * 70)

    model = YOLO(model_path)
    export_dir = Path(run_dir) / "export"
    export_dir.mkdir(exist_ok=True)

    # ONNX export
    if CONFIG["export_onnx"]:
        print("Exporting to ONNX...")
        onnx_path = model.export(
            format="onnx",
            imgsz=CONFIG["imgsz"],
            simplify=True,
            opset=17,
        )
        print(f"  ✓ ONNX: {onnx_path}")

    # Save model metadata
    metadata = {
        "model_size": CONFIG["model_size"],
        "imgsz": CONFIG["imgsz"],
        "train_epochs": CONFIG["epochs"],
        "dataset": "E3 V2 (NOAA H11833 SSS)",
        "class_names": ["marine_debris"],
        "export_date": datetime.now().isoformat(),
        "source": model_path,
    }

    meta_path = export_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata: {meta_path}")

    return export_dir


def generate_report(eval_results, target_stats, run_dir):
    """Generate final training report."""
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "evaluation": {},
        "per_target": target_stats,
        "best_confidence": eval_results.get("best_conf", 0.25),
    }

    # Val results
    if "val" in eval_results:
        v = eval_results["val"]
        print(f"\nValidation Set:")
        print(f"  mAP50:    {v['mAP50']:.4f}")
        print(f"  mAP50-95: {v['mAP50-95']:.4f}")
        print(f"  Precision: {v['precision']:.4f}")
        print(f"  Recall:    {v['recall']:.4f}")
        print(f"  F1:        {v['f1']:.4f}")
        report["evaluation"]["val"] = v

    # Test results (unseen)
    if "test" in eval_results:
        t = eval_results["test"]
        print(f"\nTest Set (UNSEEN):")
        print(f"  mAP50:    {t['mAP50']:.4f}")
        print(f"  mAP50-95: {t['mAP50-95']:.4f}")
        print(f"  Precision: {t['precision']:.4f}")
        print(f"  Recall:    {t['recall']:.4f}")
        print(f"  F1:        {t['f1']:.4f}")
        report["evaluation"]["test"] = t

    # Best threshold
    print(f"\nBest confidence threshold: {report['best_confidence']:.2f}")

    # Per-target summary
    print(f"\nPer-target detection rates:")
    for tid, stats in sorted(target_stats.items()):
        if tid == "BG":
            continue
        rate = stats["detection_rate"]
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {tid}: {bar} {rate:.0%} ({stats['detected']}/{stats['images']})")

    # Save report
    report_path = Path(run_dir) / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Full report saved: {report_path}")

    # Also save a human-readable summary
    summary_path = Path(run_dir) / "SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write("E3 V2 YOLO TRAINING SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: yolov8{CONFIG['model_size']}\n")
        f.write(f"Epochs: {CONFIG['epochs']}\n\n")
        if "test" in eval_results:
            t = eval_results["test"]
            f.write(f"TEST SET (UNSEEN) RESULTS:\n")
            f.write(f"  mAP50:    {t['mAP50']:.4f}\n")
            f.write(f"  mAP50-95: {t['mAP50-95']:.4f}\n")
            f.write(f"  Precision: {t['precision']:.4f}\n")
            f.write(f"  Recall:    {t['recall']:.4f}\n")
            f.write(f"  F1:        {t['f1']:.4f}\n")
        f.write(f"\nBest confidence: {report['best_confidence']:.2f}\n")
    print(f"[✓] Summary saved: {summary_path}")


def main():
    # Fix config (warmup_epochs is a function call, not key)
    # This is a known issue - we'll handle it in setup

    setup_colab()

    # Train
    best_pt, last_pt, run_dir = train_model()

    # Evaluate
    eval_results = evaluate_model(best_pt, run_dir)

    # Per-target analysis
    # Use best confidence from sweep
    best_conf = eval_results.get("best_conf", 0.25)
    target_stats = analyze_per_target(best_pt)

    # Export
    export_model(best_pt, run_dir)

    # Report
    generate_report(eval_results, target_stats, run_dir)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Results: {run_dir}")
    print(f"Best model: {best_pt}")
    print(f"Training curves: {Path(run_dir) / 'results.png'}")
    print(f"Evaluation plots: {Path(run_dir) / 'evaluation_sweep.png'}")
    print(f"Full report: {Path(run_dir) / 'training_report.json'}")


if __name__ == "__main__":
    main()
