#!/usr/bin/env python3
"""
SonarVision — Visual QA for Multi-Source Dataset
=================================================

Randomly samples images from train/val/test, draws YOLO bounding boxes,
and saves visual QA sheets for manual inspection.
"""

import os
import sys
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow required: pip install Pillow")
    sys.exit(1)

MASTER_CLASSES = {
    0: "unknown_debris",
    1: "airplane",
    2: "drowning_victim",
    3: "mine",
    4: "wreck",
}

COLORS = {
    0: (255, 0, 0),      # red
    1: (0, 255, 0),      # green
    2: (0, 0, 255),      # blue
    3: (255, 165, 0),    # orange
    4: (255, 0, 255),    # magenta
}


def draw_boxes(img_path: str, label_path: str) -> Image.Image:
    """Draw YOLO bounding boxes on image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORS.get(cls_id, (255, 255, 255))
            class_name = MASTER_CLASSES.get(cls_id, f"class_{cls_id}")

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{class_name} ({cls_id})"
            draw.text((x1, max(0, y1 - 12)), label, fill=color)

    return img


def create_qa_sheet(images, cols=4, cell_size=(300, 300), title="Visual QA"):
    """Create a grid of images with bounding boxes."""
    rows = (len(images) + cols - 1) // cols
    sheet_w = cols * cell_size[0]
    sheet_h = rows * cell_size[1] + 40  # extra for title

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    for idx, (img, label, source, split) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_size[0]
        y = row * cell_size[1] + 35

        try:
            qa_img = draw_boxes(img, label)
            qa_img = qa_img.resize(cell_size, Image.LANCZOS)
            sheet.paste(qa_img, (x, y))
        except Exception as e:
            draw.text((x + 10, y + 10), f"ERROR: {e}", fill=(255, 0, 0))

    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/sonarvision_multisource_v1")
    parser.add_argument("--output", default="reports/visual_qa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-group", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)

    dataset = args.dataset
    splits = ["train", "val", "test"]
    sources = ["NOAA", "MILCO", "KAGGLE"]

    for split in splits:
        img_dir = os.path.join(dataset, "images", split)
        lbl_dir = os.path.join(dataset, "labels", split)

        if not os.path.isdir(img_dir):
            continue

        # Collect all images with their source
        all_images = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            source = "UNKNOWN"
            for s in sources:
                if stem.startswith(s + "_"):
                    source = s
                    break
            all_images.append((os.path.join(img_dir, fname),
                              os.path.join(lbl_dir, stem + ".txt"),
                              source, split))

        if not all_images:
            continue

        # Sample from each source
        by_source = {}
        for img, lbl, src, sp in all_images:
            by_source.setdefault(src, []).append((img, lbl, src, sp))

        sampled = []
        for src, items in sorted(by_source.items()):
            n = min(args.samples_per_group, len(items))
            sampled.extend(rng.sample(items, n))

        if not sampled:
            continue

        title = f"Split: {split} | Sources: {', '.join(sorted(by_source.keys()))} | {len(sampled)} samples"
        sheet = create_qa_sheet(sampled, title=title)

        out_path = os.path.join(args.output, f"qa_{split}.png")
        sheet.save(out_path, quality=95)
        print(f"  Saved {out_path} ({len(sampled)} samples from {split})")

    print("Visual QA complete.")


if __name__ == "__main__":
    main()
