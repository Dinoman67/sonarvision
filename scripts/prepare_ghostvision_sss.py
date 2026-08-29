#!/usr/bin/env python3
"""
scripts/prepare_ghostvision_sss.py

Extracts and converts ONLY the GhostVision side-scan-sonar dataset from
GhostVision_DatasetAndModels.zip into a clean YOLO object-detection dataset.

Outputs:
- ~/sonar-vision/datasets/ghostvision_sss_yolo/
  ├── images/ (train, val, test)
  ├── labels/ (train, val, test)
  ├── data.yaml
  ├── metadata/conversion_manifest.csv
  └── qa/ (visual QA overlay samples)
"""

import os
import sys
import json
import csv
import zipfile
import yaml
from pathlib import Path
from PIL import Image, ImageDraw
import io

CLASS_MAP = {
    "Crab-Pot": 0,
    "Maybe-Crab-Pot": 1,
    "Maybe-Pot": 1,
}

CLASS_NAMES = {
    0: "Crab-Pot",
    1: "Maybe-Crab-Pot",
}


def convert_and_extract(
    zip_path: str = "/home/ashish/Downloads/GhostVision_DatasetAndModels.zip",
    output_base: str = "/home/ashish/sonar-vision/datasets/ghostvision_sss_yolo",
):
    zip_path = Path(zip_path)
    output_dir = Path(output_base)

    if not zip_path.exists():
        raise FileNotFoundError(f"Source ZIP not found: {zip_path}")

    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "qa").mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    manifest_rows = []
    stats = {
        "total_images": 0,
        "split_images": {"train": 0, "val": 0, "test": 0},
        "total_objects": 0,
        "class_counts": {0: 0, 1: 0},
        "malformed_missing": 0,
        "empty_images": {"train": 0, "val": 0, "test": 0},
    }

    qa_samples = {
        "Crab-Pot": [],
        "Maybe-Crab-Pot": [],
        "background": [],
    }

    print(f"Opening archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        all_zip_names = set(z.namelist())

        for source_split, out_split in split_mapping.items():
            meta_zip_path = f"sss-crab-pot-detection-ds/{source_split}/metadata.jsonl"
            if meta_zip_path not in all_zip_names:
                print(f"Warning: {meta_zip_path} not found in zip")
                continue

            lines = z.read(meta_zip_path).decode("utf-8").strip().split("\n")
            print(f"Processing source split '{source_split}' -> output split '{out_split}' ({len(lines)} entries)...")

            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                fn = record["file_name"]
                img_zip_path = f"sss-crab-pot-detection-ds/{source_split}/{fn}"

                if img_zip_path not in all_zip_names:
                    print(f"Error: Image {img_zip_path} missing in zip!")
                    stats["malformed_missing"] += 1
                    continue

                stats["total_images"] += 1
                stats["split_images"][out_split] += 1

                # Read image binary
                img_bytes = z.read(img_zip_path)
                img = Image.open(io.BytesIO(img_bytes))
                w, h = img.size

                # Destination paths
                dest_img_path = output_dir / "images" / out_split / fn
                stem = fn.rsplit(".", 1)[0]
                dest_lbl_path = output_dir / "labels" / out_split / f"{stem}.txt"

                # Write image
                with open(dest_img_path, "wb") as f_img:
                    f_img.write(img_bytes)

                # Process bounding boxes
                raw_bboxes = record.get("objects", {}).get("bbox", [])
                raw_categories = record.get("objects", {}).get("category", [])

                if len(raw_bboxes) != len(raw_categories):
                    print(f"Mismatch in bboxes vs categories for {fn}")
                    stats["malformed_missing"] += 1

                yolo_lines = []
                classes_present_set = set()
                box_records_for_qa = []

                for bbox, cat_name in zip(raw_bboxes, raw_categories):
                    if cat_name not in CLASS_MAP:
                        print(f"Unknown category '{cat_name}' in {fn}")
                        stats["malformed_missing"] += 1
                        continue

                    cid = CLASS_MAP[cat_name]
                    if len(bbox) != 4:
                        print(f"Invalid bbox length {len(bbox)} in {fn}: {bbox}")
                        stats["malformed_missing"] += 1
                        continue

                    bx, by, bw, bh = bbox
                    xc = (bx + bw / 2.0) / float(w)
                    yc = (by + bh / 2.0) / float(h)
                    wn = bw / float(w)
                    hn = bh / float(h)

                    # Validate YOLO coordinates
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < wn <= 1.0 and 0.0 < hn <= 1.0):
                        print(f"Malformed normalized bbox in {fn}: xc={xc}, yc={yc}, w={wn}, h={hn}")
                        stats["malformed_missing"] += 1
                        continue

                    stats["total_objects"] += 1
                    stats["class_counts"][cid] += 1
                    classes_present_set.add(CLASS_NAMES[cid])
                    yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                    box_records_for_qa.append((cid, CLASS_NAMES[cid], xc, yc, wn, hn))

                # Write label file (empty if no objects)
                with open(dest_lbl_path, "w") as f_lbl:
                    if yolo_lines:
                        f_lbl.write("\n".join(yolo_lines) + "\n")

                if len(yolo_lines) == 0:
                    stats["empty_images"][out_split] += 1

                # Format classes_present
                classes_present_str = ",".join(sorted(list(classes_present_set)))

                # Manifest row
                manifest_rows.append({
                    "filename": fn,
                    "source_split": source_split,
                    "output_split": out_split,
                    "width": w,
                    "height": h,
                    "num_objects": len(yolo_lines),
                    "classes_present": classes_present_str,
                })

                # Sample for QA
                if len(box_records_for_qa) == 0 and len(qa_samples["background"]) < 10:
                    qa_samples["background"].append((dest_img_path, box_records_for_qa, fn, out_split))
                elif any(b[0] == 0 for b in box_records_for_qa) and len(qa_samples["Crab-Pot"]) < 20:
                    qa_samples["Crab-Pot"].append((dest_img_path, box_records_for_qa, fn, out_split))
                elif any(b[0] == 1 for b in box_records_for_qa) and len(qa_samples["Maybe-Crab-Pot"]) < 10:
                    qa_samples["Maybe-Crab-Pot"].append((dest_img_path, box_records_for_qa, fn, out_split))

    # Write conversion_manifest.csv
    manifest_path = output_dir / "metadata" / "conversion_manifest.csv"
    with open(manifest_path, "w", newline="") as f_csv:
        fieldnames = [
            "filename",
            "source_split",
            "output_split",
            "width",
            "height",
            "num_objects",
            "classes_present",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    print(f"Wrote conversion manifest: {manifest_path} ({len(manifest_rows)} rows)")

    # Write data.yaml
    data_yaml_content = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "Crab-Pot",
            1: "Maybe-Crab-Pot",
        },
    }
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f_yaml:
        yaml.dump(data_yaml_content, f_yaml, sort_keys=False)
    print(f"Wrote data.yaml: {data_yaml_path}")

    # Generate QA Overlays
    qa_dir = output_dir / "qa"
    generate_qa_overlays(qa_samples, qa_dir)

    print("\nExtraction & Conversion Complete!")
    print(f"Total GhostVision images: {stats['total_images']}")
    print(f"Train / Val / Test: {stats['split_images']['train']} / {stats['split_images']['val']} / {stats['split_images']['test']}")
    print(f"Total Objects: {stats['total_objects']}")
    print(f"Crab-Pot objects: {stats['class_counts'][0]}")
    print(f"Maybe-Crab-Pot objects: {stats['class_counts'][1]}")
    print(f"Malformed / Missing: {stats['malformed_missing']}")

    return stats


def generate_qa_overlays(qa_samples, qa_dir: Path):
    """Generates visual QA overlays with bounding boxes."""
    crab_samples = qa_samples["Crab-Pot"][:10]
    maybe_samples = qa_samples["Maybe-Crab-Pot"][:5]
    bg_samples = qa_samples["background"][:5]

    colors = {
        0: (0, 255, 0),    # Green for Crab-Pot
        1: (255, 165, 0),  # Orange for Maybe-Crab-Pot
    }

    qa_index = []

    for category, sample_list in [("Crab-Pot", crab_samples), ("Maybe-Crab-Pot", maybe_samples), ("Background", bg_samples)]:
        for idx, (img_path, boxes, fn, split) in enumerate(sample_list):
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size

            for cid, cname, xc, yc, wn, hn in boxes:
                bw = wn * w
                bh = hn * h
                x0 = (xc * w) - (bw / 2.0)
                y0 = (yc * h) - (bh / 2.0)
                x1 = x0 + bw
                y1 = y0 + bh

                color = colors.get(cid, (255, 0, 0))
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                label_text = f"{cname} [{cid}]"
                draw.text((x0 + 2, max(0, y0 - 12)), label_text, fill=color)

            safe_cat = category.lower().replace("-", "_")
            out_name = f"qa_{safe_cat}_{idx+1:02d}_{fn}"
            out_path = qa_dir / out_name
            img.save(out_path, quality=95)

            qa_index.append({
                "qa_file": out_name,
                "category_group": category,
                "original_filename": fn,
                "split": split,
                "num_boxes": len(boxes),
            })

    with open(qa_dir / "qa_manifest.json", "w") as f:
        json.dump(qa_index, f, indent=2)
    print(f"Generated {len(qa_index)} Visual QA overlay samples in {qa_dir}")


if __name__ == "__main__":
    convert_and_extract()
