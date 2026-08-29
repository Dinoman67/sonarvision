#!/usr/bin/env python3
"""
scripts/package_e3_for_colab.py

Zips the NOAA H11833 E3 dataset for upload to Google Colab.
Includes: images, labels, data.yaml, and QA visualizations.

Usage:
    python scripts/package_e3_for_colab.py
    
Output: noaa_debris_e3_colab.zip in project root
"""
import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
E3_DIR = BASE_DIR / "datasets" / "noaa-debris" / "e3"
OUTPUT_ZIP = BASE_DIR / "noaa_debris_e3_colab.zip"

INCLUDE_QA = True  # Set False to skip QA images (saves ~50% zip size)

def main():
    if not E3_DIR.exists():
        print(f"ERROR: E3 directory not found: {E3_DIR}")
        return

    # Count files
    image_count = len(list((E3_DIR / "images").rglob("*.png")))
    label_count = len(list((E3_DIR / "labels").rglob("*.txt")))
    qa_count = len(list((E3_DIR / "qa").glob("*.png"))) if (E3_DIR / "qa").exists() else 0

    print(f"E3 Dataset Summary:")
    print(f"  Images: {image_count}")
    print(f"  Labels: {label_count}")
    print(f"  QA images: {qa_count}")
    print(f"  Zipping to: {OUTPUT_ZIP}")
    print()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add data.yaml
        data_yaml = E3_DIR / "data.yaml"
        if data_yaml.exists():
            zf.write(data_yaml, "e3/data.yaml")
            print(f"  Added: data.yaml")

        # Add all images
        for img_path in sorted((E3_DIR / "images").rglob("*.png")):
            arcname = f"e3/{img_path.relative_to(E3_DIR)}"
            zf.write(img_path, arcname)

        # Add all labels
        for lbl_path in sorted((E3_DIR / "labels").rglob("*.txt")):
            arcname = f"e3/{lbl_path.relative_to(E3_DIR)}"
            zf.write(lbl_path, arcname)

        # Add QA images (optional)
        if INCLUDE_QA and (E3_DIR / "qa").exists():
            for qa_path in sorted((E3_DIR / "qa").glob("*.png")):
                arcname = f"e3/{qa_path.relative_to(E3_DIR)}"
                zf.write(qa_path, arcname)

        # Add metadata
        meta_dir = E3_DIR / "metadata"
        if meta_dir.exists():
            for meta_file in sorted(meta_dir.glob("*")):
                if meta_file.is_file():
                    arcname = f"e3/{meta_file.relative_to(E3_DIR)}"
                    zf.write(meta_file, arcname)

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nDone! {OUTPUT_ZIP.name}: {zip_size_mb:.1f} MB")
    print(f"\nNext steps:")
    print(f"  1. Upload {OUTPUT_ZIP.name} to Google Drive (or directly to Colab)")
    print(f"  2. Open the Colab training script")
    print(f"  3. Upload and run!")


if __name__ == "__main__":
    main()
