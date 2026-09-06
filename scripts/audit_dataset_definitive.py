#!/usr/bin/env python3
"""
SonarVision — Definitive Audit of Rebuilt Multi-Source Dataset v2
=================================================================
Uses persisted group_key + original_id from metadata.csv (reliable), not
renamed filenames:
  1. Augmented-copy leakage (aug copy vs source image split)
  2. Group/sequence leakage (same group_key in >1 split)
  3. Box geometry + per-split/per-class box counts (from labels on disk)
  4. Metadata <-> disk consistency
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATASET = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/sonarvision_multisource_v2_new")
# Classes are read from the dataset's own dataset.yaml (data-driven) so the
# audit stays valid as the dataset grows (v2: 5 classes, v4: 6).
import yaml
with open(DATASET / "dataset.yaml") as _f:
    _cfg = yaml.safe_load(_f)
CLASSES = {int(k): v for k, v in _cfg["names"].items()}


def iter_boxes(lbl_path):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            i = 0
            while i + 4 < len(parts):
                try:
                    boxes.append((int(parts[i]), float(parts[i + 1]), float(parts[i + 2]),
                                  float(parts[i + 3]), float(parts[i + 4])))
                    i += 5
                except (ValueError, IndexError):
                    i += 1
    return boxes


def main():
    rows = list(csv.DictReader(open(DATASET / "metadata.csv")))
    print(f"[metadata] {len(rows)} rows")

    # ---- 0. metadata <-> disk consistency ----
    missing = [r["image_path"] for r in rows if not (DATASET / r["image_path"]).exists()]
    print(f"\n0) metadata rows whose image is missing on disk: {len(missing)}")
    for m in missing[:5]:
        print(f"    {m}")
    for split in ["train", "val", "test"]:
        n_img = len(list((DATASET / "images" / split).glob("*")))
        n_lbl = len(list((DATASET / "labels" / split).glob("*.txt")))
        n_meta = sum(1 for r in rows if r["split"] == split)
        print(f"   {split}: disk images={n_img} labels={n_lbl} metadata={n_meta} "
              f"{'OK' if n_img == n_lbl == n_meta else 'MISMATCH'}")

    # ---- 0b. uniform 512x512 grayscale PNG check ----
    from PIL import Image
    sizes = Counter()
    modes = Counter()
    for split in ["train", "val", "test"]:
        for f in os.listdir(DATASET / "images" / split):
            with Image.open(DATASET / "images" / split / f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
    bad_size = {k: v for k, v in sizes.items() if k != (512, 512)}
    print(f"\n0b) image sizes: {dict(sizes)} | modes: {dict(modes)}")
    if bad_size:
        print(f"    NON-512 images: {bad_size} -> FAIL")
        return 1

    # ---- 1. augmented-copy leakage ----
    orig_split = {r["original_id"]: r["split"] for r in rows if r["is_augmented"] == "False"}
    aug_leak = []
    for r in rows:
        if r["is_augmented"] == "True":
            src_stem = os.path.splitext(os.path.basename(r["augmented_from"]))[0]
            ss = orig_split.get(src_stem) or orig_split.get(r["original_id"])
            if ss is not None and ss != r["split"]:
                aug_leak.append((r["image_id"], r["split"], ss))
    print(f"\n1) augmented copies: {sum(1 for r in rows if r['is_augmented']=='True')} "
          f"| LEAKED across splits: {len(aug_leak)}")
    for a in aug_leak[:5]:
        print(f"    {a}")

    # ---- 2. group-key leakage ----
    grp = defaultdict(lambda: {"splits": set(), "n": 0})
    for r in rows:
        g = r["group_key"]
        grp[g]["splits"].add(r["split"])
        grp[g]["n"] += 1
    leaked = {g: v for g, v in grp.items() if len(v["splits"]) > 1}
    n_leak_imgs = sum(v["n"] for v in leaked.values())
    print(f"\n2) groups: {len(grp)} | spanning >1 split: {len(leaked)} ({n_leak_imgs} imgs)")
    for g, v in list(leaked.items())[:10]:
        print(f"    {g}: {sorted(v['splits'])} ({v['n']} imgs)")

    # ---- 3. box geometry + class/split counts (from labels on disk) ----
    per_class_split = defaultdict(Counter)
    sizes = []
    tiny = []
    for split in ["train", "val", "test"]:
        for f in os.listdir(DATASET / "labels" / split):
            for cls, xc, yc, w, h in iter_boxes(DATASET / "labels" / split / f):
                per_class_split[CLASSES[cls]][split] += 1
                wpx, hpx = w * 512, h * 512
                sizes.append((wpx, hpx))
                # Sub-3px boxes are unusable junk; 3-8px thin targets (e.g.
                # drowning victims) are legitimate annotations.
                if wpx < 3 or hpx < 3:
                    tiny.append((split, f))
    wpxs = sorted(s[0] for s in sizes)
    print(f"\n3) boxes total: {len(sizes)} | degenerate(<3px): {len(tiny)}")
    for t in tiny[:5]:
        print(f"    {t}")
    print(f"   width px min/p25/med/p75/max: "
          f"{wpxs[0]:.0f}/{wpxs[len(wpxs)//4]:.0f}/{wpxs[len(wpxs)//2]:.0f}/"
          f"{wpxs[3*len(wpxs)//4]:.0f}/{wpxs[-1]:.0f}")
    print("\n4) per-class box counts (train/val/test = total):")
    for c in CLASSES.values():
        tr, vl, te = (per_class_split[c].get("train", 0), per_class_split[c].get("val", 0),
                      per_class_split[c].get("test", 0))
        frac = f"{(tr+vl+te) and (tr/(tr+vl+te))*100:.1f}% train"
        print(f"   {c:<16} {tr:>5} {vl:>5} {te:>5} = {tr+vl+te:<5} ({frac})")

    # ---- 5. background ratio ----
    print("\n5) background-only (empty-label) images:")
    for split in ["train", "val", "test"]:
        empty = sum(1 for f in os.listdir(DATASET / "labels" / split)
                    if os.path.getsize(DATASET / "labels" / split / f) == 0)
        total = len(list((DATASET / "images" / split).glob("*")))
        print(f"   {split}: {empty} / {total}")

    print("\n" + "=" * 70)
    ok = (len(missing) == 0 and not aug_leak and not leaked and len(tiny) == 0)
    print(f"VERDICT: {'PASS - dataset is clean' if ok else 'ISSUES FOUND'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())