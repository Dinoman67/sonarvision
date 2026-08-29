#!/usr/bin/env python3
"""
scripts/multifeature_analysis.py

Full multi-feature analysis combining:
1. Object shape metrics (aspect ratio, compactness, elongation)
2. Acoustic intensity (mean, peak, p90, anomaly)
3. Shadow geometry (depth, connectivity, directionality, area ratio)
4. Surrounding seabed context (texture, homogeneity, gradient field)

Tests whether the COMBINATION of all features provides meaningful
discrimination between debris and background bright spots.
"""

import os
import re
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy import ndimage, stats
from PIL import Image, ImageDraw
import itertools

# ── Targets ──────────────────────────────────────────────────────────────────
VERIFIED_TARGETS = [
    {"target_id": "TGT001", "lat_str": "28° 59' 03.3504\" N", "lon_str": "089° 18' 09.3672\" W",
     "length_m": 12.0, "width_m": 6.0, "height_m": 2.1, "shape_type": "pipeline"},
    {"target_id": "TGT002", "lat_str": "28° 58' 58.6344\" N", "lon_str": "089° 18' 07.3908\" W",
     "length_m": 14.0, "width_m": 10.0, "height_m": 1.4, "shape_type": "obstruction"},
    {"target_id": "TGT003", "lat_str": "29° 01' 35.0112\" N", "lon_str": "089° 17' 25.5372\" W",
     "length_m": 12.0, "width_m": 12.0, "height_m": None, "shape_type": "platform"},
    {"target_id": "TGT004", "lat_str": "28° 54' 58.5288\" N", "lon_str": "089° 21' 56.8800\" W",
     "length_m": 16.0, "width_m": 6.0, "height_m": None, "shape_type": "pipeline"},
    {"target_id": "TGT005", "lat_str": "28° 54' 48.1320\" N", "lon_str": "089° 22' 05.1348\" W",
     "length_m": 16.0, "width_m": 6.0, "height_m": None, "shape_type": "pipeline"},
    {"target_id": "TGT006", "lat_str": "28° 54' 54.197\" N", "lon_str": "089° 22' 16.974\" W",
     "length_m": 16.0, "width_m": 8.0, "height_m": None, "shape_type": "linear"},
    {"target_id": "TGT007", "lat_str": "28° 54' 10.6380\" N", "lon_str": "089° 25' 30.9252\" W",
     "length_m": 14.0, "width_m": 8.0, "height_m": 2.1, "shape_type": "obstruction"},
    {"target_id": "TGT008", "lat_str": "28° 54' 42.4764\" N", "lon_str": "089° 25' 24.0312\" W",
     "length_m": 16.0, "width_m": 10.0, "height_m": 2.7, "shape_type": "structure"},
    {"target_id": "TGT009", "lat_str": "28° 54' 41.62\" N", "lon_str": "089° 25' 22.30\" W",
     "length_m": 22.0, "width_m": 10.0, "height_m": None, "shape_type": "structure"},
    {"target_id": "TGT010", "lat_str": "28° 54' 38.79\" N", "lon_str": "089° 25' 17.40\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 6.7, "shape_type": "obstruction"},
    {"target_id": "TGT011", "lat_str": "28° 54' 59.10\" N", "lon_str": "089° 25' 34.45\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 5.2, "shape_type": "obstruction"},
    {"target_id": "TGT012", "lat_str": "28° 54' 59.78\" N", "lon_str": "089° 25' 32.53\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 5.5, "shape_type": "obstruction"},
    {"target_id": "TGT013", "lat_str": "28° 54' 32.396\" N", "lon_str": "089° 25' 59.387\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 4.3, "shape_type": "obstruction"},
    {"target_id": "TGT014", "lat_str": "28° 54' 30.07\" N", "lon_str": "089° 26' 03.29\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 3.0, "shape_type": "obstruction"},
    {"target_id": "TGT015", "lat_str": "28° 54' 18.14\" N", "lon_str": "089° 25' 45.09\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": None, "shape_type": "obstruction"},
    {"target_id": "TGT016", "lat_str": "28° 53' 35.68\" N", "lon_str": "089° 25' 44.09\" W",
     "length_m": 14.0, "width_m": 6.0, "height_m": None, "shape_type": "pipe"},
    {"target_id": "TGT017", "lat_str": "28° 53' 19.464\" N", "lon_str": "089° 26' 14.276\" W",
     "length_m": 12.0, "width_m": 10.0, "height_m": None, "shape_type": "wreck"},
]

TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]
PX_SIZE = 0.5
SEARCH_R = 150


def dms_to_dd(s):
    parts = re.split(r'[°\'\"/\-\s]+', s.strip())
    parts = [p for p in parts if p]
    d, m, sec = float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0, float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + sec / 3600.0
    if 'S' in s.upper() or 'W' in s.upper():
        dd = -dd
    return dd


def safe_read(ds, col, row, radius):
    c, r = int(round(col)), int(round(row))
    x1, y1 = max(0, c - radius), max(0, r - radius)
    x2, y2 = min(ds.width, c + radius), min(ds.height, r + radius)
    if x1 >= x2 or y1 >= y2:
        return None, None
    return ds.read(1, window=Window(x1, y1, x2 - x1, y2 - y1)), (x1, y1)


def extract_all_features(data, target_local, bw_px, bh_px, label=""):
    """Extract the full multi-dimensional feature vector."""
    h, w = data.shape
    cx, cy = int(round(target_local[0])), int(round(target_local[1]))

    valid = data > 0
    if valid.sum() < 200:
        return None

    # ── Region masks ──
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Target bbox
    x1 = max(0, cx - int(bw_px))
    x2 = min(w, cx + int(bw_px))
    y1 = max(0, cy - int(bh_px))
    y2 = min(h, cy + int(bh_px))
    target_mask = np.zeros_like(valid)
    target_mask[y1:y2, x1:x2] = True

    # Context ring (surrounding seabed)
    ctx_mask = valid & (dist > max(bw_px, bh_px) * 1.2) & (dist < SEARCH_R * 0.8)

    # Near-adjacent zone (where shadow would be)
    near_mask = valid & (dist > bw_px * 0.5) & (dist < bw_px * 2.5)

    if ctx_mask.sum() < 50 or target_mask.sum() < 10:
        return None

    # ── BASELINE CONTEXT (surrounding seabed) ──
    ctx_vals = data[ctx_mask]
    ctx_mean = float(ctx_vals.mean())
    ctx_std = float(ctx_vals.std())
    ctx_median = float(np.median(ctx_vals))
    ctx_p25 = float(np.percentile(ctx_vals, 25))
    ctx_p75 = float(np.percentile(ctx_vals, 75))
    ctx_iqr = ctx_p75 - ctx_p25

    if ctx_std < 2:
        return None

    # ── 1. SHAPE FEATURES (from bright return morphology) ──
    target_thresh = ctx_mean + 1.5 * ctx_std
    bright_mask = valid & target_mask & (data > target_thresh)

    if bright_mask.sum() > 5:
        # Morphological analysis of bright return
        # Compactness: how round vs elongated
        labeled, nf = ndimage.label(bright_mask)
        if nf > 0:
            sizes = ndimage.sum(bright_mask, labeled, range(1, nf + 1))
            largest_label = np.argmax(sizes) + 1
            component = labeled == largest_label
            component_area = int(component.sum())

            # Perimeter approximation
            eroded = ndimage.binary_erosion(component)
            perimeter_mask = component & ~eroded
            perimeter = int(perimeter_mask.sum())

            # Compactness = 4π·area / perimeter² (circle = 1, elongated < 1)
            compactness = (4 * np.pi * component_area) / max(perimeter**2, 1)

            # Aspect ratio of bright return
            ys, xs = np.where(component)
            if len(xs) > 3:
                # Use PCA for orientation
                coords = np.column_stack([xs, ys])
                coords_centered = coords - coords.mean(axis=0)
                cov = np.cov(coords_centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                aspect_ratio = np.sqrt(eigenvalues[0] / max(eigenvalues[1], 1e-10))
                elongation = 1.0 - (1.0 / aspect_ratio)  # 0=circle, →1=line
            else:
                aspect_ratio = 1.0
                elongation = 0.0

            # Eccentricity
            if len(xs) > 3:
                cov2 = np.cov(coords_centered.T)
                eigvals = np.linalg.eigvalsh(cov2)
                eigvals = np.sort(eigvals)[::-1]
                eccentricity = float(np.sqrt(max(0, 1 - eigvals[1] / max(eigvals[0], 1e-10))))
            else:
                eccentricity = 0.0
        else:
            compactness = 0
            aspect_ratio = 1.0
            elongation = 0
            eccentricity = 0
            component_area = 0
    else:
        compactness = 0
        aspect_ratio = 1.0
        elongation = 0
        eccentricity = 0
        component_area = 0

    # ── 2. INTENSITY FEATURES ──
    target_vals = data[target_mask & valid]
    if len(target_vals) == 0:
        return None

    intensity_mean = float(target_vals.mean())
    intensity_max = float(target_vals.max())
    intensity_p90 = float(np.percentile(target_vals, 90))
    intensity_p10 = float(np.percentile(target_vals, 10))
    intensity_median = float(np.median(target_vals))
    intensity_std = float(target_vals.std())

    # Anomalies relative to context
    intensity_anomaly_mean = (intensity_mean - ctx_mean) / max(ctx_std, 1.0)
    intensity_anomaly_peak = (intensity_p90 - ctx_mean) / max(ctx_std, 1.0)
    intensity_range = (intensity_p90 - intensity_p10) / max(ctx_iqr, 1.0)

    # Brightness excess: what fraction of target bbox is above context mean
    bright_fraction = float((target_vals > ctx_mean).sum()) / max(len(target_vals), 1)
    very_bright_fraction = float((target_vals > ctx_mean + 2 * ctx_std).sum()) / max(len(target_vals), 1)

    # ── 3. SHADOW GEOMETRY FEATURES ──
    # Search for shadow in the near-adjacent zone
    near_vals = data[near_mask & valid]
    if len(near_vals) > 20:
        # Shadow depth: darkest 10% of near zone vs context
        shadow_thresh = np.percentile(near_vals, 10)
        shadow_mask = near_mask & valid & (data <= shadow_thresh)

        if shadow_mask.sum() > 3:
            shadow_vals = data[shadow_mask]
            shadow_mean_intensity = float(shadow_vals.mean())
            shadow_depth = (ctx_mean - shadow_mean_intensity) / max(ctx_mean, 1.0)

            # Shadow area as fraction of near zone
            shadow_area_frac = float(shadow_mask.sum()) / max(near_mask.sum(), 1)

            # Shadow connectivity (is it a coherent region or scattered?)
            labeled_s, nf_s = ndimage.label(shadow_mask)
            if nf_s > 0:
                sizes_s = ndimage.sum(shadow_mask, labeled_s, range(1, nf_s + 1))
                largest_s = float(max(sizes_s)) if len(sizes_s) > 0 else 0
                total_s = float(shadow_mask.sum())
                shadow_connectivity = largest_s / max(total_s, 1)

                # Shadow centroid offset from target center (directionality)
                dark_ys, dark_xs = np.where(shadow_mask)
                shadow_cx = float(np.mean(dark_xs)) - cx
                shadow_cy = float(np.mean(dark_ys)) - cy
                shadow_offset_dist = np.sqrt(shadow_cx**2 + shadow_cy**2)
                shadow_offset_angle = np.degrees(np.arctan2(shadow_cy, shadow_cx))
            else:
                shadow_connectivity = 0
                shadow_offset_dist = 0
                shadow_offset_angle = 0
        else:
            shadow_depth = 0
            shadow_area_frac = 0
            shadow_connectivity = 0
            shadow_offset_dist = 0
            shadow_offset_angle = 0
    else:
        shadow_depth = 0
        shadow_area_frac = 0
        shadow_connectivity = 0
        shadow_offset_dist = 0
        shadow_offset_angle = 0

    # ── 4. CONTEXT / TEXTURE FEATURES ──
    # GLCM-like texture metrics on context region
    # Local variance in context (texture complexity)
    ctx_patch = data.copy()
    ctx_patch[~ctx_mask] = 0
    local_var = ndimage.uniform_filter(ctx_patch.astype(float)**2, size=11) - ndimage.uniform_filter(ctx_patch.astype(float), size=11)**2
    ctx_local_var_mean = float(local_var[ctx_mask & (local_var > 0)].mean()) if (ctx_mask & (local_var > 0)).sum() > 0 else 0

    # Gradient field in context
    smoothed = ndimage.gaussian_filter(data.astype(float), sigma=3.0)
    gy, gx = np.gradient(smoothed)
    grad_mag = np.sqrt(gx**2 + gy**2)
    ctx_grad_mean = float(grad_mag[ctx_mask].mean()) if ctx_mask.sum() > 0 else 0

    # Gradient in target vs context
    target_grad_mean = float(grad_mag[target_mask & valid].mean()) if (target_mask & valid).sum() > 0 else 0
    grad_ratio = target_grad_mean / max(ctx_grad_mean, 0.01)

    # Texture homogeneity in context (low = uniform, high = varied)
    ctx_cv = ctx_std / max(ctx_mean, 1.0)  # coefficient of variation

    # ── 5. BRIGHT-SHADOW COUPLING ──
    # Is the bright return spatially coupled to a dark region?
    # Measure the gradient between bright return and nearby dark zone
    if bright_mask.sum() > 5:
        labeled_b, nf_b = ndimage.label(bright_mask)
        if nf_b > 0:
            sizes_b = ndimage.sum(bright_mask, labeled_b, range(1, nf_b + 1))
            largest_b_label = np.argmax(sizes_b) + 1
            bright_component = labeled_b == largest_b_label
            bright_ys, bright_xs = np.where(bright_component)
            bright_cx = float(np.mean(bright_xs))
            bright_cy = float(np.mean(bright_ys))

            # For each direction from bright center, measure gradient
            max_neg_gradient = 0
            for angle in range(0, 360, 15):
                rad = np.radians(angle)
                profile_len = 80
                grad_vals = []
                for d in range(5, profile_len):
                    px = int(bright_cx + d * np.cos(rad))
                    py = int(bright_cy + d * np.sin(rad))
                    if 0 <= px < w and 0 <= py < h:
                        grad_vals.append(float(smoothed[py, px]))
                if len(grad_vals) > 3:
                    diffs = np.diff(grad_vals)
                    min_diff = float(np.min(diffs))
                    if min_diff < max_neg_gradient:
                        max_neg_gradient = min_diff

            coupling_score = abs(max_neg_gradient) / max(ctx_std, 1.0)
        else:
            coupling_score = 0
    else:
        coupling_score = 0

    return {
        # Shape
        "shape_compactness": round(compactness, 4),
        "shape_aspect_ratio": round(aspect_ratio, 3),
        "shape_elongation": round(elongation, 4),
        "shape_eccentricity": round(eccentricity, 4),
        "shape_bright_area_px": component_area,
        # Intensity
        "intensity_anomaly_mean": round(intensity_anomaly_mean, 3),
        "intensity_anomaly_peak": round(intensity_anomaly_peak, 3),
        "intensity_range": round(intensity_range, 3),
        "bright_fraction": round(bright_fraction, 4),
        "very_bright_fraction": round(very_bright_fraction, 4),
        # Shadow
        "shadow_depth": round(shadow_depth, 4),
        "shadow_area_frac": round(shadow_area_frac, 4),
        "shadow_connectivity": round(shadow_connectivity, 4),
        "shadow_offset_dist": round(shadow_offset_dist, 2),
        # Context
        "ctx_cv": round(ctx_cv, 4),
        "ctx_local_var": round(ctx_local_var_mean, 2),
        "grad_ratio": round(grad_ratio, 4),
        # Coupling
        "coupling_score": round(coupling_score, 4),
    }


def compute_auc(dv, bv):
    all_vals = sorted(set(dv + bv))
    n_pos, n_neg = len(dv), len(bv)
    tpr = [0.0]; fpr = [0.0]
    for thresh in all_vals:
        tp = sum(1 for s in dv if s >= thresh)
        fp = sum(1 for s in bv if s >= thresh)
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    tpr.append(1.0); fpr.append(1.0)
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return auc


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(base_dir, "results", "shadow_analysis")
    os.makedirs(out_dir, exist_ok=True)

    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    ds_list = []
    for rel in TIFF_FILES:
        ds = rasterio.open(os.path.join(base_dir, rel))
        ds_list.append((os.path.basename(rel), ds))

    all_features = {"debris": [], "background": []}
    feature_names = [
        "shape_compactness", "shape_aspect_ratio", "shape_elongation", "shape_eccentricity",
        "intensity_anomaly_mean", "intensity_anomaly_peak", "intensity_range",
        "bright_fraction", "very_bright_fraction",
        "shadow_depth", "shadow_area_frac", "shadow_connectivity", "shadow_offset_dist",
        "ctx_cv", "ctx_local_var", "grad_ratio",
        "coupling_score",
    ]

    print("=" * 90)
    print("MULTI-FEATURE ANALYSIS: Shape + Intensity + Shadow + Context + Coupling")
    print("=" * 90)

    # ── Debris ──
    for t in VERIFIED_TARGETS:
        lat = dms_to_dd(t["lat_str"])
        lon = dms_to_dd(t["lon_str"])
        if lon > 0: lon = -lon
        ux, uy = tf.transform(lon, lat)
        bw_px = t["length_m"] / PX_SIZE
        bh_px = t["width_m"] / PX_SIZE

        for tiff_name, ds in ds_list:
            inv = ~ds.transform
            tc, tr = inv * (ux, uy)
            data, off = safe_read(ds, tc, tr, SEARCH_R)
            if data is None:
                continue
            lc, lr = tc - off[0], tr - off[1]
            feat = extract_all_features(data, (lc, lr), bw_px, bh_px,
                                        label=f"{t['target_id']}|{tiff_name}")
            if feat:
                feat["target_id"] = t["target_id"]
                feat["source_tiff"] = tiff_name
                feat["height_m"] = t["height_m"]
                feat["shape_type"] = t["shape_type"]
                all_features["debris"].append(feat)

    # ── Background ──
    bg_regions = [
        ("H11833_1of2.tif", 262500, 3199000, "SW_bg1"),
        ("H11833_1of2.tif", 264500, 3201500, "SW_bg2"),
        ("H11833_2of2.tif", 263500, 3199500, "SW_bg3"),
        ("H11833_1of2.tif", 276000, 3210000, "NE_bg1"),
        ("H11833_2of2.tif", 277000, 3209000, "NE_bg2"),
        ("H11833_1of2.tif", 269500, 3201000, "CE_bg1"),
        ("H11833_1of2.tif", 270500, 3202000, "CE_bg2"),
        ("H11833_2of2.tif", 264000, 3202000, "SW_bg4"),
        ("H11833_1of2.tif", 265200, 3198200, "SW_bg5"),
        ("H11833_2of2.tif", 262800, 3201000, "SW_bg6"),
        ("H11833_1of2.tif", 277200, 3208500, "NE_bg3"),
        ("H11833_2of2.tif", 269800, 3201800, "CE_bg3"),
    ]

    for tiff_name, ux, uy, desc in bg_regions:
        ds = next((d for n, d in ds_list if n == tiff_name), None)
        if ds is None: continue
        inv = ~ds.transform
        col, row = inv * (ux, uy)
        data, off = safe_read(ds, col, row, SEARCH_R)
        if data is None: continue
        lc, lr = col - off[0], row - off[1]
        feat = extract_all_features(data, (lc, lr), 24, 16, label=f"{desc}|{tiff_name}")
        if feat:
            feat["region"] = desc
            feat["source_tiff"] = tiff_name
            all_features["background"].append(feat)

    for _, ds in ds_list:
        ds.close()

    # ── Per-feature AUC analysis ──
    d_feats = all_features["debris"]
    b_feats = all_features["background"]
    print(f"\nN debris={len(d_feats)}, N background={len(b_feats)}")

    print(f"\n{'Feature':<28} {'Debris μ':>9} {'BG μ':>9} {'Δ':>8} {'AUC':>6} {'Cohen d':>8} {'Rating':>8}")
    print("-" * 90)

    feature_aucs = {}
    for feat_name in feature_names:
        dv = [r[feat_name] for r in d_feats if feat_name in r]
        bv = [r[feat_name] for r in b_feats if feat_name in r]
        if not dv or not bv:
            continue

        d_mean, b_mean = np.mean(dv), np.mean(bv)
        delta = d_mean - b_mean

        # Cohen's d
        pooled = np.sqrt((np.var(dv, ddof=1) + np.var(bv, ddof=1)) / 2)
        d_cohen = delta / pooled if pooled > 0 else 0

        # AUC (>0.5 means debris scores higher, good)
        auc = compute_auc(dv, bv)

        # Rating
        if auc > 0.7:
            rating = "***GOOD***"
        elif auc > 0.6:
            rating = "*OK*"
        elif auc > 0.55:
            rating = "weak"
        else:
            rating = "none"

        feature_aucs[feat_name] = auc
        print(f"{feat_name:<28} {d_mean:>9.3f} {b_mean:>9.3f} {delta:>+8.3f} {auc:>6.3f} {d_cohen:>+8.3f} {rating:>8}")

    # ── Feature ranking ──
    print(f"\n{'=' * 60}")
    print("FEATURE RANKING (by AUC):")
    print("=" * 60)
    ranked = sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True)
    for i, (name, auc) in enumerate(ranked, 1):
        marker = " ← BEST" if i == 1 else ""
        print(f"  {i:2d}. {name:<28} AUC={auc:.3f}{marker}")

    # ── Best single feature ──
    best_name, best_auc = ranked[0]
    print(f"\nBest single feature: {best_name} (AUC={best_auc:.3f})")

    # ── Top-N feature combinations ──
    print(f"\n{'=' * 60}")
    print("FEATURE COMBINATIONS (testing all pairs and triples):")
    print("=" * 60)

    def normalize(values):
        mn, mx = min(values), max(values)
        if mx - mn < 1e-10:
            return [0.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    # Build normalized feature matrix
    all_dv = {f: [r[f] for r in d_feats if f in r] for f in feature_names}
    all_bv = {f: [r[f] for r in b_feats if f in r] for f in feature_names}

    # Test pairs
    best_pair_auc = 0
    best_pair = None
    for f1, f2 in itertools.combinations(feature_names, 2):
        if len(all_dv[f1]) != len(d_feats) or len(all_bv[f1]) != len(b_feats):
            continue
        if len(all_dv[f2]) != len(d_feats) or len(all_bv[f2]) != len(b_feats):
            continue

        # Normalize and combine
        d1 = normalize(all_dv[f1])
        b1 = normalize(all_bv[f1])
        d2 = normalize(all_dv[f2])
        b2 = normalize(all_bv[f2])

        d_combo = [a + b for a, b in zip(d1, d2)]
        b_combo = [a + b for a, b in zip(b1, b2)]

        auc = compute_auc(d_combo, b_combo)
        if auc > best_pair_auc:
            best_pair_auc = auc
            best_pair = (f1, f2)

    if best_pair:
        print(f"  Best pair: {best_pair[0]} + {best_pair[1]} → AUC={best_pair_auc:.3f}")

    # Test triples
    best_triple_auc = 0
    best_triple = None
    for f1, f2, f3 in itertools.combinations(feature_names, 3):
        if any(len(all_dv[f]) != len(d_feats) for f in [f1, f2, f3]):
            continue
        if any(len(all_bv[f]) != len(b_feats) for f in [f1, f2, f3]):
            continue

        d_combo = [normalize(all_dv[f1])[i] + normalize(all_dv[f2])[i] + normalize(all_dv[f3])[i]
                   for i in range(len(d_feats))]
        b_combo = [normalize(all_bv[f1])[i] + normalize(all_bv[f2])[i] + normalize(all_bv[f3])[i]
                   for i in range(len(b_feats))]

        auc = compute_auc(d_combo, b_combo)
        if auc > best_triple_auc:
            best_triple_auc = auc
            best_triple = (f1, f2, f3)

    if best_triple:
        print(f"  Best triple: {' + '.join(best_triple)} → AUC={best_triple_auc:.3f}")

    # Test all features combined (equal weight)
    all_d_combo = []
    all_b_combo = []
    for i in range(len(d_feats)):
        total = 0
        count = 0
        for f in feature_names:
            if f in all_dv and len(all_dv[f]) > i:
                vals = normalize(all_dv[f] + all_bv[f])
                d_vals = vals[:len(all_dv[f])]
                b_vals = vals[len(all_dv[f]):]
                total += d_vals[i]
                count += 1
        all_d_combo.append(total / max(count, 1))

    for i in range(len(b_feats)):
        total = 0
        count = 0
        for f in feature_names:
            if f in all_bv and len(all_bv[f]) > i:
                vals = normalize(all_dv[f] + all_bv[f])
                d_vals = vals[:len(all_dv[f])]
                b_vals = vals[len(all_dv[f]):]
                total += b_vals[i]
                count += 1
        all_b_combo.append(total / max(count, 1))

    all_auc = compute_auc(all_d_combo, all_b_combo)
    print(f"  All 17 features (equal weight): AUC={all_auc:.3f}")

    # ── Save results ──
    results = {
        "feature_aucs": feature_aucs,
        "best_single": {"feature": best_name, "auc": best_auc},
        "best_pair": {"features": list(best_pair), "auc": best_pair_auc} if best_pair else None,
        "best_triple": {"features": list(best_triple), "auc": best_triple_auc} if best_triple else None,
        "all_features_auc": all_auc,
        "n_debris": len(d_feats),
        "n_background": len(b_feats),
    }

    results_path = os.path.join(out_dir, "multifeature_analysis.json")
    with open(results_path, "w") as f:
        json.dump({"summary": results, "debris_features": d_feats, "bg_features": b_feats},
                  f, indent=2, default=str)
    print(f"\n[✓] Saved to {results_path}")

    # ── Final verdict ──
    print(f"\n{'=' * 60}")
    print("FINAL VERDICT")
    print("=" * 60)
    if best_pair_auc >= 0.7:
        print(f"  PAIR of features achieves AUC={best_pair_auc:.3f} — USEFUL!")
        print(f"  Recommend: {best_pair[0]} + {best_pair[1]}")
    elif best_triple_auc >= 0.7:
        print(f"  TRIPLE of features achieves AUC={best_triple_auc:.3f} — PROMISING")
    elif all_auc >= 0.65:
        print(f"  Full feature set achieves AUC={all_auc:.3f} — MARGINAL")
    else:
        print(f"  Best AUC={max(best_pair_auc, best_triple_auc, all_auc):.3f} — NOT ENOUGH")
        print(f"  Individual features and combinations do NOT reliably")
        print(f"  discriminate debris from background bright spots.")
        print(f"  RECOMMENDATION: Continue with intensity-only approach.")


if __name__ == "__main__":
    main()
