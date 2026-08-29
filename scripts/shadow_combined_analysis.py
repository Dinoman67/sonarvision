#!/usr/bin/env python3
"""
scripts/shadow_combined_analysis.py

Re-evaluates shadow + brightness as a COMBINED spatial feature:
- The "acoustic signature" of debris = bright return followed by dark shadow
- Tests whether the PAIRED pattern (not individual metrics) discriminates
- Measures gradient sharpness, spatial coupling, profile shapes
- Compares debris vs background bright spots on these combined metrics
"""

import os
import sys
import re
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy import ndimage, stats
from PIL import Image, ImageDraw

# Re-use targets from prepare_noaa_e3
VERIFIED_TARGETS = [
    {"target_id": "TGT001", "name": "DtoN_1_Pipeline_Exposed",
     "lat_str": "28° 59' 03.3504\" N", "lon_str": "089° 18' 09.3672\" W",
     "length_m": 12.0, "width_m": 6.0, "height_m": 2.1, "cluster": "NE_Cluster"},
    {"target_id": "TGT002", "name": "DtoN_2_Obstruction_Debris",
     "lat_str": "28° 58' 58.6344\" N", "lon_str": "089° 18' 07.3908\" W",
     "length_m": 14.0, "width_m": 10.0, "height_m": 1.4, "cluster": "NE_Cluster"},
    {"target_id": "TGT003", "name": "DtoN_5_Damaged_Wellhead_Ruins",
     "lat_str": "29° 01' 35.0112\" N", "lon_str": "089° 17' 25.5372\" W",
     "length_m": 12.0, "width_m": 12.0, "height_m": None, "cluster": "NE_Cluster"},
    {"target_id": "TGT004", "name": "DtoN_3_1_Elevated_Pipeline",
     "lat_str": "28° 54' 58.5288\" N", "lon_str": "089° 21' 56.8800\" W",
     "length_m": 16.0, "width_m": 6.0, "height_m": None, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT005", "name": "DtoN_3_2_Elevated_Pipeline",
     "lat_str": "28° 54' 48.1320\" N", "lon_str": "089° 22' 05.1348\" W",
     "length_m": 16.0, "width_m": 6.0, "height_m": None, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT006", "name": "Rig_Caisson_Pipeline_Feature",
     "lat_str": "28° 54' 54.197\" N", "lon_str": "089° 22' 16.974\" W",
     "length_m": 16.0, "width_m": 8.0, "height_m": None, "cluster": "Central_East_Cluster"},
    {"target_id": "TGT007", "name": "DtoN_3_3_Obstruction_Debris",
     "lat_str": "28° 54' 10.6380\" N", "lon_str": "089° 25' 30.9252\" W",
     "length_m": 14.0, "width_m": 8.0, "height_m": 2.1, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT008", "name": "DtoN_4_Ruined_Training_Wall",
     "lat_str": "28° 54' 42.4764\" N", "lon_str": "089° 25' 24.0312\" W",
     "length_m": 16.0, "width_m": 10.0, "height_m": 2.7, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT009", "name": "Training_Wall_Ruins_Center",
     "lat_str": "28° 54' 41.62\" N", "lon_str": "089° 25' 22.30\" W",
     "length_m": 22.0, "width_m": 10.0, "height_m": None, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT010", "name": "Obstruction_22ft",
     "lat_str": "28° 54' 38.79\" N", "lon_str": "089° 25' 17.40\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 6.7, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT011", "name": "Obstruction_17ft",
     "lat_str": "28° 54' 59.10\" N", "lon_str": "089° 25' 34.45\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 5.2, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT012", "name": "Obstruction_18ft",
     "lat_str": "28° 54' 59.78\" N", "lon_str": "089° 25' 32.53\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 5.5, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT013", "name": "Obstruction_14ft",
     "lat_str": "28° 54' 32.396\" N", "lon_str": "089° 25' 59.387\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 4.3, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT014", "name": "Obstruction_10ft",
     "lat_str": "28° 54' 30.07\" N", "lon_str": "089° 26' 03.29\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": 3.0, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT015", "name": "Obstruction_Charted_Retained",
     "lat_str": "28° 54' 18.14\" N", "lon_str": "089° 25' 45.09\" W",
     "length_m": 12.0, "width_m": 8.0, "height_m": None, "cluster": "SW_Pass_Main_Cluster"},
    {"target_id": "TGT016", "name": "Pipe_PA_Debris",
     "lat_str": "28° 53' 35.68\" N", "lon_str": "089° 25' 44.09\" W",
     "length_m": 14.0, "width_m": 6.0, "height_m": None, "cluster": "SW_Pass_South_Cluster"},
    {"target_id": "TGT017", "name": "AWOIS_11622_Wreck_Debris",
     "lat_str": "28° 53' 19.464\" N", "lon_str": "089° 26' 14.276\" W",
     "length_m": 12.0, "width_m": 10.0, "height_m": None, "cluster": "SW_Pass_South_Cluster"},
]

TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]
PIXEL_SIZE_M = 0.5
SEARCH_R = 150  # pixels to search around target


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
    data = ds.read(1, window=Window(x1, y1, x2 - x1, y2 - y1))
    return data, (x1, y1)


def intensity_profile(data, p1, p2, n_samples=50):
    """Sample intensity along a line between two points."""
    xs = np.linspace(p1[0], p2[0], n_samples)
    ys = np.linspace(p1[1], p2[1], n_samples)
    h, w = data.shape
    values = []
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            values.append(float(data[iy, ix]))
        else:
            values.append(0.0)
    return np.array(values)


def gradient_magnitude(data, sigma=2.0):
    """Compute gradient magnitude with Gaussian smoothing."""
    smoothed = ndimage.gaussian_filter(data.astype(float), sigma=sigma)
    gy, gx = np.gradient(smoothed)
    return np.sqrt(gx**2 + gy**2), gx, gy


def analyze_spatial_pattern(data, target_local, bw_px, bh_px, label=""):
    """
    Analyze the combined bright+shadow spatial pattern around a target.
    Returns a dict of metrics describing the acoustic signature.
    """
    h, w = data.shape
    cx, cy = int(round(target_local[0])), int(round(target_local[1]))

    # Valid data mask
    valid = data > 0
    if valid.sum() < 100:
        return None

    # Local background statistics (ring around target, far enough to be "seabed")
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    ring_mask = (dist_from_center > 60) & (dist_from_center < 140) & valid
    if ring_mask.sum() < 50:
        return None
    bg_mean = float(data[ring_mask].mean())
    bg_std = float(data[ring_mask].std())
    if bg_std < 2:
        return None

    # ── 1. Gradient analysis ──
    grad_mag, gx, gy = gradient_magnitude(data.astype(float), sigma=3.0)

    # Gradient within target bounding box (should be high for debris with sharp edges)
    x1_bb = max(0, cx - int(bw_px))
    x2_bb = min(w, cx + int(bw_px))
    y1_bb = max(0, cy - int(bh_px))
    y2_bb = min(h, cy + int(bh_px))
    target_grad = grad_mag[y1_bb:y2_bb, x1_bb:x2_bb]
    target_valid_grad = target_grad[target_grad > 0]
    mean_grad_target = float(target_valid_grad.mean()) if len(target_valid_grad) > 0 else 0

    # Gradient in surrounding background ring
    ring_grad = grad_mag[ring_mask]
    mean_grad_bg = float(ring_grad.mean()) if len(ring_grad) > 0 else 0

    # Gradient ratio: target should have sharper edges than background
    grad_ratio = mean_grad_target / max(mean_grad_bg, 0.01)

    # ── 2. Intensity profile analysis (4 cardinal directions) ──
    profile_len = 120  # pixels each direction from center
    directions = {
        'east': ((cx, cy), (min(w-1, cx + profile_len), cy)),
        'west': ((cx, cy), (max(0, cx - profile_len), cy)),
        'north': ((cx, cy), (cx, max(0, cy - profile_len))),
        'south': ((cx, cy), (cx, min(h-1, cy + profile_len))),
    }

    profile_metrics = {}
    max_gradient_any_dir = 0
    max_gradient_location = None

    for dir_name, (p1, p2) in directions.items():
        profile = intensity_profile(data, p1, p2, n_samples=profile_len)

        # Find steepest gradient in this profile
        if len(profile) > 2:
            diffs = np.diff(profile)
            steepest_idx = np.argmax(np.abs(diffs))
            steepest_gradient = float(diffs[steepest_idx])
            # Normalize by local background
            local_val = float(np.median(profile)) if np.median(profile) > 0 else 1.0
            steepest_gradient_normalized = steepest_gradient / local_val

            if abs(steepest_gradient_normalized) > abs(max_gradient_any_dir):
                max_gradient_any_dir = steepest_gradient_normalized
                max_gradient_location = dir_name

            profile_metrics[dir_name] = {
                'steepest_gradient': round(steepest_gradient, 2),
                'steepest_gradient_normalized': round(steepest_gradient_normalized, 4),
                'mean': round(float(profile.mean()), 1),
                'min': round(float(profile.min()), 1),
                'max': round(float(profile.max()), 1),
            }

    # ── 3. Bright-dark couplet detection ──
    # In the direction of steepest gradient, check if there's a bright-dark pair
    couplet_score = 0.0
    couplet_detected = False

    if max_gradient_location:
        p1, p2 = directions[max_gradient_location]
        profile = intensity_profile(data, p1, p2, n_samples=profile_len)

        # Find the center of the profile (target location)
        center_idx = profile_len // 2

        # Check for bright-dark pattern:
        # Half of profile should be bright (debris return) and other half dark (shadow)
        left_half = profile[:center_idx]
        right_half = profile[center_idx+1:]

        # Which side is brighter?
        left_mean = float(np.mean(left_half[left_half > 0])) if (left_half > 0).any() else 0
        right_mean = float(np.mean(right_half[right_half > 0])) if (right_half > 0).any() else 0

        # Bright-dark couplet = one side significantly brighter than the other
        brighter_side_mean = max(left_mean, right_mean)
        darker_side_mean = min(left_mean, right_mean)

        if brighter_side_mean > 0:
            couplet_score = (brighter_side_mean - darker_side_mean) / brighter_side_mean
            couplet_detected = couplet_score > 0.05  # at least 5% asymmetry

    # ── 4. Asymmetry metric ──
    # Debris should create asymmetric intensity pattern (bright on one side, dark on other)
    # Background texture should be more symmetric
    target_region = data[y1_bb:y2_bb, x1_bb:x2_bb]
    valid_target_region = target_region[target_region > 0]
    if len(valid_target_region) > 0:
        # Compare variance in different quadrants
        th, tw = target_region.shape
        if th > 4 and tw > 4:
            q1 = target_region[:th//2, :tw//2]
            q2 = target_region[:th//2, tw//2:]
            q3 = target_region[th//2:, :tw//2]
            q4 = target_region[th//2:, tw//2:]
            means = [float(q[q>0].mean()) if (q>0).any() else 0 for q in [q1, q2, q3, q4]]
            asymmetry = np.std(means) / max(np.mean(means), 1.0)
        else:
            asymmetry = 0.0
    else:
        asymmetry = 0.0

    # ── 5. Brightness anomaly within target bbox ──
    target_vals = target_region[target_region > 0]
    if len(target_vals) > 0:
        target_mean = float(target_vals.mean())
        target_max = float(target_vals.max())
        target_p90 = float(np.percentile(target_vals, 90))
        brightness_anomaly = (target_mean - bg_mean) / max(bg_std, 1.0)
        peak_anomaly = (target_p90 - bg_mean) / max(bg_std, 1.0)
    else:
        target_mean = target_max = target_p90 = 0.0
        brightness_anomaly = peak_anomaly = 0.0

    # ── 6. Shadow detection: darkest connected region adjacent to bright return ──
    # Find the darkest region within 80px of center
    shadow_search_r = 80
    shadow_mask = valid & (dist_from_center < shadow_search_r) & (dist_from_center > 15)
    if shadow_mask.sum() > 20:
        shadow_vals = data[shadow_mask]
        shadow_threshold = np.percentile(shadow_vals, 10)  # bottom 10%
        dark_mask = shadow_mask & (data <= shadow_threshold)

        # Find the centroid of the dark region
        if dark_mask.sum() > 5:
            dark_ys, dark_xs = np.where(dark_mask)
            dark_cx = float(np.mean(dark_xs))
            dark_cy = float(np.mean(dark_ys))

            # Shadow-illumination axis: line from bright center to dark centroid
            shadow_vec_x = dark_cx - cx
            shadow_vec_y = dark_cy - cy
            shadow_dist = np.sqrt(shadow_vec_x**2 + shadow_vec_y**2)

            # Shadow depth: how dark compared to local bg
            dark_region_mean = float(data[dark_mask].mean())
            shadow_depth = (bg_mean - dark_region_mean) / max(bg_mean, 1.0)

            # Shadow continuity: is the dark region a connected blob (true shadow) vs scattered noise?
            labeled, nf = ndimage.label(dark_mask)
            if nf > 0:
                # Find largest component
                sizes = ndimage.sum(dark_mask, labeled, range(1, nf+1))
                largest_size = float(max(sizes)) if len(sizes) > 0 else 0
                total_dark = dark_mask.sum()
                shadow_connectivity = largest_size / max(total_dark, 1)
            else:
                shadow_connectivity = 0.0
        else:
            shadow_dist = 0
            shadow_depth = 0
            shadow_connectivity = 0
    else:
        shadow_dist = 0
        shadow_depth = 0
        shadow_connectivity = 0

    # ── COMBINED SIGNATURE SCORE ──
    # This is the key metric: combines brightness + shadow + gradient into one score
    signature_score = (
        brightness_anomaly * 0.3 +        # Is there a bright return?
        shadow_depth * 0.3 +               # Is there a shadow?
        grad_ratio * 0.2 +                 # Are the edges sharp?
        couplet_score * 0.2                # Is there a bright-dark pair?
    )

    return {
        "label": label,
        "bg_mean": round(bg_mean, 1),
        "bg_std": round(bg_std, 1),
        # Gradient
        "mean_grad_target": round(mean_grad_target, 2),
        "mean_grad_bg": round(mean_grad_bg, 2),
        "grad_ratio": round(grad_ratio, 3),
        # Brightness
        "target_mean": round(target_mean, 1),
        "target_max": round(target_max, 1),
        "target_p90": round(target_p90, 1),
        "brightness_anomaly": round(brightness_anomaly, 3),
        "peak_anomaly": round(peak_anomaly, 3),
        # Shadow
        "shadow_depth": round(shadow_depth, 4),
        "shadow_connectivity": round(shadow_connectivity, 3),
        # Couplet
        "couplet_score": round(couplet_score, 4),
        "couplet_detected": couplet_detected,
        "max_gradient_direction": max_gradient_location,
        # Asymmetry
        "asymmetry": round(asymmetry, 4),
        # COMBINED
        "signature_score": round(signature_score, 4),
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(base_dir, "results", "shadow_analysis")
    os.makedirs(out_dir, exist_ok=True)

    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    # Open TIFFs
    ds_list = []
    for rel_path in TIFF_FILES:
        ds = rasterio.open(os.path.join(base_dir, rel_path))
        ds_list.append((os.path.basename(rel_path), ds))

    # ── Analyze debris targets ──
    print("=" * 80)
    print("COMBINED BRIGHT+SHADOW SIGNATURE ANALYSIS")
    print("=" * 80)

    all_results = {"debris": [], "background": []}

    for t in VERIFIED_TARGETS:
        lat = dms_to_dd(t["lat_str"])
        lon = dms_to_dd(t["lon_str"])
        if lon > 0:
            lon = -lon
        utm_x, utm_y = tf.transform(lon, lat)
        bw_px = t["length_m"] / PIXEL_SIZE_M
        bh_px = t["width_m"] / PIXEL_SIZE_M

        for tiff_name, ds in ds_list:
            inv = ~ds.transform
            tgt_col, tgt_row = inv * (utm_x, utm_y)

            data, offset = safe_read(ds, tgt_col, tgt_row, SEARCH_R)
            if data is None:
                continue

            local_c = tgt_col - offset[0]
            local_r = tgt_row - offset[1]

            result = analyze_spatial_pattern(
                data, (local_c, local_r), bw_px, bh_px,
                label=f"{t['target_id']}|{tiff_name}"
            )
            if result:
                result["target_id"] = t["target_id"]
                result["height_m"] = t["height_m"]
                result["source_tiff"] = tiff_name
                all_results["debris"].append(result)

    # ── Analyze background bright spots ──
    # Pick several background regions with varying characteristics
    bg_regions = [
        ("H11833_1of2.tif", 262500, 3199000, "SW_bg1"),
        ("H11833_1of2.tif", 264500, 3201500, "SW_bg2"),
        ("H11833_2of2.tif", 263500, 3199500, "SW_bg3"),
        ("H11833_1of2.tif", 276000, 3210000, "NE_bg1"),
        ("H11833_2of2.tif", 277000, 3209000, "NE_bg2"),
        ("H11833_1of2.tif", 269500, 3201000, "CE_bg1"),
        ("H11833_1of2.tif", 270500, 3202000, "CE_bg2"),
        ("H11833_2of2.tif", 264000, 3202000, "SW_bg4"),
    ]

    for tiff_name, utm_x, utm_y, desc in bg_regions:
        ds = next((d for name, d in ds_list if name == tiff_name), None)
        if ds is None:
            continue

        inv = ~ds.transform
        col, row = inv * (utm_x, utm_y)

        data, offset = safe_read(ds, col, row, SEARCH_R)
        if data is None:
            continue

        local_c = col - offset[0]
        local_r = row - offset[1]

        # For background, use a "fake" target size (debris-like dimensions)
        result = analyze_spatial_pattern(
            data, (local_c, local_r), 24, 16,  # 12m x 8m in pixels
            label=f"{desc}|{tiff_name}"
        )
        if result:
            result["region"] = desc
            result["source_tiff"] = tiff_name
            all_results["background"].append(result)

    # Close TIFFs
    for _, ds in ds_list:
        ds.close()

    # ── Print results ──
    print("\n--- DEBRIS TARGETS ---")
    print(f"{'ID':8s} {'Tiff':12s} {'sig_score':>9s} {'bright_anom':>11s} "
          f"{'shadow_dep':>10s} {'grad_ratio':>10s} {'couplet':>7s} {'connect':>7s} {'asym':>6s}")
    print("-" * 90)
    for r in all_results["debris"]:
        print(f"{r['target_id']:8s} {r['source_tiff']:12s} {r['signature_score']:>9.3f} "
              f"{r['brightness_anomaly']:>11.3f} {r['shadow_depth']:>10.4f} "
              f"{r['grad_ratio']:>10.3f} {r['couplet_score']:>7.4f} "
              f"{r['shadow_connectivity']:>7.3f} {r['asymmetry']:>6.3f}")

    print("\n--- BACKGROUND SPOTS ---")
    print(f"{'Region':12s} {'sig_score':>9s} {'bright_anom':>11s} "
          f"{'shadow_dep':>10s} {'grad_ratio':>10s} {'couplet':>7s} {'connect':>7s} {'asym':>6s}")
    print("-" * 90)
    for r in all_results["background"]:
        print(f"{r['region']:12s} {r['signature_score']:>9.3f} "
              f"{r['brightness_anomaly']:>11.3f} {r['shadow_depth']:>10.4f} "
              f"{r['grad_ratio']:>10.3f} {r['couplet_score']:>7.4f} "
              f"{r['shadow_connectivity']:>7.3f} {r['asymmetry']:>6.3f}")

    # ── Statistical comparison ──
    d_sig = [r["signature_score"] for r in all_results["debris"]]
    b_sig = [r["signature_score"] for r in all_results["background"]]
    d_bright = [r["brightness_anomaly"] for r in all_results["debris"]]
    b_bright = [r["brightness_anomaly"] for r in all_results["background"]]
    d_grad = [r["grad_ratio"] for r in all_results["debris"]]
    b_grad = [r["grad_ratio"] for r in all_results["background"]]
    d_shadow = [r["shadow_depth"] for r in all_results["debris"]]
    b_shadow = [r["shadow_depth"] for r in all_results["background"]]
    d_conn = [r["shadow_connectivity"] for r in all_results["debris"]]
    b_conn = [r["shadow_connectivity"] for r in all_results["background"]]
    d_asym = [r["asymmetry"] for r in all_results["debris"]]
    b_asym = [r["asymmetry"] for r in all_results["background"]]

    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return 0.0
        pooled = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
        return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: Debris vs Background")
    print("=" * 80)
    print(f"{'Metric':<25} {'Debris μ':>10} {'BG μ':>10} {'Δ':>8} {'p-value':>8} {'Cohen d':>8}")
    print("-" * 80)

    for name, dv, bv in [
        ("signature_score", d_sig, b_sig),
        ("brightness_anomaly", d_bright, b_bright),
        ("grad_ratio", d_grad, b_grad),
        ("shadow_depth", d_shadow, b_shadow),
        ("shadow_connectivity", d_conn, b_conn),
        ("asymmetry", d_asym, b_asym),
    ]:
        _, p = stats.mannwhitneyu(dv, bv, alternative='two-sided')
        d = cohens_d(dv, bv)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:<25} {np.mean(dv):>10.3f} {np.mean(bv):>10.3f} {np.mean(dv)-np.mean(bv):>+8.3f} {p:>8.4f} {d:>+8.3f} {sig}")

    # ── Couplet detection rate ──
    d_couplet = sum(1 for r in all_results["debris"] if r["couplet_detected"])
    b_couplet = sum(1 for r in all_results["background"] if r["couplet_detected"])
    print(f"\nBright-dark couplet detected: Debris {d_couplet}/{len(all_results['debris'])} "
          f"({100*d_couplet/len(all_results['debris']):.0f}%) | "
          f"BG {b_couplet}/{len(all_results['background'])} "
          f"({100*b_couplet/len(all_results['background']):.0f}%)")

    # ── Save results ──
    results_path = os.path.join(out_dir, "combined_signature_analysis.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[✓] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
