#!/usr/bin/env python3
"""
Diagnostic script: Inspect TGT015 center pixel and neighborhood values
to understand why it fails the zero-center check and determine the right fix.
"""
import sys
import re
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

# TGT015 definition
TGT015 = {
    "target_id": "TGT015",
    "name": "Obstruction_Charted_Retained",
    "lat_str": '28° 54\' 18.14" N',
    "lon_str": '089° 25\' 45.09" W',
    "length_m": 12.0,
    "width_m": 8.0,
}

TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]

def dms_to_dd(dms_str):
    parts = re.split(r'[°\'"\s]+', dms_str.strip())
    parts = [p for p in parts if p]
    d = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    s = float(parts[2].rstrip('NSEWnsew')) if len(parts) > 2 else 0.0
    dd = d + m / 60.0 + s / 3600.0
    if 'S' in dms_str.upper() or 'W' in dms_str.upper() or dms_str.startswith('-'):
        dd = -dd
    return dd

transformer = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

lat = dms_to_dd(TGT015["lat_str"])
lon = dms_to_dd(TGT015["lon_str"])
if lon > 0:
    lon = -lon

utm_x, utm_y = transformer.transform(lon, lat)
print(f"TGT015 coordinates: lat={lat:.8f}, lon={lon:.8f}, utm_x={utm_x:.2f}, utm_y={utm_y:.2f}")

for tiff_path in TIFF_FILES:
    print(f"\n{'='*60}")
    print(f"Inspecting: {tiff_path}")
    with rasterio.open(tiff_path) as ds:
        # Convert UTM to pixel
        inv_transform = ~ds.transform
        tgt_col, tgt_row = inv_transform * (utm_x, utm_y)
        print(f"  Pixel position: col={tgt_col:.4f}, row={tgt_row:.4f}")
        print(f"  Raster size: {ds.width} x {ds.height}")

        # Exact center pixel
        c_col = int(round(tgt_col))
        c_row = int(round(tgt_row))
        print(f"  Nearest integer pixel: col={c_col}, row={c_row}")

        if 0 <= c_col < ds.width and 0 <= c_row < ds.height:
            center_val = ds.read(1, window=Window(c_col, c_row, 1, 1))[0, 0]
            print(f"  Center pixel value: {center_val}")
        else:
            print(f"  Center pixel OUT OF BOUNDS")
            continue

        # Neighborhood analysis
        for kernel_size in [3, 5, 7, 9, 11, 15]:
            half = kernel_size // 2
            kx1 = c_col - half
            ky1 = c_row - half
            kx2 = kx1 + kernel_size
            ky2 = ky1 + kernel_size

            # Clamp to raster bounds
            kx1_c = max(0, kx1)
            ky1_c = max(0, ky1)
            kx2_c = min(ds.width, kx2)
            ky2_c = min(ds.height, ky2)

            if kx1_c >= kx2_c or ky1_c >= ky2_c:
                print(f"  {kernel_size}x{kernel_size} kernel: OUT OF BOUNDS")
                continue

            window = Window(kx1_c, ky1_c, kx2_c - kx1_c, ky2_c - ky1_c)
            data = ds.read(1, window=window)
            total = data.size
            nz = np.count_nonzero(data)
            nz_ratio = nz / total
            mean_val = float(np.mean(data))
            std_val = float(np.std(data))
            min_val = int(np.min(data))
            max_val = int(np.max(data))
            nonzero_mean = float(np.mean(data[data > 0])) if nz > 0 else 0

            print(f"  {kernel_size}x{kernel_size} kernel: "
                  f"nonzero={nz}/{total} ({nz_ratio:.3f}), "
                  f"mean={mean_val:.1f}, nonzero_mean={nonzero_mean:.1f}, "
                  f"std={std_val:.1f}, min={min_val}, max={max_val}")

        # Check the actual pixel neighborhood around TGT015
        print(f"\n  Raw pixel values around center ({c_col},{c_row}):")
        for dy in range(-3, 4):
            row_vals = []
            for dx in range(-3, 4):
                px, py = c_col + dx, c_row + dy
                if 0 <= px < ds.width and 0 <= py < ds.height:
                    v = ds.read(1, window=Window(px, py, 1, 1))[0, 0]
                    row_vals.append(str(v).rjust(5))
                else:
                    row_vals.append("  OOB")
            prefix = f"    row {c_row + dy:+d}: "
            print(prefix + " ".join(row_vals))

        # Also check in the shifted crop window that prepare_noaa_e3.py would use
        # TGT015 is in SW_Pass_Main_Cluster -> train -> will get offsets
        # Check a 512x512 window centered at TGT015
        print(f"\n  512x512 crop window centered at TGT015:")
        win_col = int(round(tgt_col - 256))
        win_row = int(round(tgt_row - 256))
        win_col = max(0, min(ds.width - 512, win_col))
        win_row = max(0, min(ds.height - 512, win_row))
        window = Window(win_col, win_row, 512, 512)
        data = ds.read(1, window=window)
        print(f"    Window: col={win_col}, row={win_row}")
        print(f"    Total pixels: {data.size}, Nonzero: {np.count_nonzero(data)} ({np.count_nonzero(data)/data.size:.3f})")
        print(f"    Mean: {np.mean(data):.1f}, Std: {np.std(data):.1f}")
        print(f"    Min: {np.min(data)}, Max: {np.max(data)}")

        # Target bounding box in this crop
        bw = TGT015["length_m"] / 0.5  # 24 pixels
        bh = TGT015["width_m"] / 0.5   # 16 pixels
        tc_x = tgt_col - win_col
        tc_y = tgt_row - win_row
        tx1 = tc_x - bw/2
        tx2 = tc_x + bw/2
        ty1 = tc_y - bh/2
        ty2 = tc_y + bh/2
        print(f"    Target bbox in crop: x1={tx1:.1f}, y1={ty1:.1f}, x2={tx2:.1f}, y2={ty2:.1f}")
        print(f"    Target bbox valid: {0 <= tx1 and 0 <= ty1 and tx2 <= 512 and ty2 <= 512}")

        # Check non-zero values within the target bounding box region
        target_data = data[max(0,int(ty1)):min(512,int(ty2)), max(0,int(tx1)):min(512,int(tx2))]
        if target_data.size > 0:
            print(f"    Target region: size={target_data.size}, nonzero={np.count_nonzero(target_data)}, mean={np.mean(target_data):.1f}")
        else:
            print(f"    Target region: EMPTY")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
