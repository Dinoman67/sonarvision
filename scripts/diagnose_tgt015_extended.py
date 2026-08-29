#!/usr/bin/env python3
"""
Extended diagnostic: Find the actual sonar return for TGT015.
Search progressively larger neighborhoods and different offsets from the annotated center.
"""
import sys
import re
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

TGT015 = {
    "target_id": "TGT015",
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

# Target dimensions in pixels at 0.5m/px
bw = TGT015["length_m"] / 0.5  # 24 pixels
bh = TGT015["width_m"] / 0.5   # 16 pixels

for tiff_path in TIFF_FILES:
    print(f"\n{'='*70}")
    print(f"FILE: {tiff_path}")
    with rasterio.open(tiff_path) as ds:
        inv_transform = ~ds.transform
        tgt_col, tgt_row = inv_transform * (utm_x, utm_y)
        print(f"Target center pixel: col={tgt_col:.4f}, row={tgt_row:.4f}")

        # Search for first nonzero pixel in expanding neighborhoods
        print(f"\nSearching for nonzero pixels in expanding neighborhoods from center:")
        for radius in [10, 20, 30, 40, 50, 60, 75, 100, 120, 150]:
            c = int(round(tgt_col))
            r = int(round(tgt_row))
            x1 = max(0, c - radius)
            y1 = max(0, r - radius)
            x2 = min(ds.width, c + radius)
            y2 = min(ds.height, r + radius)
            data = ds.read(1, window=Window(x1, y1, x2-x1, y2-y1))
            nz = np.count_nonzero(data)
            total = data.size
            if nz > 0:
                # Find the bounding box of nonzero pixels
                nonzero_mask = data > 0
                rows_with_nz = np.any(nonzero_mask, axis=1)
                cols_with_nz = np.any(nonzero_mask, axis=0)
                if np.any(rows_with_nz) and np.any(cols_with_nz):
                    r_min_rel = np.argmax(rows_with_nz)
                    r_max_rel = len(rows_with_nz) - np.argmax(rows_with_nz[::-1]) - 1
                    c_min_rel = np.argmax(cols_with_nz)
                    c_max_rel = len(cols_with_nz) - np.argmax(cols_with_nz[::-1]) - 1
                    # Convert to absolute pixel coords
                    r_min_abs = y1 + r_min_rel
                    r_max_abs = y1 + r_max_rel
                    c_min_abs = x1 + c_min_rel
                    c_max_abs = x1 + c_max_rel
                    # Center of nonzero region
                    nz_center_c = (c_min_abs + c_max_abs) / 2.0
                    nz_center_r = (r_min_abs + r_max_abs) / 2.0
                    offset_c = nz_center_c - tgt_col
                    offset_r = nz_center_r - tgt_row
                    print(f"  radius={radius:3d}: nonzero={nz}/{total} ({nz/total:.3f}), "
                          f"nz_bbox=({c_min_abs},{r_min_abs})-({c_max_abs},{r_max_abs}), "
                          f"nz_center=({nz_center_c:.1f},{nz_center_r:.1f}), "
                          f"offset_from_target=({offset_c:+.1f},{offset_r:+.1f})")
                else:
                    print(f"  radius={radius:3d}: nonzero={nz}/{total} but no bbox found")
                break  # Found nonzero
            else:
                print(f"  radius={radius:3d}: ALL zeros ({nz}/{total})")

        # What does the 1024x1024 SSS patch look like?
        print(f"\n1024x1024 SSS patch centered at TGT015 (like prepare_noaa_sss.py):")
        ss_win_col = int(tgt_col - 512)
        ss_win_row = int(tgt_row - 512)
        ss_win_col = max(0, min(ds.width - 1024, ss_win_col))
        ss_win_row = max(0, min(ds.height - 1024, ss_win_row))
        data_1024 = ds.read(1, window=Window(ss_win_col, ss_win_row, 1024, 1024))
        nz = np.count_nonzero(data_1024)
        print(f"  Window: col={ss_win_col}, row={ss_win_row}")
        print(f"  Nonzero: {nz}/{data_1024.size} ({nz/data_1024.size:.3f})")
        print(f"  Mean: {np.mean(data_1024):.1f}, Std: {np.std(data_1024):.1f}")

        # Now check various offsets the SSS script would use (9 offsets for train)
        print(f"\nPatch validity at each SSS-style offset:")
        offsets = [
            (0, 0), (-128, -128), (128, 128), (-128, 128), (128, -128),
            (-256, 0), (256, 0), (0, -256), (0, 256)
        ]
        for off_x, off_y in offsets:
            c = int(tgt_col - 512 + off_x)
            r = int(tgt_row - 512 + off_y)
            if c < 0 or r < 0 or c + 1024 > ds.width or r + 1024 > ds.height:
                print(f"  offset ({off_x:+d},{off_y:+d}): OUT OF BOUNDS")
                continue
            data = ds.read(1, window=Window(c, r, 1024, 1024))
            nz = np.count_nonzero(data)
            vr = nz / data.size
            mean = np.mean(data)
            ok = "PASS" if vr >= 0.70 and mean >= 15 else "FAIL"
            # Check if target center is within patch
            rel_c = tgt_col - c
            rel_r = tgt_row - r
            center_in = 0 <= rel_c <= 1024 and 0 <= rel_r <= 1024
            print(f"  offset ({off_x:+d},{off_y:+d}): vr={vr:.3f}, mean={mean:.1f}, "
                  f"target_center_in_patch={center_in}, {ok}")

        # Also check a shifted 512x512 E3 crop where the target center is NOT at the center
        print(f"\nTesting shifted E3 512x512 crops:")
        for off_x, off_y in [(0, 0), (-64, 0), (64, 0), (0, -64), (0, 64),
                              (-128, 0), (128, 0), (0, -128), (0, 128)]:
            c = int(round(tgt_col - 256 + off_x))
            r = int(round(tgt_row - 256 + off_y))
            c = max(0, min(ds.width - 512, c))
            r = max(0, min(ds.height - 512, r))
            data = ds.read(1, window=Window(c, r, 512, 512))
            nz = np.count_nonzero(data)
            vr = nz / data.size
            mean_val = np.mean(data)
            # Check if target bbox falls in this crop
            tx1 = tgt_col - bw/2 - c
            tx2 = tgt_col + bw/2 - c
            ty1 = tgt_row - bh/2 - r
            ty2 = tgt_row + bh/2 - r
            bbox_valid = 0 <= tx1 and 0 <= ty1 and tx2 <= 512 and ty2 <= 512
            # Check values within target bbox
            target_data = data[max(0,int(ty1)):min(512,int(ty2)), max(0,int(tx1)):min(512,int(tx2))]
            target_nz = np.count_nonzero(target_data)
            print(f"  offset ({off_x:+d},{off_y:+d}): vr={vr:.3f}, mean={mean_val:.1f}, "
                  f"bbox_valid={bbox_valid}, target_nz={target_nz}/{target_data.size}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
