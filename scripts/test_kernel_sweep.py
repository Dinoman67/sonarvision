#!/usr/bin/env python3
"""Quick kernel radius sweep for TGT015"""
import re, numpy as np, rasterio
from rasterio.windows import Window
from pyproj import Transformer

TGT015 = {"lat_str": '28° 54\' 18.14" N', "lon_str": '089° 25\' 45.09" W'}
TIFF_FILES = [
    "datasets/noaa-debris/raw/H11833/H11833_1of2.tif",
    "datasets/noaa-debris/raw/H11833/H11833_2of2.tif",
]

def dms_to_dd(s):
    parts = re.split(r'[°\'"\s]+', s.strip())
    parts = [p for p in parts if p]
    d,m,sec = float(parts[0]), float(parts[1]) if len(parts)>1 else 0, float(parts[2].rstrip('NSEWnsew')) if len(parts)>2 else 0
    dd = d + m/60 + sec/3600
    if 'S' in s.upper() or 'W' in s.upper(): dd = -dd
    return dd

tf = Transformer.from_crs("EPSG:4326","EPSG:26916",always_xy=True)
lat,lon = dms_to_dd(TGT015["lat_str"]), dms_to_dd(TGT015["lon_str"])
if lon > 0: lon = -lon
ux,uy = tf.transform(lon,lat)

for tpath in TIFF_FILES:
    print(f"\n{tpath}")
    with rasterio.open(tpath) as ds:
        inv = ~ds.transform
        tc,tr = inv*(ux,uy)
        c,r = int(round(tc)), int(round(tr))
        for radius in range(5, 65, 5):
            x1,y1 = max(0,c-radius), max(0,r-radius)
            x2,y2 = min(ds.width,c+radius+1), min(ds.height,r+radius+1)
            data = ds.read(1, window=Window(x1,y1,x2-x1,y2-y1))
            total = data.size
            nz = np.count_nonzero(data)
            vr = nz/total if total>0 else 0
            marker = " <-- FIRST NONZERO" if vr > 0 and radius <= 35 else ""
            print(f"  r={radius:2d}: kernel={2*radius+1}x{2*radius+1}, nz={nz}/{total}, vr={vr:.4f}{marker}")
            if vr >= 0.05:
                # Found it - now check minimum radius more precisely
                break
