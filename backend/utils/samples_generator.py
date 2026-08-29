import os
import shutil
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

def ensure_sample_assets():
    """
    Ensures preloaded sample assets are present in backend/static/samples/
    """
    samples_dir = Path("/home/ashish/sonar-vision/backend/static/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real Georeferenced GeoTIFF crop from NOAA survey
    geotiff_out = samples_dir / "sample_noaa_geotiff_debris.tif"
    raw_tif = "/home/ashish/sonar-vision/datasets/noaa-debris/raw/H11833/H11833_1of2.tif"
    if not geotiff_out.exists() and os.path.exists(raw_tif):
        try:
            with rasterio.open(raw_tif) as src:
                col_off = 29200
                row_off = 10500
                crop_w, crop_h = 512, 512
                win = Window(col_off, row_off, crop_w, crop_h)
                data = src.read(window=win)
                win_transform = rasterio.windows.transform(win, src.transform)
                
                profile = src.profile.copy()
                profile.update({
                    "height": crop_h,
                    "width": crop_w,
                    "transform": win_transform
                })
                with rasterio.open(geotiff_out, "w", **profile) as dst:
                    dst.write(data)
                print(f"[Samples] Created sample GeoTIFF: {geotiff_out}")
        except Exception as e:
            print(f"[Samples] Error creating sample GeoTIFF: {e}")

    # 2. Real NOAA SSS Debris PNG
    sample_debris_png = samples_dir / "sample_sss_marine_debris.png"
    src_png_debris = "/home/ashish/sonar-vision/datasets/noaa-debris/e3_enhanced/images/train/E3_H11833_TGT014_0011.png"
    if not sample_debris_png.exists() and os.path.exists(src_png_debris):
        shutil.copy2(src_png_debris, sample_debris_png)

    # 3. Real NOAA Seabed Background PNG (No Debris)
    sample_bg_png = samples_dir / "sample_seabed_background.png"
    src_png_bg = "/home/ashish/sonar-vision/datasets/noaa-debris/e3_enhanced/images/train/E3_H11833_BG_0017.png"
    if not sample_bg_png.exists() and os.path.exists(src_png_bg):
        shutil.copy2(src_png_bg, sample_bg_png)

    # 4. Geotagged Drone/Aerial JPG
    sample_drone_jpg = samples_dir / "sample_drone_aerial_geotagged.jpg"
    if not sample_drone_jpg.exists() and os.path.exists(src_png_debris):
        try:
            im = Image.open(src_png_debris).convert("RGB")
            exif = im.getexif()
            gps_ifd = exif.get_ifd(0x8825)
            gps_ifd[1] = "N"
            gps_ifd[2] = (28, 54, 30.07)
            gps_ifd[3] = "W"
            gps_ifd[4] = (89, 26, 3.29)
            im.save(sample_drone_jpg, exif=exif)
            print(f"[Samples] Created sample geotagged aerial image: {sample_drone_jpg}")
        except Exception as e:
            print(f"[Samples] Error creating sample drone image: {e}")

if __name__ == "__main__":
    ensure_sample_assets()
