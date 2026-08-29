import os
import unittest
from pathlib import Path
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from pyproj import Transformer

from backend.geospatial.metadata import extract_geospatial_metadata, NOAA_PARENT_TIFF_METADATA
from backend.geospatial.coordinates import pixel_to_geographic
from backend.api.routes_analysis import run_full_pipeline

class TestGeospatialPipeline(unittest.TestCase):

    def test_raw_geotiff_source(self):
        """Verify direct GeoTIFF file metadata extraction."""
        path = "datasets/noaa-debris/raw/H11833/H11833_1of2.tif"
        if os.path.exists(path):
            meta = extract_geospatial_metadata(path, orig_filename="H11833_1of2.tif")
            self.assertTrue(meta["georeferenced"])
            self.assertEqual(meta["crs"], "EPSG:26916")
            self.assertEqual(meta["coordinate_source"], "GeoTIFF Affine Transform")
            self.assertIsNotNone(meta["transform"])
            self.assertIsNotNone(meta["bounds"])

    def test_sample_geotiff_crop(self):
        """Verify sample GeoTIFF crop."""
        path = "backend/static/samples/sample_noaa_geotiff_debris.tif"
        meta = extract_geospatial_metadata(path, orig_filename="sample_noaa_geotiff_debris.tif")
        self.assertTrue(meta["georeferenced"])
        self.assertEqual(meta["crs"], "EPSG:26916")
        self.assertEqual(meta["coordinate_source"], "GeoTIFF Affine Transform")
        self.assertTrue(meta["lat_lon_available"])

    def test_unseen_derived_noaa_crop(self):
        """Verify derived NOAA crop (UNSEEN_0011.png) parent raster reconstruction."""
        path = "datasets/noaa-debris/h8_unseen_test/images/test/UNSEEN_0011.png"
        if os.path.exists(path):
            meta = extract_geospatial_metadata(path, orig_filename="UNSEEN_0011.png")
            self.assertTrue(meta["georeferenced"])
            self.assertEqual(meta["crs"], "EPSG:26916")
            self.assertIn("NOAA Survey Parent Raster Reconstruction", meta["coordinate_source"])
            self.assertEqual(meta["transform"].a, 0.5)
            self.assertEqual(meta["transform"].e, -0.5)
            self.assertEqual(meta["transform"].c, 263328.75)
            self.assertEqual(meta["transform"].f, 3201299.75)

    def test_known_point_accuracy(self):
        """Verify coordinate accuracy for a known pixel center in NOAA H11833."""
        crop_trans = Affine(0.5, 0.0, 263328.75, 0.0, -0.5, 3201299.75)
        
        # Crop pixel (361.75, 264.5)
        cx, cy = 361.75, 264.5
        proj_x, proj_y = crop_trans * (cx, cy)
        self.assertAlmostEqual(proj_x, 263509.625, places=3)
        self.assertAlmostEqual(proj_y, 3201167.500, places=3)

        # pyproj transformation
        transformer = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(proj_x, proj_y)
        self.assertAlmostEqual(lat, 28.9166026, places=5)
        self.assertAlmostEqual(lon, -89.4257008, places=5)

    def test_non_georeferenced_image(self):
        """Verify non-georeferenced image is correctly reported without fake coordinates."""
        path = "backend/static/samples/sample_seabed_background.png"
        meta = extract_geospatial_metadata(path, orig_filename="sample_seabed_background.png")
        self.assertFalse(meta["georeferenced"])
        self.assertIsNone(meta["crs"])
        self.assertIsNone(meta["transform"])
        self.assertIsNone(meta["bounds"])
        self.assertFalse(meta["lat_lon_available"])

    def test_exif_gps_image(self):
        """Verify EXIF GPS camera positioning without fabricating detection coordinates."""
        path = "backend/static/samples/sample_drone_aerial_geotagged.jpg"
        meta = extract_geospatial_metadata(path, orig_filename="sample_drone_aerial_geotagged.jpg")
        self.assertTrue(meta["georeferenced"])
        self.assertEqual(meta["coordinate_source"], "EXIF GPS")
        self.assertIsNotNone(meta["camera_latitude"])
        self.assertIsNotNone(meta["camera_longitude"])
        self.assertIsNone(meta["transform"])

        # Object-level detection coordinates must NOT be produced from camera GPS
        geo_det = pixel_to_geographic(100.0, 100.0, meta)
        self.assertIsNone(geo_det)

    def test_full_pipeline_execution(self):
        """Verify end-to-end pipeline on UNSEEN_0011.png."""
        path = "datasets/noaa-debris/h8_unseen_test/images/test/UNSEEN_0011.png"
        if os.path.exists(path):
            res = run_full_pipeline(path, "UNSEEN_0011.png", file_size_bytes=100000)
            self.assertTrue(res.summary.debris_detected)
            self.assertGreater(len(res.detections), 0)
            det = res.detections[0]
            self.assertIsNotNone(det.geolocation)
            self.assertAlmostEqual(det.geolocation.latitude, 28.9166026, places=5)
            self.assertAlmostEqual(det.geolocation.longitude, -89.4257008, places=5)

if __name__ == "__main__":
    unittest.main()
