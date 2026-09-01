export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface CenterPixel {
  x: number;
  y: number;
}

export interface Geolocation {
  latitude: number;
  longitude: number;
  crs: string;
  coordinate_source: string;
  utm_x?: number | null;
  utm_y?: number | null;
}

export interface DetectionRecord {
  id: number;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  center_pixel: CenterPixel;
  geolocation?: Geolocation | null;
}

export interface FileMetadata {
  filename: string;
  format: string;
  width: number;
  height: number;
  file_size_bytes: number;
  file_size_human: string;
  channels: number;
}

export interface GeospatialMetadata {
  georeferenced: boolean;
  crs?: string | null;
  bounds?: {
    left?: number;
    bottom?: number;
    right?: number;
    top?: number;
    wgs84_min_lon?: number;
    wgs84_min_lat?: number;
    wgs84_max_lon?: number;
    wgs84_max_lat?: number;
  } | null;
  pixel_resolution?: [number, number] | null;
  lat_lon_available: boolean;
  coordinate_source?: string | null;
  status_message: string;
  camera_latitude?: number | null;
  camera_longitude?: number | null;
  camera_altitude?: number | null;
  capture_direction?: number | null;
  footprint_geojson?: any | null;
}

export interface ModelMetadata {
  model_name: string;
  format: string;
  input_resolution: string;
  execution_provider: string;
  model_path: string;
  sha256_hash: string;
  classes: Record<number, string>;
  architecture: string;
  attention_mechanism: string;
}

export interface AnalysisSummary {
  debris_detected: boolean;
  total_detections: number;
  highest_confidence?: number | null;
  average_confidence?: number | null;
  class_counts: Record<string, number>;
  inference_time_ms: number;
  total_time_ms: number;
  status: string;
  message: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  timestamp: string;
  summary: AnalysisSummary;
  detections: DetectionRecord[];
  file_metadata: FileMetadata;
  geospatial_metadata: GeospatialMetadata;
  model_metadata: ModelMetadata;
  original_image_url: string;
  annotated_image_url: string;
  detection_mask_url: string;
  colormap_image_url: string;
  evidence_image_url: string;
  csv_export_url: string;
  json_export_url: string;
  pdf_report_url: string;
}

export interface SampleItem {
  id: string;
  name: string;
  type: string;
  description: string;
  has_geolocation: boolean;
  filename: string;
}
