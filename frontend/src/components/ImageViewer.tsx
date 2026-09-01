import React, { useState, useRef, useMemo } from 'react';
import { Layers, Eye, Crosshair, ZoomIn, ZoomOut, RotateCcw, Image as ImageIcon, Palette, ShieldCheck, Maximize2 } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface ImageViewerProps {
  analysis: AnalysisResponse | null;
  hoveredDetectionId?: number | null;
  selectedDetectionId?: number | null;
  onSelectDetection?: (id: number | null) => void;
  isAnalyzing: boolean;
}

type ViewMode = 'annotated' | 'original' | 'mask' | 'colormap' | 'evidence';

export const ImageViewer: React.FC<ImageViewerProps> = ({
  analysis,
  isAnalyzing,
  selectedDetectionId,
  onSelectDetection,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('annotated');
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [showZoomCrop, setShowZoomCrop] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsPanning(true);
      setStartPos({ x: e.clientX - position.x, y: e.clientY - position.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPosition({ x: e.clientX - startPos.x, y: e.clientY - startPos.y });
    }
  };

  const handleMouseUp = () => setIsPanning(false);

  const getActiveImageUrl = () => {
    if (!analysis) return null;
    switch (viewMode) {
      case 'annotated':
        return analysis.annotated_image_url;
      case 'original':
        return analysis.original_image_url;
      case 'mask':
        return analysis.detection_mask_url;
      case 'colormap':
        return analysis.colormap_image_url;
      case 'evidence':
        return analysis.evidence_image_url;
      default:
        return analysis.annotated_image_url;
    }
  };

  // Find the selected detection for the zoom crop panel
  const selectedDetection = useMemo(() => {
    if (!analysis || selectedDetectionId == null) return null;
    return analysis.detections.find((d) => d.id === selectedDetectionId) || null;
  }, [analysis, selectedDetectionId]);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded flex flex-col h-full overflow-hidden">
      {/* Top Toolbar */}
      <div className="border-b border-slate-800 bg-slate-950/60 p-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Layer Mode Switcher */}
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-0.5 rounded font-mono text-xs flex-wrap">
          <button
            type="button"
            onClick={() => setViewMode('annotated')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'annotated'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Crosshair className="h-3 w-3" />
            Annotated
          </button>
          <button
            type="button"
            onClick={() => setViewMode('original')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'original'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="h-3 w-3" />
            Source
          </button>
          <button
            type="button"
            onClick={() => setViewMode('mask')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'mask'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3 w-3" />
            Spotlight
          </button>
          <button
            type="button"
            onClick={() => setViewMode('colormap')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'colormap'
                ? 'bg-purple-500 text-white font-bold shadow-[0_0_10px_rgba(168,85,247,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Palette className="h-3 w-3" />
            Colormap
          </button>
          <button
            type="button"
            onClick={() => setViewMode('evidence')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'evidence'
                ? 'bg-emerald-500 text-white font-bold shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ShieldCheck className="h-3 w-3" />
            Evidence
          </button>
        </div>

        {/* Right side: Zoom crop toggle + Zoom controls */}
        <div className="flex items-center gap-2 font-mono text-xs">
          {/* Zoom Crop Toggle */}
          {analysis && analysis.detections.length > 0 && (
            <button
              type="button"
              onClick={() => setShowZoomCrop(!showZoomCrop)}
              className={`px-2 py-1.5 rounded flex items-center gap-1.5 border transition-all ${
                showZoomCrop
                  ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
              title="Toggle Detection Zoom Panel"
            >
              <Maximize2 className="h-3.5 w-3.5" />
              Zoom
            </button>
          )}

          {/* Viewport Zoom Controls */}
          <button
            type="button"
            onClick={handleZoomIn}
            className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
            title="Zoom In"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={handleZoomOut}
            className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
            title="Zoom Out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={handleResetZoom}
            className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
            title="Reset Viewport"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
          <span className="px-2 py-1 text-[11px] text-slate-400 bg-slate-950 border border-slate-800 rounded">
            {Math.round(zoom * 100)}%
          </span>
        </div>
      </div>

      {/* Main Content: Image + Optional Zoom Crop Panel */}
      <div className="flex-1 flex overflow-hidden">
        {/* Image Canvas */}
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className={`relative flex-1 bg-slate-950 flex items-center justify-center min-h-[420px] overflow-hidden cursor-grab active:cursor-grabbing bg-grid-pattern ${
            showZoomCrop && selectedDetection ? 'lg:w-2/3' : 'w-full'
          }`}
        >
          {isAnalyzing && (
            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center gap-3">
              <div className="h-10 w-10 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-sm font-mono text-cyan-300 font-semibold tracking-wider">
                RUNNING YOLO-ESI INFERENCE...
              </p>
              <p className="text-xs text-slate-400 font-mono">
                Extracting acoustic/spatial features & georeferencing
              </p>
            </div>
          )}

          {analysis ? (
            <div
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
                transition: isPanning ? 'none' : 'transform 0.15s ease-out',
              }}
              className="relative select-none max-w-full max-h-full flex items-center justify-center p-4"
            >
              <img
                src={getActiveImageUrl() || ''}
                alt="Analyzed Imagery"
                className="max-h-[520px] w-auto object-contain rounded border border-slate-800 shadow-2xl pointer-events-none"
              />
            </div>
          ) : (
            <div className="text-center p-8 flex flex-col items-center gap-3 text-slate-500">
              <div className="h-16 w-16 rounded border border-dashed border-slate-800 flex items-center justify-center">
                <ImageIcon className="h-8 w-8 text-slate-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-400">
                  Awaiting Imagery Ingestion
                </p>
                <p className="text-xs text-slate-600 mt-1 max-w-md">
                  Upload a GeoTIFF or sonar image, or select a preloaded mission dataset scenario on the left.
                </p>
              </div>
            </div>
          )}

          {/* Bottom Overlay Telemetry Badges */}
          {analysis && (
            <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 pointer-events-none z-10">
              <div className="flex items-center gap-2 font-mono text-[11px] bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded backdrop-blur text-slate-300">
                <span className="text-slate-500">FILE:</span>
                <span className="text-cyan-300">{analysis.file_metadata.filename}</span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-400">{analysis.file_metadata.width} × {analysis.file_metadata.height} px</span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-400">{analysis.file_metadata.file_size_human}</span>
              </div>

              <div className="flex items-center gap-2 font-mono text-[11px] bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded backdrop-blur text-slate-300">
                <span className="text-slate-500">TARGETS:</span>
                <span className="text-emerald-400 font-bold">{analysis.summary.total_detections}</span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-500">LATENCY:</span>
                <span className="text-amber-400">{analysis.summary.inference_time_ms} ms</span>
              </div>
            </div>
          )}
        </div>

        {/* Zoom Crop Panel — shows magnified detection when selected */}
        {showZoomCrop && selectedDetection && analysis && (
          <div className="hidden lg:flex w-1/3 border-l border-slate-800 bg-slate-950/80 flex-col overflow-hidden">
            {/* Panel Header */}
            <div className="border-b border-slate-800 px-3 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Maximize2 className="h-3.5 w-3.5 text-emerald-400" />
                <span className="font-mono text-xs text-emerald-300 font-semibold">
                  DETAIL #{String(selectedDetection.id).padStart(2, '0')}
                </span>
              </div>
              <button
                type="button"
                onClick={() => onSelectDetection?.(null)}
                className="text-slate-500 hover:text-slate-300 font-mono text-xs"
              >
                ✕
              </button>
            </div>

            {/* Zoomed Crop */}
            <div className="flex-1 p-3 flex items-center justify-center bg-slate-950">
              <ZoomCropView
                imageUrl={getActiveImageUrl() || ''}
                bbox={selectedDetection.bbox}
                imageWidth={analysis.file_metadata.width}
                imageHeight={analysis.file_metadata.height}
              />
            </div>

            {/* Detection Info */}
            <div className="border-t border-slate-800 px-3 py-2.5 font-mono text-[11px] space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-500">Class</span>
                <span className="text-slate-200">{selectedDetection.class_name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Confidence</span>
                <span className="text-emerald-400 font-bold">
                  {(selectedDetection.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Bounding Box</span>
                <span className="text-slate-300">
                  [{Math.round(selectedDetection.bbox.x1)}, {Math.round(selectedDetection.bbox.y1)},
                   {Math.round(selectedDetection.bbox.x2)}, {Math.round(selectedDetection.bbox.y2)}]
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Center</span>
                <span className="text-slate-300">
                  ({Math.round(selectedDetection.center_pixel.x)}, {Math.round(selectedDetection.center_pixel.y)})
                </span>
              </div>
              {selectedDetection.geolocation && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Location</span>
                  <span className="text-cyan-300">
                    {selectedDetection.geolocation.latitude.toFixed(6)}°, {selectedDetection.geolocation.longitude.toFixed(6)}°
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Placeholder when zoom panel is open but no detection selected */}
        {showZoomCrop && !selectedDetection && analysis && analysis.detections.length > 0 && (
          <div className="hidden lg:flex w-1/3 border-l border-slate-800 bg-slate-950/80 flex-col items-center justify-center p-6">
            <div className="text-center">
              <Maximize2 className="h-8 w-8 text-slate-600 mx-auto mb-3" />
              <p className="text-xs text-slate-400 font-mono">
                Select a detection to<br />view magnified crop
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * ZoomCropView — renders a magnified crop of the detection region
 * using CSS object-position and scale to zoom into the bounding box area.
 */
const ZoomCropView: React.FC<{
  imageUrl: string;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  imageWidth: number;
  imageHeight: number;
}> = ({ imageUrl, bbox, imageWidth, imageHeight }) => {
  const cropRef = useRef<HTMLDivElement>(null);

  // Calculate crop region as percentages for CSS object-position/object-fit
  const boxW = bbox.x2 - bbox.x1;
  const boxH = bbox.y2 - bbox.y1;
  const centerX = ((bbox.x1 + boxW / 2) / imageWidth) * 100;
  const centerY = ((bbox.y1 + boxH / 2) / imageHeight) * 100;

  // Scale factor: how much to zoom in (bigger box = less zoom needed)
  const boxFraction = Math.max(boxW / imageWidth, boxH / imageHeight);
  const scaleFactor = Math.min(4, Math.max(1.5, 1 / boxFraction));

  return (
    <div
      ref={cropRef}
      className="relative w-full h-full rounded border border-emerald-500/30 overflow-hidden bg-slate-900"
    >
      <img
        src={imageUrl}
        alt="Detection Crop"
        className="w-full h-full object-cover pointer-events-none"
        style={{
          objectPosition: `${centerX}% ${centerY}%`,
          objectFit: 'cover',
          transform: `scale(${scaleFactor})`,
          transformOrigin: `${centerX}% ${centerY}%`,
        }}
      />
      {/* Crosshair overlay */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-0 right-0 h-px bg-emerald-400/30" />
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-emerald-400/30" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 border border-emerald-400/60 rounded-full" />
      </div>
      {/* Label */}
      <div className="absolute top-2 left-2 bg-slate-950/90 border border-emerald-500/40 rounded px-2 py-1 font-mono text-[10px] text-emerald-300">
        {Math.round(boxW)}×{Math.round(boxH)} px — {Math.round(scaleFactor)}× zoom
      </div>
    </div>
  );
};
