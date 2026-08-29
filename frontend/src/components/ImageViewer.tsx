import React, { useState, useRef } from 'react';
import { Layers, Eye, Crosshair, ZoomIn, ZoomOut, RotateCcw, Image as ImageIcon } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface ImageViewerProps {
  analysis: AnalysisResponse | null;
  hoveredDetectionId?: number | null;
  selectedDetectionId?: number | null;
  onSelectDetection?: (id: number | null) => void;
  isAnalyzing: boolean;
}

type ViewMode = 'annotated' | 'original' | 'mask';

export const ImageViewer: React.FC<ImageViewerProps> = ({
  analysis,
  isAnalyzing,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('annotated');
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });

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
      default:
        return analysis.annotated_image_url;
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded flex flex-col h-full overflow-hidden">
      {/* Top Toolbar */}
      <div className="border-b border-slate-800 bg-slate-950/60 p-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Layer Mode Switcher */}
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-0.5 rounded font-mono text-xs">
          <button
            type="button"
            onClick={() => setViewMode('annotated')}
            className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'annotated'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Crosshair className="h-3 w-3" />
            Annotated View
          </button>
          <button
            type="button"
            onClick={() => setViewMode('original')}
            className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'original'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="h-3 w-3" />
            Source Raw
          </button>
          <button
            type="button"
            onClick={() => setViewMode('mask')}
            className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
              viewMode === 'mask'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3 w-3" />
            Target Isolation
          </button>
        </div>

        {/* Viewport Zoom Controls */}
        <div className="flex items-center gap-1 font-mono text-xs">
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

      {/* Main Imagery Canvas Area */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="relative flex-1 bg-slate-950 flex items-center justify-center min-h-[420px] overflow-hidden cursor-grab active:cursor-grabbing bg-grid-pattern"
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
    </div>
  );
};
