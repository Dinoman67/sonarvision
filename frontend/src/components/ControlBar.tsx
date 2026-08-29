import React from 'react';
import { Sliders, RefreshCw, Grid } from 'lucide-react';

interface ControlBarProps {
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
  iouThreshold: number;
  setIouThreshold: (val: number) => void;
  useTiling: boolean;
  setUseTiling: (val: boolean) => void;
  onReanalyze: () => void;
  isAnalyzing: boolean;
  canReanalyze: boolean;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  confidenceThreshold,
  setConfidenceThreshold,
  iouThreshold,
  setIouThreshold,
  useTiling,
  setUseTiling,
  onReanalyze,
  isAnalyzing,
  canReanalyze,
}) => {
  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded p-3 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
      <div className="flex flex-wrap items-center gap-6">
        {/* Confidence Threshold Slider */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Sliders className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-300 font-semibold">CONFIDENCE:</span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="w-28 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 font-mono border border-slate-700 min-w-[44px] text-center">
            {(confidenceThreshold * 100).toFixed(0)}%
          </span>
        </div>

        {/* IoU Threshold Slider */}
        <div className="flex items-center gap-3">
          <span className="text-slate-400 font-semibold">IOU NMS:</span>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={iouThreshold}
            onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
            className="w-24 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700 min-w-[40px] text-center">
            {iouThreshold.toFixed(2)}
          </span>
        </div>

        {/* Tiled / Sliced Inference Toggle */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useTiling}
              onChange={(e) => setUseTiling(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
            />
            <span className="text-slate-300 font-semibold flex items-center gap-1">
              <Grid className="h-3.5 w-3.5 text-amber-400" />
              High-Res Tiling (SAHI)
            </span>
          </label>
        </div>
      </div>

      {/* Action Button */}
      <button
        type="button"
        disabled={!canReanalyze || isAnalyzing}
        onClick={onReanalyze}
        className={`px-4 py-1.5 rounded font-mono font-semibold flex items-center gap-2 transition-all ${
          canReanalyze && !isAnalyzing
            ? 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.3)] cursor-pointer'
            : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
        }`}
      >
        <RefreshCw className={`h-3.5 w-3.5 ${isAnalyzing ? 'animate-spin' : ''}`} />
        {isAnalyzing ? 'RUNNING INFERENCE...' : 'RE-ANALYZE'}
      </button>
    </div>
  );
};
