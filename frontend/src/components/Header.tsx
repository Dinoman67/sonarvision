import React from 'react';
import { Radio, Cpu, Layers, Activity } from 'lucide-react';
import type { ModelMetadata } from '../types';

interface HeaderProps {
  modelInfo: ModelMetadata | null;
  isAnalyzing: boolean;
}

export const Header: React.FC<HeaderProps> = ({ modelInfo, isAnalyzing }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/90 backdrop-blur px-6 py-3 sticky top-0 z-50">
      <div className="max-w-[1800px] mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand & Mission Title */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded border border-cyan-500/40 bg-cyan-950/40 flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            <Radio className={`h-5 w-5 ${isAnalyzing ? 'animate-pulse text-cyan-300' : ''}`} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-wider text-slate-100 font-mono-tech">
                YOLO-ESI <span className="text-cyan-400">//</span> DEBRIS INTELLIGENCE
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono font-semibold">
                v1.0-PROD
              </span>
            </div>
            <p className="text-xs text-slate-400 flex items-center gap-2">
              <span>Environmental Remote Sensing & Side-Scan Sonar Analysis</span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-500 font-mono text-[11px]">SIH 2026</span>
            </p>
          </div>
        </div>

        {/* Status Badges */}
        <div className="flex items-center flex-wrap gap-2 text-xs font-mono">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            <Activity className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-slate-500">ENGINE:</span>
            <span className="text-emerald-400 font-semibold">ONLINE</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            <Cpu className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-500">MODEL:</span>
            <span className="text-cyan-300 font-semibold">{modelInfo ? `${modelInfo.model_name} (${modelInfo.format})` : 'YOLOv8-ESI (ONNX)'}</span>
          </div>

          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            <Layers className="h-3.5 w-3.5 text-amber-400" />
            <span className="text-slate-500">ATTENTION:</span>
            <span className="text-amber-300">SE-Augmented</span>
          </div>

          {modelInfo?.execution_provider && (
            <div className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-400">
              <span className="text-slate-500">PROVIDER:</span>
              <span className="text-slate-300">{modelInfo.execution_provider.replace('ExecutionProvider', '')}</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
