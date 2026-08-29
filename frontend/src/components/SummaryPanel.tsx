import React from 'react';
import { ShieldCheck, AlertTriangle, Target, Percent, Clock, Tag } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface SummaryPanelProps {
  analysis: AnalysisResponse | null;
}

export const SummaryPanel: React.FC<SummaryPanelProps> = ({ analysis }) => {
  if (!analysis) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded p-4 flex flex-col items-center justify-center min-h-[220px] text-center text-slate-500 font-mono text-xs">
        <Target className="h-8 w-8 text-slate-700 mb-2" />
        <p className="font-semibold text-slate-400">ANALYSIS SUMMARY PENDING</p>
        <p className="text-[11px] text-slate-600 mt-1">Upload or select an image to inspect detection telemetry.</p>
      </div>
    );
  }

  const { summary } = analysis;
  const isDebrisDetected = summary.debris_detected;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded p-4 flex flex-col gap-3 font-mono">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono-tech flex items-center gap-1.5">
          <Target className="h-4 w-4 text-cyan-400" />
          Analysis Result
        </span>
        <span className="text-[10px] text-slate-500">{analysis.timestamp.split('T')[1].slice(0, 8)} UTC</span>
      </div>

      {/* Main Detection Status Banner */}
      <div
        className={`p-3 rounded border flex items-center justify-between ${
          isDebrisDetected
            ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300 shadow-[0_0_20px_rgba(16,185,129,0.15)]'
            : 'bg-slate-950/60 border-slate-800 text-slate-400'
        }`}
      >
        <div className="flex items-center gap-2.5">
          {isDebrisDetected ? (
            <div className="h-8 w-8 rounded bg-emerald-500/20 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
          ) : (
            <div className="h-8 w-8 rounded bg-slate-800 flex items-center justify-center text-slate-400">
              <AlertTriangle className="h-5 w-5" />
            </div>
          )}
          <div>
            <p className="text-[11px] text-slate-400 uppercase tracking-wider">Debris Detected</p>
            <p className="text-base font-bold tracking-wide">
              {isDebrisDetected ? 'YES' : 'NO DEBRIS DETECTED'}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-slate-500 uppercase">Target Count</p>
          <p className="text-xl font-bold text-slate-100">{summary.total_detections}</p>
        </div>
      </div>

      {/* KPI Metrics Grid */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2.5 flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 text-[10px]">
            <span>HIGHEST CONF</span>
            <Percent className="h-3 w-3 text-cyan-400" />
          </div>
          <p className="text-sm font-bold text-cyan-300">
            {summary.highest_confidence !== null && summary.highest_confidence !== undefined
              ? `${(summary.highest_confidence * 100).toFixed(1)}%`
              : '—'}
          </p>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2.5 flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 text-[10px]">
            <span>AVG CONF</span>
            <Percent className="h-3 w-3 text-emerald-400" />
          </div>
          <p className="text-sm font-bold text-emerald-300">
            {summary.average_confidence !== null && summary.average_confidence !== undefined
              ? `${(summary.average_confidence * 100).toFixed(1)}%`
              : '—'}
          </p>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2.5 flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 text-[10px]">
            <span>LATENCY</span>
            <Clock className="h-3 w-3 text-amber-400" />
          </div>
          <p className="text-sm font-bold text-amber-300">
            {summary.inference_time_ms.toFixed(1)} ms
          </p>
        </div>
      </div>

      {/* Detected Classes Breakdown */}
      <div className="bg-slate-950/40 border border-slate-800/80 rounded p-2.5 flex flex-col gap-2">
        <div className="flex items-center justify-between text-[11px] text-slate-400 border-b border-slate-800/60 pb-1">
          <span className="flex items-center gap-1 font-semibold">
            <Tag className="h-3 w-3 text-cyan-400" />
            Detected Classes
          </span>
          <span className="text-slate-500">Count</span>
        </div>

        {Object.keys(summary.class_counts).length > 0 ? (
          <div className="flex flex-col gap-1.5">
            {Object.entries(summary.class_counts).map(([cname, count]) => {
              const pct = summary.total_detections > 0 ? (count / summary.total_detections) * 100 : 0;
              return (
                <div key={cname} className="flex flex-col gap-0.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-200 capitalize">{cname.replace('_', ' ')}</span>
                    <span className="text-cyan-300 font-bold">{count}</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1 overflow-hidden">
                    <div
                      className="bg-cyan-400 h-full rounded-full"
                      style={{ width: `${pct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-[11px] text-slate-500 py-1">
            No debris classes identified above the configured threshold.
          </p>
        )}
      </div>
    </div>
  );
};
