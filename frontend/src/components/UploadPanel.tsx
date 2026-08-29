import React, { useRef, useState } from 'react';
import { Upload, FileImage, Database, CheckCircle2, ChevronRight, Globe } from 'lucide-react';
import type { SampleItem } from '../types';

interface UploadPanelProps {
  onFileUpload: (file: File) => void;
  onSampleSelect: (sampleId: string) => void;
  samples: SampleItem[];
  isAnalyzing: boolean;
  selectedSampleId: string | null;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({
  onFileUpload,
  onSampleSelect,
  samples,
  isAnalyzing,
  selectedSampleId,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded p-4 flex flex-col gap-4">
      {/* Section Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <Upload className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono-tech">
            Imagery Ingestion
          </span>
        </div>
        <span className="text-[11px] text-slate-500 font-mono">TIFF • JPG • PNG</span>
      </div>

      {/* Drag & Drop Upload Target */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isAnalyzing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded p-6 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-cyan-400 bg-cyan-950/20 shadow-[0_0_20px_rgba(6,182,212,0.15)]'
            : 'border-slate-700/80 bg-slate-950/40 hover:border-cyan-500/60 hover:bg-slate-950/70'
        } ${isAnalyzing ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".tif,.tiff,.jpg,.jpeg,.png"
          className="hidden"
        />
        <div className="flex flex-col items-center gap-2">
          <div className="h-10 w-10 rounded border border-slate-700 bg-slate-900 flex items-center justify-center text-slate-400 group-hover:text-cyan-400">
            <FileImage className="h-5 w-5 text-cyan-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-200">
              Drop aerial / sonar imagery here
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">
              or <span className="text-cyan-400 underline underline-offset-2">browse filesystem</span>
            </p>
          </div>
          <div className="flex items-center gap-1.5 mt-1">
            <span className="px-1.5 py-0.5 rounded bg-slate-800/80 text-[10px] text-slate-400 font-mono border border-slate-700">GeoTIFF</span>
            <span className="px-1.5 py-0.5 rounded bg-slate-800/80 text-[10px] text-slate-400 font-mono border border-slate-700">TIFF</span>
            <span className="px-1.5 py-0.5 rounded bg-slate-800/80 text-[10px] text-slate-400 font-mono border border-slate-700">JPG</span>
            <span className="px-1.5 py-0.5 rounded bg-slate-800/80 text-[10px] text-slate-400 font-mono border border-slate-700">PNG</span>
          </div>
        </div>
      </div>

      {/* Preloaded Dataset Mission Scenarios */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-slate-400 flex items-center gap-1.5 font-semibold">
            <Database className="h-3.5 w-3.5 text-cyan-400" />
            Curated Mission Datasets
          </span>
          <span className="text-[10px] text-slate-500">1-Click Test</span>
        </div>

        <div className="grid grid-cols-1 gap-2">
          {samples.map((sample) => {
            const isSelected = selectedSampleId === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                disabled={isAnalyzing}
                onClick={() => onSampleSelect(sample.id)}
                className={`w-full text-left p-2.5 rounded border transition-all flex items-start justify-between gap-2 ${
                  isSelected
                    ? 'border-cyan-500 bg-cyan-950/40 text-slate-100 shadow-[0_0_15px_rgba(6,182,212,0.1)]'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-900/60 text-slate-300'
                } ${isAnalyzing ? 'opacity-60 cursor-not-allowed' : ''}`}
              >
                <div className="flex flex-col gap-0.5 pr-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-200">
                      {sample.name}
                    </span>
                    {sample.has_geolocation ? (
                      <span className="inline-flex items-center gap-0.5 px-1 py-0.2 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800 text-[9px] font-mono">
                        <Globe className="h-2.5 w-2.5" /> GPS
                      </span>
                    ) : (
                      <span className="px-1 py-0.2 rounded bg-slate-800 text-slate-400 text-[9px] font-mono">
                        NON-GEO
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-1">
                    {sample.description}
                  </p>
                </div>
                <div className="pt-0.5 text-slate-500">
                  {isSelected ? (
                    <CheckCircle2 className="h-4 w-4 text-cyan-400 shrink-0" />
                  ) : (
                    <ChevronRight className="h-4 w-4 shrink-0 text-slate-600" />
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
