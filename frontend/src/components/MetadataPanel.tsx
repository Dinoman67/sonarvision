import React, { useState } from 'react';
import { Cpu, FileText, Globe, Hash, CheckCircle2, XCircle } from 'lucide-react';
import type { FileMetadata, GeospatialMetadata, ModelMetadata } from '../types';

interface MetadataPanelProps {
  fileMeta: FileMetadata | null;
  geoMeta: GeospatialMetadata | null;
  modelMeta: ModelMetadata | null;
}

type TabType = 'file' | 'geospatial' | 'model';

export const MetadataPanel: React.FC<MetadataPanelProps> = ({
  fileMeta,
  geoMeta,
  modelMeta,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('geospatial');

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded flex flex-col font-mono text-xs overflow-hidden">
      {/* Tab Navigation */}
      <div className="border-b border-slate-800 bg-slate-950/60 p-2 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setActiveTab('geospatial')}
          className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
            activeTab === 'geospatial'
              ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Globe className="h-3 w-3" />
          Geospatial & CRS
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('model')}
          className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
            activeTab === 'model'
              ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Cpu className="h-3 w-3" />
          Model ONNX Specs
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('file')}
          className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
            activeTab === 'file'
              ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_10px_rgba(6,182,212,0.3)]'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileText className="h-3 w-3" />
          File Properties
        </button>
      </div>

      {/* Tab Content */}
      <div className="p-3 bg-slate-950/40 min-h-[160px]">
        {/* 1. Geospatial Tab */}
        {activeTab === 'geospatial' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">GEOREFERENCED:</span>
                <span className={`font-bold flex items-center gap-1 ${
                  geoMeta?.camera_latitude !== undefined && geoMeta?.camera_latitude !== null
                    ? 'text-cyan-400'
                    : geoMeta?.georeferenced
                    ? 'text-emerald-400'
                    : 'text-slate-400'
                }`}>
                  {geoMeta?.georeferenced ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                  {geoMeta?.camera_latitude !== undefined && geoMeta?.camera_latitude !== null
                    ? 'CAMERA GPS'
                    : geoMeta?.georeferenced
                    ? 'YES'
                    : 'NO'}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">CRS:</span>
                <span className="text-cyan-300 font-semibold">{geoMeta?.crs || 'Unavailable'}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">COORDINATE SOURCE:</span>
                <span className="text-slate-300">{geoMeta?.coordinate_source || 'None'}</span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              {geoMeta?.camera_latitude !== undefined && geoMeta?.camera_latitude !== null ? (
                <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                  <span className="text-slate-500">CAMERA GPS:</span>
                  <span className="text-cyan-300 font-semibold">
                    {geoMeta.camera_latitude.toFixed(6)}°, {geoMeta.camera_longitude?.toFixed(6)}°
                  </span>
                </div>
              ) : (
                <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                  <span className="text-slate-500">PIXEL RESOLUTION:</span>
                  <span className="text-slate-300">
                    {geoMeta?.pixel_resolution ? `${geoMeta.pixel_resolution[0]} × ${geoMeta.pixel_resolution[1]} m/px` : 'N/A'}
                  </span>
                </div>
              )}
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">LAT/LON RECEPTIVE:</span>
                <span className={geoMeta?.lat_lon_available ? 'text-emerald-400' : 'text-slate-500'}>
                  {geoMeta?.lat_lon_available ? 'ENABLED' : 'DISABLED'}
                </span>
              </div>
              <div className="flex flex-col gap-0.5 text-[11px]">
                <span className="text-slate-500">STATUS:</span>
                <span className="text-slate-400 italic">{geoMeta?.status_message || 'Awaiting image analysis'}</span>
              </div>
            </div>
          </div>
        )}

        {/* 2. Model Specs Tab */}
        {activeTab === 'model' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">MODEL NAME:</span>
                <span className="text-cyan-300 font-bold">{modelMeta?.model_name || 'YOLOv8-ESI'}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">FORMAT & INPUT:</span>
                <span className="text-slate-300">{modelMeta?.format || 'ONNX'} ({modelMeta?.input_resolution || '256x256'})</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">ATTENTION MODULE:</span>
                <span className="text-amber-300">{modelMeta?.attention_mechanism || 'SE Attention'}</span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">EXECUTION PROVIDER:</span>
                <span className="text-emerald-400 font-semibold">{modelMeta?.execution_provider || 'CPUExecutionProvider'}</span>
              </div>
              <div className="flex flex-col gap-1 border-b border-slate-800/60 pb-1">
                <span className="text-slate-500 flex items-center gap-1">
                  <Hash className="h-3 w-3 text-cyan-400" />
                  SHA-256 (IMMUTABLE):
                </span>
                <span className="text-[10px] text-slate-400 break-all font-mono">
                  {modelMeta?.sha256_hash || 'Calculating...'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* 3. File Properties Tab */}
        {activeTab === 'file' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">FILENAME:</span>
                <span className="text-slate-200 font-semibold truncate max-w-[200px]">
                  {fileMeta?.filename || 'No file selected'}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">FORMAT:</span>
                <span className="text-cyan-300 font-bold">{fileMeta?.format || 'N/A'}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">FILE SIZE:</span>
                <span className="text-slate-300">{fileMeta?.file_size_human || '0 B'}</span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">WIDTH:</span>
                <span className="text-slate-300">{fileMeta?.width ? `${fileMeta.width} px` : '—'}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">HEIGHT:</span>
                <span className="text-slate-300">{fileMeta?.height ? `${fileMeta.height} px` : '—'}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1">
                <span className="text-slate-500">CHANNELS:</span>
                <span className="text-slate-300">{fileMeta?.channels ? `${fileMeta.channels} Bands` : '—'}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
