import React, { useState } from 'react';
import { Table, Search } from 'lucide-react';
import type { DetectionRecord } from '../types';

interface DetectionTableProps {
  detections: DetectionRecord[];
  hoveredDetectionId: number | null;
  selectedDetectionId: number | null;
  onHoverDetection: (id: number | null) => void;
  onSelectDetection: (id: number | null) => void;
}

export const DetectionTable: React.FC<DetectionTableProps> = ({
  detections,
  hoveredDetectionId,
  selectedDetectionId,
  onHoverDetection,
  onSelectDetection,
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = detections.filter((det) => {
    const term = searchTerm.toLowerCase();
    const idMatch = det.id.toString().includes(term);
    const classMatch = det.class_name.toLowerCase().includes(term);
    const latMatch = det.geolocation?.latitude?.toString().includes(term) ?? false;
    const lonMatch = det.geolocation?.longitude?.toString().includes(term) ?? false;
    return idMatch || classMatch || latMatch || lonMatch;
  });

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded flex flex-col font-mono text-xs overflow-hidden">
      {/* Table Header & Search */}
      <div className="border-b border-slate-800 bg-slate-950/60 p-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Table className="h-4 w-4 text-cyan-400" />
          <span className="font-bold uppercase tracking-wider text-slate-200 font-mono-tech">
            Target Detection Inventory ({detections.length})
          </span>
        </div>

        <div className="relative">
          <Search className="h-3.5 w-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Filter ID / coords..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8 pr-2.5 py-1 bg-slate-900 border border-slate-800 rounded text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500 text-xs w-44"
          />
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto max-h-[300px]">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/90 text-slate-400 text-[11px]">
              <th className="py-2 px-3 font-semibold">ID</th>
              <th className="py-2 px-3 font-semibold">CLASS</th>
              <th className="py-2 px-3 font-semibold">CONFIDENCE</th>
              <th className="py-2 px-3 font-semibold">BOUNDING BOX [X1, Y1, X2, Y2]</th>
              <th className="py-2 px-3 font-semibold">CENTER PIXEL</th>
              <th className="py-2 px-3 font-semibold">LATITUDE</th>
              <th className="py-2 px-3 font-semibold">LONGITUDE</th>
              <th className="py-2 px-3 font-semibold">CRS</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filtered.length > 0 ? (
              filtered.map((det) => {
                const isHovered = hoveredDetectionId === det.id;
                const isSelected = selectedDetectionId === det.id;

                return (
                  <tr
                    key={det.id}
                    onMouseEnter={() => onHoverDetection(det.id)}
                    onMouseLeave={() => onHoverDetection(null)}
                    onClick={() => onSelectDetection(det.id)}
                    className={`cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-cyan-950/60 text-slate-100 border-l-2 border-l-amber-400'
                        : isHovered
                        ? 'bg-slate-800/60 text-slate-200'
                        : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <td className="py-2 px-3 font-bold text-cyan-400">
                      #{det.id.toString().padStart(2, '0')}
                    </td>
                    <td className="py-2 px-3 capitalize">
                      {det.class_name.replace('_', ' ')}
                    </td>
                    <td className="py-2 px-3">
                      <span className="px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800 text-[11px] font-bold">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-400">
                      [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}, {det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                    </td>
                    <td className="py-2 px-3 text-slate-400">
                      ({det.center_pixel.x.toFixed(0)}, {det.center_pixel.y.toFixed(0)})
                    </td>
                    <td className="py-2 px-3">
                      {det.geolocation?.latitude !== undefined && det.geolocation?.latitude !== null ? (
                        <span className="text-emerald-400 font-semibold">
                          {det.geolocation.latitude.toFixed(7)}°
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      {det.geolocation?.longitude !== undefined && det.geolocation?.longitude !== null ? (
                        <span className="text-emerald-400 font-semibold">
                          {det.geolocation.longitude.toFixed(7)}°
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-[11px] text-slate-500">
                      {det.geolocation?.crs || 'N/A'}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={8} className="text-center py-6 text-slate-500">
                  {detections.length === 0 ? 'No objects detected above the confidence threshold.' : 'No matching detections found.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
