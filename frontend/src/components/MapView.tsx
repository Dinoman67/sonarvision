import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Globe, AlertCircle, Compass } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface MapViewProps {
  analysis: AnalysisResponse | null;
  selectedDetectionId: number | null;
  onSelectDetection: (id: number | null) => void;
}

export const MapView: React.FC<MapViewProps> = ({
  analysis,
  selectedDetectionId,
  onSelectDetection,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const footprintLayerRef = useRef<L.GeoJSON | null>(null);

  const hasCameraGps = Boolean(
    analysis?.geospatial_metadata?.camera_latitude !== undefined &&
    analysis?.geospatial_metadata?.camera_latitude !== null &&
    analysis?.geospatial_metadata?.camera_longitude !== undefined &&
    analysis?.geospatial_metadata?.camera_longitude !== null
  );

  const isGeoreferenced = Boolean(analysis?.geospatial_metadata?.georeferenced && analysis?.geospatial_metadata?.lat_lon_available);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [28.95, -89.35],
        zoom: 12,
        zoomControl: true,
        attributionControl: true,
      });

      // Dark Matter tile layer
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);

      markersLayerRef.current = L.layerGroup().addTo(map);
      mapInstanceRef.current = map;
    }

    return () => {
      // Cleanup if needed
    };
  }, []);

  // Update markers & footprint when analysis changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Clear previous layers
    if (markersLayerRef.current) {
      markersLayerRef.current.clearLayers();
    }
    if (footprintLayerRef.current) {
      map.removeLayer(footprintLayerRef.current);
      footprintLayerRef.current = null;
    }

    if (!analysis || !isGeoreferenced) return;

    const detections = analysis.detections.filter((d) => d.geolocation && d.geolocation.latitude && d.geolocation.longitude);
    const bounds = L.latLngBounds([]);

    // 1. Add Footprint GeoJSON if present (for raster bounds)
    if (analysis.geospatial_metadata.footprint_geojson && analysis.geospatial_metadata.footprint_geojson.type === 'Polygon') {
      try {
        const footprint = L.geoJSON(analysis.geospatial_metadata.footprint_geojson, {
          style: {
            color: '#06b6d4',
            weight: 2,
            opacity: 0.8,
            fillColor: '#06b6d4',
            fillOpacity: 0.1,
            dashArray: '4, 4',
          },
        }).addTo(map);
        footprintLayerRef.current = footprint;
        bounds.extend(footprint.getBounds());
      } catch (err) {
        console.error('Footprint render error:', err);
      }
    }

    // 2. Case A: Object-level detections available
    if (detections.length > 0) {
      detections.forEach((det) => {
        if (!det.geolocation) return;
        const { latitude, longitude, crs, coordinate_source, utm_x, utm_y } = det.geolocation;
        const latLng: [number, number] = [latitude, longitude];
        bounds.extend(latLng);

        const isSelected = selectedDetectionId === det.id;

        // Custom SVG Pin Icon
        const customIcon = L.divIcon({
          className: 'custom-map-marker',
          html: `
            <div style="
              width: 28px;
              height: 28px;
              background: ${isSelected ? '#f59e0b' : '#0284c7'};
              border: 2px solid ${isSelected ? '#fef08a' : '#38bdf8'};
              border-radius: 50%;
              display: flex;
              align-items: center;
              justify-content: center;
              color: #0f172a;
              font-weight: 800;
              font-size: 11px;
              font-family: monospace;
              box-shadow: 0 0 15px ${isSelected ? 'rgba(245, 158, 11, 0.8)' : 'rgba(6, 182, 212, 0.6)'};
              transform: translate(-50%, -50%);
            ">
              ${det.id}
            </div>
          `,
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        });

        const marker = L.marker(latLng, { icon: customIcon });

        const popupContent = `
          <div style="font-family: monospace; font-size: 12px; line-height: 1.4; min-width: 200px;">
            <div style="border-bottom: 1px solid #334155; padding-bottom: 4px; margin-bottom: 6px; display: flex; justify-content: space-between;">
              <strong style="color: #38bdf8;">TARGET #${det.id.toString().padStart(2, '0')}</strong>
              <span style="color: #10b981; font-weight: bold;">${(det.confidence * 100).toFixed(1)}%</span>
            </div>
            <div style="margin-bottom: 3px;">
              <span style="color: #94a3b8;">Class:</span> <span style="color: #f1f5f9;">${det.class_name}</span>
            </div>
            <div style="margin-bottom: 3px;">
              <span style="color: #94a3b8;">Latitude:</span> <span style="color: #f1f5f9; font-weight: bold;">${latitude.toFixed(7)}°</span>
            </div>
            <div style="margin-bottom: 3px;">
              <span style="color: #94a3b8;">Longitude:</span> <span style="color: #f1f5f9; font-weight: bold;">${longitude.toFixed(7)}°</span>
            </div>
            ${utm_x ? `
            <div style="margin-bottom: 3px;">
              <span style="color: #94a3b8;">UTM (E, N):</span> <span style="color: #cbd5e1;">${utm_x.toFixed(1)}, ${utm_y?.toFixed(1)}</span>
            </div>` : ''}
            <div style="margin-bottom: 3px;">
              <span style="color: #94a3b8;">CRS:</span> <span style="color: #38bdf8;">${crs}</span>
            </div>
            <div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #334155; font-size: 10px; color: #64748b;">
              Source: ${coordinate_source}
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);
        marker.on('click', () => {
          onSelectDetection(det.id);
        });

        if (markersLayerRef.current) {
          markersLayerRef.current.addLayer(marker);
        }
      });
    } else if (hasCameraGps) {
      // 2. Case B: Camera GPS / Image location only
      const camLat = analysis.geospatial_metadata.camera_latitude!;
      const camLon = analysis.geospatial_metadata.camera_longitude!;
      const camLatLng: [number, number] = [camLat, camLon];
      bounds.extend(camLatLng);

      const cameraIcon = L.divIcon({
        className: 'camera-map-marker',
        html: `
          <div style="
            width: 32px;
            height: 32px;
            background: #0d9488;
            border: 2px solid #5eead4;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 800;
            font-size: 13px;
            box-shadow: 0 0 15px rgba(20, 184, 166, 0.7);
            transform: translate(-50%, -50%);
          ">
            📷
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const camMarker = L.marker(camLatLng, { icon: cameraIcon });
      const camPopup = `
        <div style="font-family: monospace; font-size: 12px; line-height: 1.4; min-width: 190px;">
          <div style="border-bottom: 1px solid #334155; padding-bottom: 4px; margin-bottom: 6px;">
            <strong style="color: #2dd4bf;">CAMERA / SENSOR LOCATION</strong>
          </div>
          <div style="margin-bottom: 3px;">
            <span style="color: #94a3b8;">Latitude:</span> <span style="color: #f1f5f9; font-weight: bold;">${camLat.toFixed(7)}°</span>
          </div>
          <div style="margin-bottom: 3px;">
            <span style="color: #94a3b8;">Longitude:</span> <span style="color: #f1f5f9; font-weight: bold;">${camLon.toFixed(7)}°</span>
          </div>
          ${analysis.geospatial_metadata.camera_altitude ? `
          <div style="margin-bottom: 3px;">
            <span style="color: #94a3b8;">Altitude:</span> <span style="color: #f1f5f9;">${analysis.geospatial_metadata.camera_altitude} m</span>
          </div>` : ''}
          <div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #334155; font-size: 10px; color: #64748b;">
            EXIF GPS (Camera location only)
          </div>
        </div>
      `;
      camMarker.bindPopup(camPopup);
      if (markersLayerRef.current) {
        markersLayerRef.current.addLayer(camMarker);
      }
    }

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 17 });
    }
  }, [analysis, selectedDetectionId, isGeoreferenced, hasCameraGps, onSelectDetection]);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded flex flex-col h-full overflow-hidden">
      {/* Map Header */}
      <div className="border-b border-slate-800 bg-slate-950/60 p-2.5 flex items-center justify-between font-mono text-xs">
        <div className="flex items-center gap-2">
          <Globe className="h-4 w-4 text-cyan-400" />
          <span className="font-bold uppercase tracking-wider text-slate-200 font-mono-tech">
            Geospatial Cartography & Positioning
          </span>
        </div>

        <div className="flex items-center gap-2">
          {isGeoreferenced ? (
            <span className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800">
              <Compass className="h-3 w-3" />
              {hasCameraGps && !analysis?.detections.some((d) => d.geolocation)
                ? 'Camera GPS / image location only'
                : analysis?.geospatial_metadata.crs || 'WGS84'}
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">
              NOT GEOREFERENCED
            </span>
          )}
        </div>
      </div>

      {/* Map Container */}
      <div className="relative flex-1 min-h-[360px] bg-slate-950">
        <div ref={mapContainerRef} className="w-full h-full min-h-[360px]" />

        {/* Unavailable overlay message if non-georeferenced */}
        {analysis && !isGeoreferenced && (
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-sm z-[1000] flex flex-col items-center justify-center p-6 text-center font-mono">
            <div className="h-12 w-12 rounded border border-slate-700 bg-slate-900 flex items-center justify-center text-slate-400 mb-3">
              <AlertCircle className="h-6 w-6 text-amber-400" />
            </div>
            <p className="text-sm font-bold text-slate-200">
              GEOSPATIAL STATUS: NOT GEOREFERENCED
            </p>
            <p className="text-xs text-slate-400 mt-1 max-w-md">
              Reason: No valid GeoTIFF transform, CRS, EXIF GPS, or world-file information was found.
            </p>
            <div className="mt-3 px-3 py-1.5 rounded bg-slate-900 border border-slate-800 text-[11px] text-slate-400">
              Pixel coordinates are recorded accurately in the Target Inventory table.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
