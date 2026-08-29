import axios from 'axios';
import type { AnalysisResponse, ModelMetadata, SampleItem } from '../types';

const API_BASE = '/api';

export async function fetchModelInfo(): Promise<ModelMetadata> {
  const res = await axios.get(`${API_BASE}/model-info`);
  return res.data;
}

export async function fetchSamples(): Promise<SampleItem[]> {
  const res = await axios.get(`${API_BASE}/samples`);
  return res.data;
}

export async function analyzeImage(
  file: File,
  confThreshold: number = 0.25,
  iouThreshold: number = 0.45,
  useTiling?: boolean
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('confidence_threshold', confThreshold.toString());
  formData.append('iou_threshold', iouThreshold.toString());
  if (useTiling !== undefined) {
    formData.append('use_tiling', useTiling.toString());
  }

  const res = await axios.post(`${API_BASE}/analyze`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

export async function analyzeSample(
  sampleId: string,
  confThreshold: number = 0.25,
  iouThreshold: number = 0.45,
  useTiling?: boolean
): Promise<AnalysisResponse> {
  const res = await axios.post(`${API_BASE}/analyze-sample`, {
    sample_id: sampleId,
    confidence_threshold: confThreshold,
    iou_threshold: iouThreshold,
    use_tiling: useTiling,
  });
  return res.data;
}
