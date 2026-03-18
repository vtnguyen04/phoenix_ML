const API_BASE = '/api';

export interface HealthResponse {
  status: string;
  version: string;
}

export interface PredictionResponse {
  prediction_id: string;
  model_id: string;
  version: string;
  result: number;
  confidence: { value: number };
  latency_ms: number;
}

export interface DriftReport {
  feature_name: string;
  drift_detected: boolean;
  p_value: number;
  statistic: number;
  threshold: number;
  method: string;
  recommendation: string;
  sample_size: number;
}

export interface ModelInfo {
  id: string;
  version: string;
  framework: string;
  stage: string;
  is_active: boolean;
  metadata: Record<string, unknown>;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error ${response.status}: ${errorText}`);
  }
  return response.json();
}

export const mlService = {
  getHealth: async (): Promise<HealthResponse> => {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<HealthResponse>(res);
  },

  predict: async (modelId: string, entityId: string): Promise<PredictionResponse> => {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, entity_id: entityId }),
    });
    return handleResponse<PredictionResponse>(res);
  },

  getDrift: async (modelId: string): Promise<DriftReport> => {
    const res = await fetch(`${API_BASE}/monitoring/drift/${modelId}`);
    return handleResponse<DriftReport>(res);
  },

  getModel: async (modelId: string): Promise<ModelInfo> => {
    const res = await fetch(`${API_BASE}/models/${modelId}`);
    return handleResponse<ModelInfo>(res);
  },
};
