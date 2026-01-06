export interface Model {
  id: string;
  version: string;
  framework: string;
  stage: string;
  is_active: boolean;
  metadata: Record<string, unknown>;
}

export interface PredictionResult {
  model_id: string;
  version: string;
  result: number;
  confidence: number;
  latency_ms: number;
}

export interface DriftReport {
  feature_name: string;
  drift_detected: boolean;
  p_value: number;
  statistic: number;
}
