import axios from 'axios';
import { PredictionResult, DriftReport } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export const mlService = {
  getHealth: async () => {
    const { data } = await apiClient.get('/health');
    return data;
  },

  predict: async (modelId: string, entityId: string): Promise<PredictionResult> => {
    const { data } = await apiClient.post('/predict', {
      model_id: modelId,
      entity_id: entityId,
    });
    return data;
  },

  getDrift: async (modelId: string): Promise<DriftReport> => {
    const { data } = await apiClient.get(`/monitoring/drift/${modelId}`);
    return data;
  },
};
