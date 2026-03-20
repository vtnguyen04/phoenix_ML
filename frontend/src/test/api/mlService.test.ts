import { describe, it, expect, beforeEach, vi } from 'vitest';
import { mlService } from '../../api/mlService';

const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('mlService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getHealth', () => {
    it('fetches from /api/health', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy', version: '0.1.0' }),
      });
      const result = await mlService.getHealth();
      expect(mockFetch).toHaveBeenCalledWith('/api/health');
      expect(result.status).toBe('healthy');
    });
  });

  describe('predict', () => {
    it('posts to /api/predict', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ prediction_id: '123', result: 1 }),
      });
      await mlService.predict('credit-risk', 'customer-1');
      expect(mockFetch).toHaveBeenCalledWith('/api/predict', expect.objectContaining({
        method: 'POST',
      }));
    });
  });

  describe('getModel', () => {
    it('fetches from /api/models/{modelId}', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ model_id: 'credit-risk', version: 'v1' }),
      });
      const result = await mlService.getModel('credit-risk');
      expect(mockFetch).toHaveBeenCalledWith('/api/models/credit-risk');
      expect(result.model_id).toBe('credit-risk');
    });
  });

  describe('getDrift', () => {
    it('fetches from /api/monitoring/drift/{modelId}', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ drift_detected: false }),
      });
      await mlService.getDrift('credit-risk');
      expect(mockFetch).toHaveBeenCalledWith('/api/monitoring/drift/credit-risk');
    });

    it('throws on 400 error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: () => Promise.resolve('Not enough data'),
      });
      await expect(mlService.getDrift('x')).rejects.toThrow('API Error 400');
    });
  });

  describe('getDriftReports', () => {
    it('fetches from /api/monitoring/reports/{modelId}', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      });
      const result = await mlService.getDriftReports('credit-risk');
      expect(mockFetch).toHaveBeenCalledWith('/api/monitoring/reports/credit-risk?limit=5');
      expect(result).toEqual([]);
    });
  });

  describe('getPerformance', () => {
    it('fetches from /api/monitoring/performance/{modelId}', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ total_predictions: 100 }),
      });
      const result = await mlService.getPerformance('credit-risk');
      expect(mockFetch).toHaveBeenCalledWith('/api/monitoring/performance/credit-risk');
      expect(result.total_predictions).toBe(100);
    });
  });

  describe('getModels', () => {
    it('fetches from /api/models', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([
          { model_id: 'credit-risk', version: 'v1' },
          { model_id: 'fraud-detection', version: 'v1' },
        ]),
      });
      const result = await mlService.getModels();
      expect(mockFetch).toHaveBeenCalledWith('/api/models');
      expect(result).toHaveLength(2);
      expect(result[0].model_id).toBe('credit-risk');
    });
  });
});
