import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mlService } from '../../api/mlService';

// We need to test the real mlService against a mock fetch
const originalFetch = globalThis.fetch;

describe('mlService', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe('getHealth', () => {
    it('returns health data on success', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy', version: '1.0' }),
      });
      const res = await mlService.getHealth();
      expect(res.status).toBe('healthy');
      expect(res.version).toBe('1.0');
    });

    it('throws on HTTP error', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve('Internal Server Error'),
      });
      await expect(mlService.getHealth()).rejects.toThrow('API Error 500');
    });

    it('throws on network error', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
      await expect(mlService.getHealth()).rejects.toThrow('Failed to fetch');
    });
  });

  describe('predict', () => {
    it('sends correct request body', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          prediction_id: 'id-1',
          model_id: 'credit-risk',
          version: 'v1',
          result: 1,
          confidence: { value: 0.9 },
          latency_ms: 0.5,
        }),
      });
      globalThis.fetch = mockFetch;

      await mlService.predict('credit-risk', 'customer-001');

      expect(mockFetch).toHaveBeenCalledWith('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: 'credit-risk', entity_id: 'customer-001' }),
      });
    });

    it('returns prediction response', async () => {
      const expected = {
        prediction_id: 'id-2',
        model_id: 'credit-risk',
        version: 'v1',
        result: 0,
        confidence: { value: 0.65 },
        latency_ms: 1.2,
      };
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(expected),
      });

      const res = await mlService.predict('credit-risk', 'c-1');
      expect(res.result).toBe(0);
      expect(res.confidence.value).toBe(0.65);
    });

    it('throws on 404 model not found', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        text: () => Promise.resolve('Model not found'),
      });
      await expect(mlService.predict('bad-model', 'c-1')).rejects.toThrow('API Error 404');
    });
  });

  describe('getDrift', () => {
    it('calls correct endpoint', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          feature_name: 'f0', drift_detected: false,
          p_value: 0.5, statistic: 0.01, threshold: 0.05,
          method: 'ks', recommendation: 'OK', sample_size: 50,
        }),
      });
      globalThis.fetch = mockFetch;

      await mlService.getDrift('credit-risk');
      expect(mockFetch).toHaveBeenCalledWith('/api/monitoring/drift/credit-risk');
    });

    it('throws on error', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        text: () => Promise.resolve('Not enough data'),
      });
      await expect(mlService.getDrift('x')).rejects.toThrow('API Error 400');
    });
  });

  describe('getModel', () => {
    it('calls correct endpoint', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          id: 'credit-risk', version: 'v1', framework: 'onnx',
          stage: 'champion', is_active: true, metadata: {},
        }),
      });
      globalThis.fetch = mockFetch;

      await mlService.getModel('credit-risk');
      expect(mockFetch).toHaveBeenCalledWith('/api/models/credit-risk');
    });
  });
});
