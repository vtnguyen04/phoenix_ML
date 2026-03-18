import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';

// Mock the entire mlService module
vi.mock('../api/mlService', () => ({
  mlService: {
    getHealth: vi.fn().mockResolvedValue({ status: 'healthy', version: '0.1.0' }),
    predict: vi.fn().mockResolvedValue({
      prediction_id: 'test-123',
      model_id: 'credit-risk',
      version: 'v1',
      result: 1,
      confidence: { value: 0.85 },
      latency_ms: 0.42,
    }),
    getDrift: vi.fn().mockResolvedValue({
      feature_name: 'credit_amount',
      drift_detected: false,
      p_value: 0.45,
      statistic: 0.05,
      threshold: 0.1,
      method: 'ks',
      recommendation: 'System is healthy.',
      sample_size: 100,
    }),
    getModel: vi.fn().mockResolvedValue({
      id: 'credit-risk',
      version: 'v1',
      framework: 'onnx',
      stage: 'champion',
      is_active: true,
      metadata: {},
    }),
  },
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the dashboard header', () => {
    render(<App />);
    expect(screen.getByText('Production Dashboard')).toBeInTheDocument();
  });

  it('renders sidebar brand', () => {
    render(<App />);
    expect(screen.getByText('PHOENIX ML')).toBeInTheDocument();
  });

  it('renders all navigation items', () => {
    render(<App />);
    // Use emoji prefix to match nav items specifically (not header)
    expect(screen.getByText('📊 Dashboard')).toBeInTheDocument();
    expect(screen.getByText('🎯 Inference')).toBeInTheDocument();
    expect(screen.getByText('🛡️ Monitoring')).toBeInTheDocument();
    expect(screen.getByText('📈 Grafana')).toBeInTheDocument();
    expect(screen.getByText('🔥 Prometheus')).toBeInTheDocument();
    expect(screen.getByText('📦 MinIO')).toBeInTheDocument();
  });

  it('renders stats grid with correct labels', () => {
    render(<App />);
    expect(screen.getByText('Total Predictions')).toBeInTheDocument();
    expect(screen.getByText('Last Latency')).toBeInTheDocument();
    expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Active Models')).toBeInTheDocument();
  });

  it('displays model accuracy value', () => {
    render(<App />);
    expect(screen.getByText('78.5%')).toBeInTheDocument();
  });

  it('shows Predict Credit Risk button', () => {
    render(<App />);
    expect(screen.getByText(/Predict Credit Risk/)).toBeInTheDocument();
  });

  it('shows Scan Drift button', () => {
    render(<App />);
    expect(screen.getByText(/Scan Drift/)).toBeInTheDocument();
  });

  it('renders infrastructure services section', () => {
    render(<App />);
    expect(screen.getByText('API Server')).toBeInTheDocument();
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument();
    expect(screen.getByText('Redis')).toBeInTheDocument();
    expect(screen.getByText('Kafka')).toBeInTheDocument();
  });

  it('shows system online status after health check', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('System Online')).toBeInTheDocument();
    });
  });

  it('handles predict click and shows result', async () => {
    render(<App />);
    const predictBtn = screen.getByText(/Predict Credit Risk/);
    fireEvent.click(predictBtn);

    await waitFor(() => {
      expect(screen.getByText(/GOOD CREDIT/)).toBeInTheDocument();
    });
  });
});
