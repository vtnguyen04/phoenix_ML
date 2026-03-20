import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import App from '../App';

vi.mock('../api/mlService', () => ({
  mlService: {
    getHealth: vi.fn().mockResolvedValue({ status: 'healthy', version: '0.1.0' }),
    getModel: vi.fn().mockResolvedValue({
      model_id: 'credit-risk',
      version: 'v1',
      status: 'champion',
      metadata: {
        features: ['f1', 'f2', 'f3'],
        role: 'champion',
        dataset: 'german-credit',
        metrics: {
          accuracy: 0.785,
          f1_score: 0.854,
          precision: 0.813,
          recall: 0.9,
          model_type: 'GradientBoosting',
          n_features: 30,
          train_samples: 800,
          test_samples: 200,
          cv_accuracy_mean: 0.755,
          cv_accuracy_std: 0.019,
          all_features: ['duration', 'credit_amount', 'age'],
        },
      },
    }),
    getModels: vi.fn().mockResolvedValue([
      { model_id: 'credit-risk', version: 'v1', status: 'champion', metadata: { role: 'champion' } },
      { model_id: 'fraud-detection', version: 'v1', status: 'champion', metadata: { role: 'champion' } },
    ]),
    getDriftReports: vi.fn().mockResolvedValue([]),
    predict: vi.fn(),
    predictWithFeatures: vi.fn(),
    getDrift: vi.fn(),
    getPerformance: vi.fn().mockResolvedValue({
      model_id: 'credit-risk',
      version: 'v1',
      total_predictions: 100,
      metrics: { avg_latency_ms: 2.5, avg_confidence: 0.85 },
    }),
  },
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the dashboard header', async () => {
    render(<App />);
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('renders sidebar brand', () => {
    render(<App />);
    expect(screen.getByText('PHOENIX ML')).toBeInTheDocument();
  });

  it('renders navigation links', () => {
    render(<App />);
    const grafana = screen.getAllByText(/Grafana/);
    expect(grafana.length).toBeGreaterThanOrEqual(1);
  });

  it('renders stats grid with real metrics', async () => {
    render(<App />);
    await waitFor(() => {
      const cells = screen.getAllByText('78.5%');
      expect(cells.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders model info card', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('Champion Model')).toBeInTheDocument();
    });
  });

  it('renders champion vs challenger', () => {
    render(<App />);
    expect(screen.getByText('Champion vs Challenger')).toBeInTheDocument();
  });

  it('renders Grafana embed section', () => {
    render(<App />);
    expect(screen.getByText(/Live Metrics/)).toBeInTheDocument();
  });

  it('renders drift monitor', () => {
    render(<App />);
    expect(screen.getByText('Drift Monitor')).toBeInTheDocument();
  });

  it('renders inference panel with feature inputs', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('Inference')).toBeInTheDocument();
    });
  });

  it('renders infrastructure services', () => {
    render(<App />);
    expect(screen.getByText('Infrastructure')).toBeInTheDocument();
  });

  it('renders performance panel', () => {
    render(<App />);
    expect(screen.getByText('Live Performance')).toBeInTheDocument();
  });
});
