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
        },
      },
    }),
    getModels: vi.fn().mockResolvedValue([
      { model_id: 'credit-risk', version: 'v1', status: 'champion', metadata: {} },
      { model_id: 'fraud-detection', version: 'v1', status: 'champion', metadata: {} },
    ]),
    getDriftReports: vi.fn().mockResolvedValue([]),
    predict: vi.fn(),
    getDrift: vi.fn(),
    getPerformance: vi.fn(),
  },
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the dashboard header', async () => {
    render(<App />);
    expect(screen.getByText('Production Dashboard')).toBeInTheDocument();
  });

  it('renders sidebar brand', () => {
    render(<App />);
    expect(screen.getByText('PHOENIX ML')).toBeInTheDocument();
  });

  it('renders all navigation links', () => {
    render(<App />);
    expect(screen.getByText('📊 Dashboard')).toBeInTheDocument();
    expect(screen.getByText('📈 Grafana')).toBeInTheDocument();
    expect(screen.getByText('🔍 Jaeger')).toBeInTheDocument();
  });

  it('renders stats grid with real metrics', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('78.5%')).toBeInTheDocument();
    });
    expect(screen.getByText('85.4%')).toBeInTheDocument();
    expect(screen.getByText('81.3%')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();
  });

  it('renders model info card', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('🤖 Champion Model')).toBeInTheDocument();
    });
  });

  it('renders pipeline status', () => {
    render(<App />);
    expect(screen.getByText('⚙️ MLOps Pipeline')).toBeInTheDocument();
  });

  it('renders Grafana embed section', () => {
    render(<App />);
    expect(screen.getByText(/Live Metrics/)).toBeInTheDocument();
  });

  it('renders drift monitor', () => {
    render(<App />);
    expect(screen.getByText('🛡️ Drift Monitor')).toBeInTheDocument();
  });

  it('renders inference test panel', () => {
    render(<App />);
    expect(screen.getByText('🎯 Real-time Inference')).toBeInTheDocument();
  });

  it('renders infrastructure services', () => {
    render(<App />);
    expect(screen.getByText('🏗️ Infrastructure')).toBeInTheDocument();
  });

  it('shows auto-refresh badge', () => {
    render(<App />);
    expect(screen.getByText('Auto-refresh: 15s')).toBeInTheDocument();
  });
});
