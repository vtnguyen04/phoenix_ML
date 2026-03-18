import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ModelInfoCard } from '../../components/dashboard/ModelInfoCard';
import type { ModelInfo } from '../../api/mlService';

const mockModel: ModelInfo = {
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
};

describe('ModelInfoCard', () => {
  it('shows loading state when model is null', () => {
    render(<ModelInfoCard model={null} />);
    expect(screen.getByText('Loading model information...')).toBeInTheDocument();
  });

  it('displays model ID', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('credit-risk')).toBeInTheDocument();
  });

  it('displays model type', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('GradientBoosting')).toBeInTheDocument();
  });

  it('displays dataset', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('german-credit')).toBeInTheDocument();
  });

  it('displays feature count', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('3 features')).toBeInTheDocument();
  });

  it('displays training samples', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('800 train · 200 test')).toBeInTheDocument();
  });

  it('shows CHAMPION badge', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('CHAMPION')).toBeInTheDocument();
  });

  it('displays CV score', () => {
    render(<ModelInfoCard model={mockModel} />);
    expect(screen.getByText('75.5% ± 1.9%')).toBeInTheDocument();
  });
});
