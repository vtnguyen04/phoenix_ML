import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PredictionResult } from '../../components/ui/PredictionResult';
import type { PredictionResponse } from '../../api/mlService';

const goodPrediction: PredictionResponse = {
  prediction_id: 'test-1',
  model_id: 'credit-risk',
  version: 'v1',
  result: 1,
  confidence: { value: 0.92 },
  latency_ms: 0.5,
};

const badPrediction: PredictionResponse = {
  prediction_id: 'test-2',
  model_id: 'credit-risk',
  version: 'v2',
  result: 0,
  confidence: { value: 0.78 },
  latency_ms: 1.23,
};

describe('PredictionResult', () => {
  it('renders model version', () => {
    render(<PredictionResult prediction={goodPrediction} />);
    expect(screen.getByText('v1')).toBeInTheDocument();
  });

  it('renders confidence as percentage', () => {
    render(<PredictionResult prediction={goodPrediction} />);
    expect(screen.getByText('92.0%')).toBeInTheDocument();
  });

  it('renders latency', () => {
    render(<PredictionResult prediction={goodPrediction} />);
    expect(screen.getByText('0.50ms')).toBeInTheDocument();
  });

  it('shows GOOD CREDIT for result=1', () => {
    render(<PredictionResult prediction={goodPrediction} />);
    expect(screen.getByText(/GOOD CREDIT/)).toBeInTheDocument();
  });

  it('shows BAD CREDIT for result=0', () => {
    render(<PredictionResult prediction={badPrediction} />);
    expect(screen.getByText(/BAD CREDIT/)).toBeInTheDocument();
  });

  it('applies good class for result=1', () => {
    const { container } = render(<PredictionResult prediction={goodPrediction} />);
    const verdict = container.querySelector('.verdict-text');
    expect(verdict).toHaveClass('good');
  });

  it('applies bad class for result=0', () => {
    const { container } = render(<PredictionResult prediction={badPrediction} />);
    const verdict = container.querySelector('.verdict-text');
    expect(verdict).toHaveClass('bad');
  });
});
