import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PredictionPanel } from '../../components/dashboard/PredictionPanel';
import type { PredictionResponse } from '../../api/mlService';

const mockPrediction: PredictionResponse = {
  prediction_id: 'test-123',
  model_id: 'credit-risk',
  version: 'v1',
  result: 1,
  confidence: { value: 0.85 },
  latency_ms: 0.42,
};

const featureNames = ['duration', 'credit_amount', 'age'];

describe('PredictionPanel', () => {
  const defaultProps = {
    modelId: 'credit-risk',
    taskType: 'classification' as const,
    featureNames,
    onPredict: vi.fn(),
    prediction: null as PredictionResponse | null,
    loading: false,
  };

  it('renders feature count badge', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.getByText('3 features')).toBeInTheDocument();
  });

  it('renders feature input fields', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.getByLabelText('duration')).toBeInTheDocument();
    expect(screen.getByLabelText('credit_amount')).toBeInTheDocument();
    expect(screen.getByLabelText('age')).toBeInTheDocument();
  });

  it('renders Run Prediction button', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.getByText('Run Prediction')).toBeInTheDocument();
  });

  it('disables button when loading', () => {
    render(<PredictionPanel {...defaultProps} loading={true} />);
    const btn = screen.getByText('Run Prediction').closest('button');
    expect(btn).toBeDisabled();
  });

  it('calls onPredict with feature values', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel {...defaultProps} onPredict={onPredict} />);
    fireEvent.change(screen.getByLabelText('duration'), { target: { value: '12' } });
    fireEvent.change(screen.getByLabelText('credit_amount'), { target: { value: '5000' } });
    fireEvent.change(screen.getByLabelText('age'), { target: { value: '30' } });
    fireEvent.click(screen.getByText('Run Prediction'));
    expect(onPredict).toHaveBeenCalledWith([12, 5000, 30]);
  });

  it('sends zeros for empty fields', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel {...defaultProps} onPredict={onPredict} />);
    fireEvent.click(screen.getByText('Run Prediction'));
    expect(onPredict).toHaveBeenCalledWith([0, 0, 0]);
  });

  it('does not show prediction result when null', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.queryByText(/GOOD CREDIT/)).not.toBeInTheDocument();
  });

  it('shows prediction result', () => {
    render(<PredictionPanel {...defaultProps} prediction={mockPrediction} />);
    expect(screen.getByText(/GOOD CREDIT/)).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
  });

  it('shows empty state when no features', () => {
    render(<PredictionPanel {...defaultProps} featureNames={[]} />);
    expect(screen.getByText(/No feature schema available/)).toBeInTheDocument();
  });

  it('shows filled count', () => {
    render(<PredictionPanel {...defaultProps} />);
    fireEvent.change(screen.getByLabelText('age'), { target: { value: '25' } });
    expect(screen.getByText('1/3 filled')).toBeInTheDocument();
  });
});
