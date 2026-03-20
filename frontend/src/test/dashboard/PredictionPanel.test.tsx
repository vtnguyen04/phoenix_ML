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

describe('PredictionPanel', () => {
  const defaultProps = {
    modelId: 'credit-risk',
    taskType: 'classification' as const,
    onPredict: vi.fn(),
    prediction: null as PredictionResponse | null,
    loading: false,
  };

  it('renders LIVE badge', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.getByText('LIVE')).toBeInTheDocument();
  });

  it('renders entity selector with 10 buttons', () => {
    render(<PredictionPanel {...defaultProps} />);
    // 10 entity buttons + 1 predict button = 11
    expect(screen.getAllByRole('button').length).toBeGreaterThanOrEqual(10);
  });

  it('renders model-specific predict button', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.getByText(/Predict Credit Risk/)).toBeInTheDocument();
  });

  it('renders fraud-detection predict button', () => {
    render(<PredictionPanel {...defaultProps} modelId="fraud-detection" />);
    expect(screen.getByText(/Predict Fraud Detection/)).toBeInTheDocument();
  });

  it('disables predict button when loading', () => {
    render(<PredictionPanel {...defaultProps} loading={true} />);
    const btns = screen.getAllByRole('button');
    const predictBtn = btns.find(b => b.textContent?.includes('Predict'));
    expect(predictBtn).toBeDisabled();
  });

  it('shows spinner when loading', () => {
    render(<PredictionPanel {...defaultProps} loading={true} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('calls onPredict with default entity', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel {...defaultProps} onPredict={onPredict} />);
    const btns = screen.getAllByRole('button');
    const predictBtn = btns.find(b => b.textContent?.includes('Predict'));
    fireEvent.click(predictBtn!);
    expect(onPredict).toHaveBeenCalledWith('customer-0001');
  });

  it('calls onPredict with selected entity after switching', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel {...defaultProps} onPredict={onPredict} />);
    fireEvent.click(screen.getByText('#0005'));
    const btns = screen.getAllByRole('button');
    const predictBtn = btns.find(b => b.textContent?.includes('Predict'));
    fireEvent.click(predictBtn!);
    expect(onPredict).toHaveBeenCalledWith('customer-0005');
  });

  it('does not show prediction result when null', () => {
    render(<PredictionPanel {...defaultProps} />);
    expect(screen.queryByText(/GOOD CREDIT/)).not.toBeInTheDocument();
  });

  it('shows prediction result after predict', () => {
    render(<PredictionPanel {...defaultProps} prediction={mockPrediction} />);
    expect(screen.getByText(/GOOD CREDIT/)).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
  });
});
