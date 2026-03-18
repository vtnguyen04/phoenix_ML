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
  it('renders LIVE badge', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={false} />);
    expect(screen.getByText('LIVE')).toBeInTheDocument();
  });

  it('renders customer selector with 10 buttons', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={false} />);
    expect(screen.getAllByRole('button').length).toBeGreaterThanOrEqual(10);
  });

  it('renders Predict Credit Risk button', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={false} />);
    expect(screen.getByText(/Predict Credit Risk/)).toBeInTheDocument();
  });

  it('disables predict button when loading', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={true} />);
    const btns = screen.getAllByRole('button');
    const predictBtn = btns.find(b => b.textContent?.includes('Predict Credit Risk'));
    expect(predictBtn).toBeDisabled();
  });

  it('shows spinner when loading', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={true} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('calls onPredict with default selected customer', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={false} />);
    const btns = screen.getAllByRole('button');
    const predictBtn = btns.find(b => b.textContent?.includes('Predict Credit Risk'));
    fireEvent.click(predictBtn!);
    expect(onPredict).toHaveBeenCalledWith('customer-0001');
  });

  it('calls onPredict with selected customer after switching', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={false} />);
    // Click on customer #0005
    fireEvent.click(screen.getByText('#0005'));
    // Then click predict
    const btns = screen.getAllByRole('button');
    const predictBtn = btns.find(b => b.textContent?.includes('Predict Credit Risk'));
    fireEvent.click(predictBtn!);
    expect(onPredict).toHaveBeenCalledWith('customer-0005');
  });

  it('does not show prediction result when null', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={null} loading={false} />);
    expect(screen.queryByText(/GOOD CREDIT/)).not.toBeInTheDocument();
    expect(screen.queryByText(/BAD CREDIT/)).not.toBeInTheDocument();
  });

  it('shows prediction result after predict', () => {
    const onPredict = vi.fn();
    render(<PredictionPanel onPredict={onPredict} prediction={mockPrediction} loading={false} />);
    expect(screen.getByText(/GOOD CREDIT/)).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('0.42ms')).toBeInTheDocument();
  });
});
