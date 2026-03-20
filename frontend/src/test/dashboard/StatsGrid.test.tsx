import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatsGrid } from '../../components/dashboard/StatsGrid';

describe('StatsGrid', () => {
  it('renders all four classification metric cards', () => {
    const metrics = { accuracy: 0.785, f1_score: 0.854, precision: 0.813, recall: 0.9 };
    render(<StatsGrid metrics={metrics} taskType="classification" />);
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('F1 Score')).toBeInTheDocument();
    expect(screen.getByText('Precision')).toBeInTheDocument();
    expect(screen.getByText('Recall')).toBeInTheDocument();
  });

  it('formats percentages correctly', () => {
    const metrics = { accuracy: 0.785, f1_score: 0.854, precision: 0.813, recall: 0.9 };
    render(<StatsGrid metrics={metrics} taskType="classification" />);
    expect(screen.getByText('78.5%')).toBeInTheDocument();
    expect(screen.getByText('85.4%')).toBeInTheDocument();
    expect(screen.getByText('81.3%')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();
  });

  it('shows dashes when metrics undefined', () => {
    render(<StatsGrid metrics={undefined} taskType="classification" />);
    const dashes = screen.getAllByText('—');
    expect(dashes).toHaveLength(4);
  });

  it('shows subtitles for each card', () => {
    const metrics = { accuracy: 0.785, f1_score: 0.854, precision: 0.813, recall: 0.9 };
    render(<StatsGrid metrics={metrics} taskType="classification" />);
    expect(screen.getByText('Test set evaluation')).toBeInTheDocument();
    expect(screen.getByText('Harmonic mean P/R')).toBeInTheDocument();
  });

  it('shows regression metrics for regression task', () => {
    const metrics = { rmse: 0.0234, mae: 0.0156, r2: 0.912, mse: 0.0005 };
    render(<StatsGrid metrics={metrics} taskType="regression" />);
    expect(screen.getByText('RMSE')).toBeInTheDocument();
    expect(screen.getByText('MAE')).toBeInTheDocument();
    expect(screen.getByText('R²')).toBeInTheDocument();
    expect(screen.getByText('MSE')).toBeInTheDocument();
  });

  it('auto-detects classification from metrics', () => {
    const metrics = { accuracy: 0.9, f1_score: 0.85 };
    render(<StatsGrid metrics={metrics} />);
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
  });

  it('auto-detects regression from metrics', () => {
    const metrics = { rmse: 0.5, mae: 0.3 };
    render(<StatsGrid metrics={metrics} />);
    expect(screen.getByText('RMSE')).toBeInTheDocument();
  });
});
