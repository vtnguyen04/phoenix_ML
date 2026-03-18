import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatsGrid } from '../../components/dashboard/StatsGrid';

describe('StatsGrid', () => {
  it('renders all four stat cards', () => {
    render(<StatsGrid predictionCount={5} lastLatency={0.42} modelAccuracy="78.5%" />);
    expect(screen.getByText('Total Predictions')).toBeInTheDocument();
    expect(screen.getByText('Last Latency')).toBeInTheDocument();
    expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Active Models')).toBeInTheDocument();
  });

  it('displays prediction count', () => {
    render(<StatsGrid predictionCount={42} lastLatency={null} modelAccuracy="80%" />);
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('displays formatted latency', () => {
    render(<StatsGrid predictionCount={0} lastLatency={1.234} modelAccuracy="80%" />);
    expect(screen.getByText('1.2ms')).toBeInTheDocument();
  });

  it('displays dash when latency is null', () => {
    render(<StatsGrid predictionCount={0} lastLatency={null} modelAccuracy="80%" />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('displays model accuracy', () => {
    render(<StatsGrid predictionCount={0} lastLatency={null} modelAccuracy="78.5%" />);
    expect(screen.getByText('78.5%')).toBeInTheDocument();
  });
});
