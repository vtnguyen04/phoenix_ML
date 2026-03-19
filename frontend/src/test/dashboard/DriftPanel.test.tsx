import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DriftPanel } from '../../components/dashboard/DriftPanel';
import type { DriftReport } from '../../api/mlService';

const stableDrift: DriftReport = {
  feature_name: 'credit_amount',
  drift_detected: false,
  p_value: 0.45,
  statistic: 0.03,
  threshold: 0.05,
  method: 'ks',
  recommendation: 'Model is stable. No action needed.',
  sample_size: 100,
};

const driftedReport: DriftReport = {
  feature_name: 'duration',
  drift_detected: true,
  p_value: 0.001,
  statistic: 0.42,
  threshold: 0.05,
  method: 'ks',
  recommendation: 'Significant drift detected. Retrain recommended.',
  sample_size: 200,
};

describe('DriftPanel', () => {
  it('shows empty state when drift is null', () => {
    render(<DriftPanel drift={null} onScan={vi.fn()} loading={false} />);
    expect(screen.getByText(/Drift reports will appear automatically/)).toBeInTheDocument();
  });

  it('shows Scan Now button', () => {
    render(<DriftPanel drift={null} onScan={vi.fn()} loading={false} />);
    expect(screen.getByText(/Scan Now/)).toBeInTheDocument();
  });

  it('calls onScan when button is clicked', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={null} onScan={onScan} loading={false} />);
    fireEvent.click(screen.getByText(/Scan Now/));
    expect(onScan).toHaveBeenCalledOnce();
  });

  it('disables button when loading', () => {
    render(<DriftPanel drift={null} onScan={vi.fn()} loading={true} />);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('displays stable drift data', () => {
    render(<DriftPanel drift={stableDrift} onScan={vi.fn()} loading={false} />);
    expect(screen.getByText(/credit_amount/)).toBeInTheDocument();
    expect(screen.getByText('STABLE')).toBeInTheDocument();
    expect(screen.getByText('Model is stable. No action needed.')).toBeInTheDocument();
  });

  it('displays drifted data with danger badge', () => {
    render(<DriftPanel drift={driftedReport} onScan={vi.fn()} loading={false} />);
    expect(screen.getByText('DRIFTED')).toBeInTheDocument();
    expect(screen.getByText(/duration/)).toBeInTheDocument();
  });

  it('shows method and sample size', () => {
    render(<DriftPanel drift={stableDrift} onScan={vi.fn()} loading={false} />);
    expect(screen.getByText(/KS=0.0300/)).toBeInTheDocument();
    expect(screen.getByText(/n=100/)).toBeInTheDocument();
  });

  it('shows error message inline when error prop is set', () => {
    render(
      <DriftPanel drift={null} onScan={vi.fn()} loading={false} error="Not enough prediction data" />
    );
    expect(screen.getByText(/Not enough prediction data/)).toBeInTheDocument();
  });
});
