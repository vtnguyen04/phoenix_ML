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
    const onScan = vi.fn();
    render(<DriftPanel drift={null} onScan={onScan} loading={false} />);
    expect(screen.getByText('Run a drift scan to analyze model health')).toBeInTheDocument();
  });

  it('shows Scan Drift button', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={null} onScan={onScan} loading={false} />);
    expect(screen.getByText(/Scan Drift/)).toBeInTheDocument();
  });

  it('calls onScan when button is clicked', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={null} onScan={onScan} loading={false} />);
    fireEvent.click(screen.getByText(/Scan Drift/));
    expect(onScan).toHaveBeenCalledOnce();
  });

  it('disables button when loading', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={null} onScan={onScan} loading={true} />);
    const btn = screen.getByRole('button');
    expect(btn).toBeDisabled();
  });

  it('shows spinner when loading', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={null} onScan={onScan} loading={true} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('displays stable drift data', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={stableDrift} onScan={onScan} loading={false} />);
    expect(screen.getByText(/credit_amount/)).toBeInTheDocument();
    expect(screen.getByText('STABLE')).toBeInTheDocument();
    expect(screen.getByText('Model is stable. No action needed.')).toBeInTheDocument();
  });

  it('displays drifted data with danger badge', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={driftedReport} onScan={onScan} loading={false} />);
    expect(screen.getByText('DRIFTED')).toBeInTheDocument();
    expect(screen.getByText(/duration/)).toBeInTheDocument();
    expect(screen.getByText(/Retrain recommended/)).toBeInTheDocument();
  });

  it('shows KS statistic and p-value formatted', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={stableDrift} onScan={onScan} loading={false} />);
    expect(screen.getByText(/KS=0.0300/)).toBeInTheDocument();
    expect(screen.getByText(/p=0.4500/)).toBeInTheDocument();
  });

  it('renders ✅ icon for stable drift', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={stableDrift} onScan={onScan} loading={false} />);
    expect(screen.getByText('✅')).toBeInTheDocument();
  });

  it('renders ⚠️ icon for detected drift', () => {
    const onScan = vi.fn();
    render(<DriftPanel drift={driftedReport} onScan={onScan} loading={false} />);
    expect(screen.getByText('⚠️')).toBeInTheDocument();
  });
});
