import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PipelineStatus } from '../../components/dashboard/PipelineStatus';

describe('PipelineStatus', () => {
  it('renders pipeline title', () => {
    render(<PipelineStatus />);
    expect(screen.getByText('⚙️ MLOps Pipeline')).toBeInTheDocument();
  });

  it('renders all 6 pipeline stages', () => {
    render(<PipelineStatus />);
    expect(screen.getByText('Data Ingestion')).toBeInTheDocument();
    expect(screen.getByText('Feature Engineering')).toBeInTheDocument();
    expect(screen.getByText('Model Training')).toBeInTheDocument();
    expect(screen.getByText('Evaluation')).toBeInTheDocument();
    expect(screen.getByText('Deployment')).toBeInTheDocument();
    expect(screen.getByText('Monitoring')).toBeInTheDocument();
  });

  it('shows completed status for stages with data', () => {
    render(
      <PipelineStatus
        modelType="GradientBoosting"
        dataset="german-credit"
        featureCount={30}
        hasMetrics={true}
        hasDrift={true}
      />
    );
    expect(screen.getByText('german-credit')).toBeInTheDocument();
    expect(screen.getByText('30 features')).toBeInTheDocument();
    expect(screen.getByText('GradientBoosting')).toBeInTheDocument();
  });

  it('shows pending status when no data', () => {
    render(<PipelineStatus />);
    expect(screen.getByText('Awaiting data')).toBeInTheDocument();
    expect(screen.getByText('Not trained')).toBeInTheDocument();
  });

  it('shows AUTOMATED badge', () => {
    render(<PipelineStatus />);
    expect(screen.getByText('AUTOMATED')).toBeInTheDocument();
  });
});
