import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatsGrid } from '../../components/dashboard/StatsGrid';

describe('StatsGrid', () => {
  it('renders all four metric cards', () => {
    render(<StatsGrid accuracy={0.785} f1Score={0.854} precision={0.813} recall={0.9} />);
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('F1 Score')).toBeInTheDocument();
    expect(screen.getByText('Precision')).toBeInTheDocument();
    expect(screen.getByText('Recall')).toBeInTheDocument();
  });

  it('formats percentages correctly', () => {
    render(<StatsGrid accuracy={0.785} f1Score={0.854} precision={0.813} recall={0.9} />);
    expect(screen.getByText('78.5%')).toBeInTheDocument();
    expect(screen.getByText('85.4%')).toBeInTheDocument();
    expect(screen.getByText('81.3%')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();
  });

  it('shows dashes for null values', () => {
    render(<StatsGrid accuracy={null} f1Score={null} precision={null} recall={null} />);
    const dashes = screen.getAllByText('—');
    expect(dashes).toHaveLength(4);
  });

  it('shows subtitles for each card', () => {
    render(<StatsGrid accuracy={0.785} f1Score={0.854} precision={0.813} recall={0.9} />);
    expect(screen.getByText('Test set evaluation')).toBeInTheDocument();
    expect(screen.getByText('Harmonic mean P/R')).toBeInTheDocument();
  });
});
