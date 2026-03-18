import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Sidebar } from '../../components/layout/Sidebar';

describe('Sidebar', () => {
  it('renders brand name', () => {
    render(<Sidebar health={null} />);
    expect(screen.getByText('PHOENIX ML')).toBeInTheDocument();
  });

  it('shows Connecting when health is null', () => {
    render(<Sidebar health={null} />);
    expect(screen.getByText('Connecting...')).toBeInTheDocument();
  });

  it('shows System Online when health data is provided', () => {
    render(<Sidebar health={{ status: 'healthy', version: '1.0.0' }} />);
    expect(screen.getByText('System Online')).toBeInTheDocument();
  });

  it('shows version when health data is provided', () => {
    render(<Sidebar health={{ status: 'healthy', version: '2.5.0' }} />);
    expect(screen.getByText('v2.5.0')).toBeInTheDocument();
  });

  it('renders offline status dot when health is null', () => {
    const { container } = render(<Sidebar health={null} />);
    const dot = container.querySelector('.status-dot');
    expect(dot).toHaveClass('offline');
  });

  it('renders online status dot when health is provided', () => {
    const { container } = render(<Sidebar health={{ status: 'healthy', version: '1.0.0' }} />);
    const dot = container.querySelector('.status-dot');
    expect(dot).toHaveClass('online');
  });

  it('renders nav links to external services', () => {
    render(<Sidebar health={null} />);
    const grafanaLink = screen.getByText(/Grafana/).closest('a');
    expect(grafanaLink).toHaveAttribute('href', 'http://localhost:3001');
    expect(grafanaLink).toHaveAttribute('target', '_blank');
  });
});
