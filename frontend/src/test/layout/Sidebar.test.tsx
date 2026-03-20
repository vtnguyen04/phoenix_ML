import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Sidebar } from '../../components/layout/Sidebar';

describe('Sidebar', () => {
  it('renders brand name', () => {
    render(<Sidebar health={null} modelCount={0} />);
    expect(screen.getByText('PHOENIX ML')).toBeInTheDocument();
  });

  it('renders brand icon', () => {
    render(<Sidebar health={null} modelCount={0} />);
    expect(screen.getByText('⚡')).toBeInTheDocument();
  });

  it('renders Dashboard nav item', () => {
    render(<Sidebar health={null} modelCount={0} />);
    expect(screen.getByText(/Dashboard/)).toBeInTheDocument();
  });

  it('renders Grafana link with correct URL', () => {
    render(<Sidebar health={null} modelCount={0} />);
    const grafanaLink = screen.getByText(/Grafana/).closest('a');
    expect(grafanaLink).toHaveAttribute('href', 'http://localhost:3001');
    expect(grafanaLink).toHaveAttribute('target', '_blank');
  });

  it('renders Jaeger link', () => {
    render(<Sidebar health={null} modelCount={0} />);
    const jaegerLink = screen.getByText(/Jaeger/).closest('a');
    expect(jaegerLink).toHaveAttribute('href', 'http://localhost:16686');
  });

  it('shows Connecting when health is null', () => {
    render(<Sidebar health={null} modelCount={0} />);
    expect(screen.getByText('Connecting...')).toBeInTheDocument();
  });

  it('shows System Online when health is available', () => {
    render(<Sidebar health={{ status: 'healthy', version: '0.1.0' }} modelCount={3} />);
    expect(screen.getByText('System Online')).toBeInTheDocument();
    expect(screen.getByText('v0.1.0')).toBeInTheDocument();
  });

  it('shows model count', () => {
    render(<Sidebar health={null} modelCount={5} />);
    expect(screen.getByText('5 models')).toBeInTheDocument();
  });
});
