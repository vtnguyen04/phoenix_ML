import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ServicesStatus } from '../../components/dashboard/ServicesStatus';

describe('ServicesStatus', () => {
  it('renders all 7 infrastructure services', () => {
    render(<ServicesStatus />);
    expect(screen.getByText('API Server')).toBeInTheDocument();
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument();
    expect(screen.getByText('Redis')).toBeInTheDocument();
    expect(screen.getByText('Kafka')).toBeInTheDocument();
    expect(screen.getByText('Prometheus')).toBeInTheDocument();
    expect(screen.getByText('Grafana')).toBeInTheDocument();
    expect(screen.getByText('MinIO')).toBeInTheDocument();
  });

  it('renders Infrastructure title', () => {
    render(<ServicesStatus />);
    expect(screen.getByText(/Infrastructure/)).toBeInTheDocument();
  });

  it('shows ALL ONLINE badge', () => {
    render(<ServicesStatus />);
    expect(screen.getByText('ALL ONLINE')).toBeInTheDocument();
  });

  it('shows correct ports for each service', () => {
    render(<ServicesStatus />);
    expect(screen.getByText(':8001')).toBeInTheDocument();
    expect(screen.getByText(':5433')).toBeInTheDocument();
    expect(screen.getByText(':6380')).toBeInTheDocument();
    expect(screen.getByText(':9094')).toBeInTheDocument();
  });
});
