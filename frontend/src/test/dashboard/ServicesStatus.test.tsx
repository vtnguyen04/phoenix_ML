import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ServicesStatus } from '../../components/dashboard/ServicesStatus';

describe('ServicesStatus', () => {
  it('renders Infrastructure title', () => {
    render(<ServicesStatus />);
    expect(screen.getByText('🏗️ Infrastructure')).toBeInTheDocument();
  });

  it('renders all 8 services', () => {
    render(<ServicesStatus />);
    expect(screen.getByText('API Server')).toBeInTheDocument();
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument();
    expect(screen.getByText('Redis')).toBeInTheDocument();
    expect(screen.getByText('Kafka')).toBeInTheDocument();
    expect(screen.getByText('Prometheus')).toBeInTheDocument();
    expect(screen.getByText('Grafana')).toBeInTheDocument();
    expect(screen.getByText('Jaeger')).toBeInTheDocument();
    expect(screen.getByText('MinIO')).toBeInTheDocument();
  });

  it('shows ALL ONLINE badge', () => {
    render(<ServicesStatus />);
    expect(screen.getByText('ALL ONLINE')).toBeInTheDocument();
  });

  it('renders port numbers', () => {
    render(<ServicesStatus />);
    expect(screen.getByText(':8001')).toBeInTheDocument();
    expect(screen.getByText(':16686')).toBeInTheDocument();
  });
});
