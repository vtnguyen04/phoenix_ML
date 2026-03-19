import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { GrafanaEmbed } from '../../components/dashboard/GrafanaEmbed';

describe('GrafanaEmbed', () => {
  it('renders the section title', () => {
    render(<GrafanaEmbed />);
    expect(screen.getByText(/Live Metrics/)).toBeInTheDocument();
  });

  it('renders an iframe', () => {
    render(<GrafanaEmbed />);
    const iframe = screen.getByTitle('Grafana Dashboard');
    expect(iframe).toBeInTheDocument();
    expect(iframe.tagName).toBe('IFRAME');
  });

  it('uses correct default Grafana URL', () => {
    render(<GrafanaEmbed />);
    const iframe = screen.getByTitle('Grafana Dashboard') as HTMLIFrameElement;
    expect(iframe.src).toContain('localhost:3001');
    expect(iframe.src).toContain('phoenix-ml-prod');
    expect(iframe.src).toContain('kiosk');
  });

  it('renders Open Full link', () => {
    render(<GrafanaEmbed />);
    expect(screen.getByText('Open Full ↗')).toBeInTheDocument();
  });

  it('allows custom dashboard UID', () => {
    render(<GrafanaEmbed dashboardUid="custom-uid" />);
    const iframe = screen.getByTitle('Grafana Dashboard') as HTMLIFrameElement;
    expect(iframe.src).toContain('custom-uid');
  });
});
