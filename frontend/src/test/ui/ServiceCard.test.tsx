import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ServiceCard } from '../../components/ui/ServiceCard';

describe('ServiceCard', () => {
  it('renders service name', () => {
    render(<ServiceCard name="PostgreSQL" port={5432} icon="🐘" />);
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument();
  });

  it('renders port with colon prefix', () => {
    render(<ServiceCard name="Redis" port={6379} icon="⚡" />);
    expect(screen.getByText(':6379')).toBeInTheDocument();
  });

  it('renders icon', () => {
    render(<ServiceCard name="Kafka" port={9092} icon="📨" />);
    expect(screen.getByText('📨')).toBeInTheDocument();
  });
});
