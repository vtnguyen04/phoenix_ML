import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatCard } from '../../components/ui/StatCard';

describe('StatCard', () => {
  it('renders label, value and sub text', () => {
    render(<StatCard label="Test Label" value="42" sub="some sub" color="blue" />);
    expect(screen.getByText('Test Label')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('some sub')).toBeInTheDocument();
  });

  it('applies correct color class for blue', () => {
    const { container } = render(<StatCard label="L" value="V" sub="S" color="blue" />);
    expect(container.firstChild).toHaveClass('stat-card', 'blue');
  });

  it('applies correct color class for green', () => {
    const { container } = render(<StatCard label="L" value="V" sub="S" color="green" />);
    expect(container.firstChild).toHaveClass('stat-card', 'green');
  });

  it('applies correct color class for orange', () => {
    const { container } = render(<StatCard label="L" value="V" sub="S" color="orange" />);
    expect(container.firstChild).toHaveClass('stat-card', 'orange');
  });

  it('applies correct color class for red', () => {
    const { container } = render(<StatCard label="L" value="V" sub="S" color="red" />);
    expect(container.firstChild).toHaveClass('stat-card', 'red');
  });
});
