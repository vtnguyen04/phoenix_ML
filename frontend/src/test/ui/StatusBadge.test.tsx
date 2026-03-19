import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '../../components/ui/StatusBadge';

describe('StatusBadge', () => {
  it('renders children text', () => {
    render(<StatusBadge variant="success">ONLINE</StatusBadge>);
    expect(screen.getByText('ONLINE')).toBeInTheDocument();
  });

  it('applies success variant class', () => {
    const { container } = render(<StatusBadge variant="success">OK</StatusBadge>);
    expect(container.firstChild).toHaveClass('badge', 'badge-success');
  });

  it('applies danger variant class', () => {
    const { container } = render(<StatusBadge variant="danger">ERROR</StatusBadge>);
    expect(container.firstChild).toHaveClass('badge', 'badge-danger');
  });

  it('applies warning variant class', () => {
    const { container } = render(<StatusBadge variant="warning">WARN</StatusBadge>);
    expect(container.firstChild).toHaveClass('badge', 'badge-warning');
  });

  it('applies info variant class', () => {
    const { container } = render(<StatusBadge variant="info">INFO</StatusBadge>);
    expect(container.firstChild).toHaveClass('badge', 'badge-info');
  });
});
