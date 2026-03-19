import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Spinner } from '../../components/ui/Spinner';

describe('Spinner', () => {
  it('renders with default size', () => {
    render(<Spinner />);
    const spinner = screen.getByRole('status');
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass('spinner');
  });

  it('renders with custom size', () => {
    render(<Spinner size={24} />);
    const spinner = screen.getByRole('status');
    expect(spinner).toHaveStyle({ width: '24px', height: '24px' });
  });

  it('has accessible label', () => {
    render(<Spinner />);
    expect(screen.getByLabelText('Loading')).toBeInTheDocument();
  });
});
