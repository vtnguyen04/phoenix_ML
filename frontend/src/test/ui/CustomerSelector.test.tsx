import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CustomerSelector } from '../../components/ui/CustomerSelector';

describe('CustomerSelector', () => {
  it('renders correct number of customer buttons', () => {
    const onSelect = vi.fn();
    render(<CustomerSelector onSelect={onSelect} selected="customer-0001" count={5} />);
    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(5);
  });

  it('renders default 10 buttons when count not specified', () => {
    const onSelect = vi.fn();
    render(<CustomerSelector onSelect={onSelect} selected="customer-0001" />);
    expect(screen.getAllByRole('button')).toHaveLength(10);
  });

  it('highlights selected customer', () => {
    const onSelect = vi.fn();
    render(<CustomerSelector onSelect={onSelect} selected="customer-0002" count={5} />);
    const selectedBtn = screen.getByText('#0002');
    expect(selectedBtn).toHaveClass('active');
  });

  it('calls onSelect with customer ID on click', () => {
    const onSelect = vi.fn();
    render(<CustomerSelector onSelect={onSelect} selected="customer-0000" count={5} />);
    fireEvent.click(screen.getByText('#0003'));
    expect(onSelect).toHaveBeenCalledWith('customer-0003');
  });

  it('sets aria-pressed on selected button', () => {
    const onSelect = vi.fn();
    render(<CustomerSelector onSelect={onSelect} selected="customer-0001" count={3} />);
    expect(screen.getByText('#0001')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('#0000')).toHaveAttribute('aria-pressed', 'false');
  });
});
