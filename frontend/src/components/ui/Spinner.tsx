interface SpinnerProps {
  size?: number;
}

/**
 * Spinner — Loading indicator.
 * SRP: Only renders a CSS spinner animation.
 */
export function Spinner({ size = 16 }: SpinnerProps) {
  return (
    <span
      className="spinner"
      style={{ width: size, height: size }}
      role="status"
      aria-label="Loading"
    />
  );
}
