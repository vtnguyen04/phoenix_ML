interface StatusBadgeProps {
  variant: 'success' | 'danger' | 'warning' | 'info';
  children: React.ReactNode;
}

/**
 * StatusBadge — Reusable status indicator badge.
 * SRP: Only renders a colored badge with text.
 */
export function StatusBadge({ variant, children }: StatusBadgeProps) {
  return <span className={`badge badge-${variant}`}>{children}</span>;
}
