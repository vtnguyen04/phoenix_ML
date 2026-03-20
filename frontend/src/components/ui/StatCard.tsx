interface StatCardProps {
  label: string;
  value: string;
  sub: string;
  color: 'blue' | 'orange' | 'green' | 'red' | 'purple';
}

/**
 * StatCard — Reusable metric display card.
 * SRP: Only renders a single statistic.
 * DIP: Depends on abstract props interface, not concrete data sources.
 */
export function StatCard({ label, value, sub, color }: StatCardProps) {
  return (
    <div className={`stat-card ${color}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  );
}
