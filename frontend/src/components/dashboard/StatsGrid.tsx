import { StatCard } from '../ui/StatCard';

interface StatsGridProps {
  predictionCount: number;
  lastLatency: number | null;
  modelAccuracy: string;
}

/**
 * StatsGrid — Displays key metrics in a responsive grid.
 * SRP: Only responsible for layout of stat cards.
 * OCP: New stats can be added without modifying existing cards.
 */
export function StatsGrid({ predictionCount, lastLatency, modelAccuracy }: StatsGridProps) {
  return (
    <div className="stats-grid">
      <StatCard
        label="Total Predictions"
        value={String(predictionCount)}
        sub="this session"
        color="blue"
      />
      <StatCard
        label="Last Latency"
        value={lastLatency ? `${lastLatency.toFixed(1)}ms` : '—'}
        sub="sub-millisecond"
        color="orange"
      />
      <StatCard
        label="Model Accuracy"
        value={modelAccuracy}
        sub="GradientBoosting · 30 features"
        color="green"
      />
      <StatCard
        label="Active Models"
        value="2"
        sub="Champion + Challenger"
        color="red"
      />
    </div>
  );
}
