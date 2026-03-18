import { StatCard } from '../ui/StatCard';

interface StatsGridProps {
  accuracy: number | null;
  f1Score: number | null;
  precision: number | null;
  recall: number | null;
}

function fmt(v: number | null): string {
  return v !== null ? `${(v * 100).toFixed(1)}%` : '—';
}

export function StatsGrid({ accuracy, f1Score, precision, recall }: StatsGridProps) {
  return (
    <div className="stats-grid">
      <StatCard
        label="Accuracy"
        value={fmt(accuracy)}
        sub="Test set evaluation"
        color="blue"
      />
      <StatCard
        label="F1 Score"
        value={fmt(f1Score)}
        sub="Harmonic mean P/R"
        color="orange"
      />
      <StatCard
        label="Precision"
        value={fmt(precision)}
        sub="True positive rate"
        color="green"
      />
      <StatCard
        label="Recall"
        value={fmt(recall)}
        sub="Sensitivity"
        color="red"
      />
    </div>
  );
}
