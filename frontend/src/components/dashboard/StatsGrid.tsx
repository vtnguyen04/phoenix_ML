import { StatCard } from '../ui/StatCard';
import type { MetricDef } from '../../config';
import { getMetricsForTask, detectTaskType } from '../../config';
import type { TaskType } from '../../config';

interface StatsGridProps {
  metrics: Record<string, number> | undefined;
  taskType?: TaskType;
}

/**
 * StatsGrid — Model-agnostic metric cards.
 * Automatically selects classification or regression metrics
 * based on the task type detected from available metrics.
 */
export function StatsGrid({ metrics, taskType }: StatsGridProps) {
  const resolvedType = taskType ?? detectTaskType(metrics);
  const defs: MetricDef[] = getMetricsForTask(resolvedType);

  return (
    <div className="stats-grid">
      {defs.map((def) => {
        const raw = metrics?.[def.key];
        const value = raw !== undefined ? def.format(raw) : '—';
        return (
          <StatCard
            key={def.key}
            label={def.label}
            value={value}
            sub={def.sub}
            color={def.color}
          />
        );
      })}
    </div>
  );
}
