import type { ModelInfo } from '../../api/mlService';
import type { TaskType } from '../../config';
import { getMetricsForTask } from '../../config';

interface ModelComparisonProps {
  champion: ModelInfo | null;
  challengers: ModelInfo[];
  taskType: TaskType;
}

export function ModelComparison({ champion, challengers, taskType }: ModelComparisonProps) {
  const metricDefs = getMetricsForTask(taskType);

  if (!champion) {
    return (
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Champion vs Challenger</h2>
        </div>
        <div className="empty-state">
          <div className="empty-state-text">Loading model data...</div>
        </div>
      </div>
    );
  }

  const activeChallenger = challengers[0] ?? null;

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Champion vs Challenger</h2>
        {activeChallenger ? (
          <span className="badge badge-warning">COMPARISON</span>
        ) : (
          <span className="badge badge-info">CHAMPION ONLY</span>
        )}
      </div>

      <table className="comparison-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>
              Champion
              <span className="version-tag">{champion.version}</span>
            </th>
            {activeChallenger && (
              <th>
                Challenger
                <span className="version-tag">{activeChallenger.version}</span>
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="metric-label">Model Type</td>
            <td className="metric-value">{champion.metadata.metrics?.model_type ?? '—'}</td>
            {activeChallenger && (
              <td className="metric-value">{activeChallenger.metadata.metrics?.model_type ?? '—'}</td>
            )}
          </tr>
          <tr>
            <td className="metric-label">Dataset</td>
            <td className="metric-value">{champion.metadata.dataset ?? '—'}</td>
            {activeChallenger && (
              <td className="metric-value">{activeChallenger.metadata.dataset ?? '—'}</td>
            )}
          </tr>
          <tr>
            <td className="metric-label">Features</td>
            <td className="metric-value">{champion.metadata.metrics?.n_features ?? '—'}</td>
            {activeChallenger && (
              <td className="metric-value">{activeChallenger.metadata.metrics?.n_features ?? '—'}</td>
            )}
          </tr>
          <tr>
            <td className="metric-label">Training</td>
            <td className="metric-value">
              {champion.metadata.metrics?.train_samples ?? '?'} / {champion.metadata.metrics?.test_samples ?? '?'}
            </td>
            {activeChallenger && (
              <td className="metric-value">
                {activeChallenger.metadata.metrics?.train_samples ?? '?'} / {activeChallenger.metadata.metrics?.test_samples ?? '?'}
              </td>
            )}
          </tr>
          {metricDefs.map((def) => {
            const champVal = champion.metadata.metrics?.[def.key] as number | undefined;
            const challVal = activeChallenger?.metadata.metrics?.[def.key] as number | undefined;
            const isHigherBetter = !['rmse', 'mae', 'mse'].includes(def.key);
            const diff = champVal != null && challVal != null
              ? (challVal - champVal) * (isHigherBetter ? 1 : -1)
              : null;

            return (
              <tr key={def.key}>
                <td className="metric-label">{def.label}</td>
                <td className="metric-value metric-highlight">{champVal != null ? def.format(champVal) : '—'}</td>
                {activeChallenger && (
                  <td className={`metric-value metric-highlight ${diff != null ? (diff > 0 ? 'metric-better' : diff < 0 ? 'metric-worse' : '') : ''}`}>
                    {challVal != null ? def.format(challVal) : '—'}
                    {diff != null && diff !== 0 && (
                      <span className="metric-diff">{diff > 0 ? '▲' : '▼'}</span>
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
