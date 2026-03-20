import { useEffect, useState } from 'react';
import type { PerformanceReport } from '../../api/mlService';
import { mlService } from '../../api/mlService';

interface PerformancePanelProps {
  modelId: string;
}

export function PerformancePanel({ modelId }: PerformancePanelProps) {
  const [perf, setPerf] = useState<PerformanceReport | null>(null);

  useEffect(() => {
    if (!modelId) return;
    mlService.getPerformance(modelId).then(setPerf).catch(() => setPerf(null));
  }, [modelId]);

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Live Performance</h2>
        <span className="badge badge-success">MONITORING</span>
      </div>

      {perf && perf.total_predictions > 0 ? (
        <div className="perf-grid">
          <div className="perf-stat">
            <div className="perf-value">{perf.total_predictions.toLocaleString()}</div>
            <div className="perf-label">Total Predictions</div>
          </div>
          <div className="perf-stat">
            <div className="perf-value">{perf.metrics.avg_latency_ms.toFixed(1)}ms</div>
            <div className="perf-label">Avg Latency</div>
          </div>
          <div className="perf-stat">
            <div className="perf-value">{(perf.metrics.avg_confidence * 100).toFixed(0)}%</div>
            <div className="perf-label">Avg Confidence</div>
          </div>
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state-text">No prediction data yet for this model.</div>
        </div>
      )}
    </div>
  );
}
