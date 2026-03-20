import type { DriftReport } from '../../api/mlService';
import { StatusBadge } from '../ui/StatusBadge';
import { Spinner } from '../ui/Spinner';

interface DriftPanelProps {
  drift: DriftReport | null;
  onScan: () => void;
  loading: boolean;
  error?: string | null;
}

export function DriftPanel({ drift, onScan, loading, error }: DriftPanelProps) {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Drift Monitor</h2>
        <button className="btn btn-sm" onClick={onScan} disabled={loading}>
          {loading ? <Spinner /> : '🔄'} Scan Now
        </button>
      </div>

      {error ? (
        <div className="drift-recommendation drift-error">⚠️ {error}</div>
      ) : drift ? (
        <DriftDetails drift={drift} />
      ) : (
        <EmptyDriftState />
      )}
    </div>
  );
}

function DriftDetails({ drift }: { drift: DriftReport }) {
  return (
    <>
      <div className="drift-status-bar">
        <div className="drift-status-info">
          <div className={`drift-indicator ${drift.drift_detected ? 'drifted' : 'stable'}`}>
            {drift.drift_detected ? '⚠️' : '✅'}
          </div>
          <div>
            <div className="drift-feature-name">Feature: {drift.feature_name}</div>
            <div className="drift-detail-mono">
              {drift.method.toUpperCase()}={drift.statistic.toFixed(4)} · p={drift.p_value.toFixed(4)} · n={drift.sample_size}
            </div>
          </div>
        </div>
        <StatusBadge variant={drift.drift_detected ? 'danger' : 'success'}>
          {drift.drift_detected ? 'DRIFTED' : 'STABLE'}
        </StatusBadge>
      </div>
      <div className="drift-recommendation">{drift.recommendation}</div>
    </>
  );
}

function EmptyDriftState() {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">📊</div>
      <div className="empty-state-text">
        Drift reports will appear automatically once predictions are logged by the monitoring loop.
      </div>
    </div>
  );
}
