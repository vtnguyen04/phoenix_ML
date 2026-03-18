import type { DriftReport } from '../../api/mlService';
import { StatusBadge } from '../ui/StatusBadge';
import { Spinner } from '../ui/Spinner';

interface DriftPanelProps {
  drift: DriftReport | null;
  onScan: () => void;
  loading: boolean;
}

/**
 * DriftPanel — Self-healing drift monitoring display.
 * SRP: Only handles drift data visualization and scan trigger.
 * DIP: Depends on DriftReport interface, not drift calculation logic.
 */
export function DriftPanel({ drift, onScan, loading }: DriftPanelProps) {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">🛡️ Self-Healing Monitor</h2>
        <button className="btn btn-sm" onClick={onScan} disabled={loading}>
          {loading ? <Spinner /> : '🔄'} Scan Drift
        </button>
      </div>

      {drift ? (
        <DriftDetails drift={drift} />
      ) : (
        <EmptyDriftState />
      )}
    </div>
  );
}

/** Drift detail view — separated for testability. */
function DriftDetails({ drift }: { drift: DriftReport }) {
  return (
    <>
      <div className="drift-status-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div className={`drift-indicator ${drift.drift_detected ? 'drifted' : 'stable'}`}>
            {drift.drift_detected ? '⚠️' : '✅'}
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>
              Feature: {drift.feature_name}
            </div>
            <div style={{
              fontSize: 12,
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
            }}>
              KS={drift.statistic.toFixed(4)} · p={drift.p_value.toFixed(4)}
            </div>
          </div>
        </div>
        <StatusBadge variant={drift.drift_detected ? 'danger' : 'success'}>
          {drift.drift_detected ? 'DRIFTED' : 'STABLE'}
        </StatusBadge>
      </div>
      <div className="drift-recommendation">
        {drift.recommendation}
      </div>
    </>
  );
}

/** Empty state when no drift scan has been performed. */
function EmptyDriftState() {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">📊</div>
      <div className="empty-state-text">Run a drift scan to analyze model health</div>
    </div>
  );
}
