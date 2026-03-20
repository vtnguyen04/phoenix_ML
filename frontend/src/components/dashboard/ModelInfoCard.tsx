import type { ModelInfo } from '../../api/mlService';

interface ModelInfoCardProps {
  model: ModelInfo | null;
}

export function ModelInfoCard({ model }: ModelInfoCardProps) {
  if (!model) {
    return (
      <div className="card model-info-card">
        <div className="card-header">
          <h2 className="card-title">Champion Model</h2>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">⏳</div>
          <div className="empty-state-text">Loading model information...</div>
        </div>
      </div>
    );
  }

  const metrics = model.metadata.metrics;
  const featureCount = model.metadata.features?.length ?? 0;

  return (
    <div className="card model-info-card">
      <div className="card-header">
        <h2 className="card-title">Champion Model</h2>
        <span className="badge badge-success">{model.metadata.role?.toUpperCase() ?? 'ACTIVE'}</span>
      </div>

      <div className="model-info-grid">
        <InfoRow label="Model ID" value={model.model_id} />
        <InfoRow label="Version" value={model.version} />
        <InfoRow label="Type" value={metrics?.model_type ?? 'Unknown'} mono />
        <InfoRow label="Dataset" value={model.metadata.dataset ?? 'N/A'} />
        <InfoRow label="Features" value={`${featureCount} features`} />
        <InfoRow
          label="Training"
          value={`${metrics?.train_samples ?? '?'} train · ${metrics?.test_samples ?? '?'} test`}
        />
        <InfoRow
          label="CV Score"
          value={
            metrics?.cv_accuracy_mean
              ? `${(metrics.cv_accuracy_mean * 100).toFixed(1)}% ± ${((metrics.cv_accuracy_std ?? 0) * 100).toFixed(1)}%`
              : 'N/A'
          }
        />
      </div>
    </div>
  );
}

function InfoRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="model-info-row">
      <span className="model-info-label">{label}</span>
      <span className={`model-info-value${mono ? ' mono' : ''}`}>{value}</span>
    </div>
  );
}
