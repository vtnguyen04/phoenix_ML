interface PipelineStage {
  name: string;
  icon: string;
  status: 'completed' | 'active' | 'pending';
  detail: string;
}

interface PipelineStatusProps {
  modelType?: string;
  dataset?: string;
  featureCount?: number;
  hasMetrics?: boolean;
  hasDrift?: boolean;
}

export function PipelineStatus({
  modelType,
  dataset,
  featureCount,
  hasMetrics = false,
  hasDrift = false,
}: PipelineStatusProps) {
  const stages: PipelineStage[] = [
    {
      name: 'Data Ingestion',
      icon: '📥',
      status: dataset ? 'completed' : 'pending',
      detail: dataset ?? 'Awaiting data',
    },
    {
      name: 'Feature Engineering',
      icon: '🔧',
      status: featureCount ? 'completed' : 'pending',
      detail: featureCount ? `${featureCount} features` : 'Pending',
    },
    {
      name: 'Model Training',
      icon: '🧠',
      status: modelType ? 'completed' : 'pending',
      detail: modelType ?? 'Not trained',
    },
    {
      name: 'Evaluation',
      icon: '📊',
      status: hasMetrics ? 'completed' : 'pending',
      detail: hasMetrics ? 'Metrics validated' : 'Not evaluated',
    },
    {
      name: 'Deployment',
      icon: '🚀',
      status: modelType ? 'completed' : 'pending',
      detail: modelType ? 'Champion serving' : 'Not deployed',
    },
    {
      name: 'Monitoring',
      icon: '🛡️',
      status: hasDrift ? 'completed' : 'active',
      detail: hasDrift ? 'Drift tracked' : 'Watching...',
    },
  ];

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">⚙️ MLOps Pipeline</h2>
        <span className="badge badge-info">AUTOMATED</span>
      </div>
      <div className="pipeline-container">
        {stages.map((stage, i) => (
          <div key={stage.name} className="pipeline-stage-wrapper">
            <div className={`pipeline-stage pipeline-${stage.status}`}>
              <div className="pipeline-icon">{stage.icon}</div>
              <div className="pipeline-name">{stage.name}</div>
              <div className="pipeline-detail">{stage.detail}</div>
            </div>
            {i < stages.length - 1 && <div className="pipeline-connector" />}
          </div>
        ))}
      </div>
    </div>
  );
}
