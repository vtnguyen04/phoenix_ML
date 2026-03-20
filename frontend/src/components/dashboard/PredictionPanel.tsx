import { useState } from 'react';
import type { PredictionResponse } from '../../api/mlService';
import type { TaskType } from '../../config';
import { formatPredictionVerdict } from '../../config';
import { Spinner } from '../ui/Spinner';

interface PredictionPanelProps {
  modelId: string;
  taskType: TaskType;
  featureNames: string[];
  onPredict: (features: number[]) => void;
  prediction: PredictionResponse | null;
  loading: boolean;
}

export function PredictionPanel({
  modelId,
  taskType,
  featureNames,
  onPredict,
  prediction,
  loading,
}: PredictionPanelProps) {
  const [features, setFeatures] = useState<Record<string, string>>({});

  const handleChange = (name: string, value: string) => {
    setFeatures((prev) => ({ ...prev, [name]: value }));
  };

  const handlePredict = () => {
    const values = featureNames.map((name) => {
      const v = features[name];
      return v ? parseFloat(v) : 0;
    });
    onPredict(values);
  };

  const filledCount = featureNames.filter((n) => features[n] && features[n].trim() !== '').length;

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Inference</h2>
        <span className="badge badge-info">{featureNames.length} features</span>
      </div>

      {featureNames.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-text">No feature schema available for this model.</div>
        </div>
      ) : (
        <>
          <div className="feature-form">
            {featureNames.map((name) => (
              <div key={name} className="feature-field">
                <label htmlFor={`feat-${name}`}>{name}</label>
                <input
                  id={`feat-${name}`}
                  type="number"
                  step="any"
                  placeholder="0"
                  value={features[name] ?? ''}
                  onChange={(e) => handleChange(name, e.target.value)}
                />
              </div>
            ))}
          </div>

          <div className="predict-actions">
            <span className="feature-count">{filledCount}/{featureNames.length} filled</span>
            <button
              className="btn btn-primary"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? <Spinner /> : null}
              Run Prediction
            </button>
          </div>
        </>
      )}

      {prediction && (
        <PredictionResultInline
          prediction={prediction}
          modelId={modelId}
          taskType={taskType}
        />
      )}
    </div>
  );
}

function PredictionResultInline({
  prediction,
  modelId,
  taskType,
}: {
  prediction: PredictionResponse;
  modelId: string;
  taskType: TaskType;
}) {
  const verdict = formatPredictionVerdict(prediction.result, modelId, taskType);

  return (
    <div className="prediction-result">
      <div className="prediction-grid">
        <div className="prediction-field">
          <label>Result</label>
          <span className={`value ${verdict.variant === 'good' ? 'color-green' : verdict.variant === 'bad' ? 'color-red' : 'color-blue'}`}>
            {verdict.label}
          </span>
        </div>
        <div className="prediction-field">
          <label>Confidence</label>
          <span className="value">{(prediction.confidence.value * 100).toFixed(1)}%</span>
        </div>
        <div className="prediction-field">
          <label>Latency</label>
          <span className="value">{prediction.latency_ms.toFixed(1)}ms</span>
        </div>
      </div>
    </div>
  );
}
