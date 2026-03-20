import type { PredictionResponse } from '../../api/mlService';
import { formatPredictionVerdict } from '../../config';
import type { TaskType } from '../../config';

interface PredictionResultProps {
  prediction: PredictionResponse;
  modelId: string;
  taskType: TaskType;
}

/**
 * PredictionResult — Model-agnostic prediction display.
 * Uses config-driven verdict formatting per model + task type.
 */
export function PredictionResult({ prediction, modelId, taskType }: PredictionResultProps) {
  const verdict = formatPredictionVerdict(prediction.result, modelId, taskType);

  return (
    <div className="prediction-result">
      <div className="prediction-grid">
        <div className="prediction-field">
          <label>Model Version</label>
          <span className="value color-orange">{prediction.version}</span>
        </div>
        <div className="prediction-field">
          <label>Confidence</label>
          <span className="value color-blue">
            {(prediction.confidence.value * 100).toFixed(1)}%
          </span>
        </div>
        <div className="prediction-field">
          <label>Latency</label>
          <span className="value color-green">
            {prediction.latency_ms.toFixed(2)}ms
          </span>
        </div>
      </div>
      <div className="prediction-verdict">
        <div className={`verdict-text ${verdict.variant === 'good' ? 'good' : verdict.variant === 'bad' ? 'bad' : 'neutral'}`}>
          {verdict.label}
        </div>
      </div>
    </div>
  );
}
