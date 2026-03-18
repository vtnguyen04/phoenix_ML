import type { PredictionResponse } from '../../api/mlService';

interface PredictionResultProps {
  prediction: PredictionResponse;
}

/**
 * PredictionResult — Displays a single prediction outcome.
 * SRP: Only renders prediction data visualization.
 * DIP: Depends on PredictionResponse interface, not concrete API.
 */
export function PredictionResult({ prediction }: PredictionResultProps) {
  const isGoodCredit = prediction.result === 1;

  return (
    <div className="prediction-result">
      <div className="prediction-grid">
        <div className="prediction-field">
          <label>Model Version</label>
          <span className="value" style={{ color: 'var(--accent-orange)' }}>
            {prediction.version}
          </span>
        </div>
        <div className="prediction-field">
          <label>Confidence</label>
          <span className="value" style={{ color: 'var(--accent-blue)' }}>
            {(prediction.confidence.value * 100).toFixed(1)}%
          </span>
        </div>
        <div className="prediction-field">
          <label>Latency</label>
          <span className="value" style={{ color: 'var(--accent-green)' }}>
            {prediction.latency_ms.toFixed(2)}ms
          </span>
        </div>
      </div>
      <div className="prediction-verdict">
        <div className={`verdict-text ${isGoodCredit ? 'good' : 'bad'}`}>
          {isGoodCredit ? '✅ GOOD CREDIT' : '🚨 BAD CREDIT'}
        </div>
      </div>
    </div>
  );
}
