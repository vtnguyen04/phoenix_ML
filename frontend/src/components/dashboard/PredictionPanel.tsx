import { useState } from 'react';
import type { PredictionResponse } from '../../api/mlService';
import { StatusBadge } from '../ui/StatusBadge';
import { Spinner } from '../ui/Spinner';
import { EntitySelector } from '../ui/EntitySelector';
import { PredictionResult } from '../ui/PredictionResult';
import { getEntityPrefix } from '../../config';
import type { TaskType } from '../../config';

interface PredictionPanelProps {
  modelId: string;
  taskType: TaskType;
  onPredict: (entityId: string) => void;
  prediction: PredictionResponse | null;
  loading: boolean;
}

/**
 * PredictionPanel — Model-agnostic inference panel.
 * Adapts entity input prefix and button label based on model.
 */
export function PredictionPanel({
  modelId,
  taskType,
  onPredict,
  prediction,
  loading,
}: PredictionPanelProps) {
  const prefix = getEntityPrefix(modelId);
  const [selectedEntity, setSelectedEntity] = useState(`${prefix}-0001`);

  const handlePredict = () => onPredict(selectedEntity);

  const label = modelId
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">🎯 Real-time Inference</h2>
        <StatusBadge variant="info">LIVE</StatusBadge>
      </div>

      <EntitySelector
        prefix={prefix}
        onSelect={setSelectedEntity}
        selected={selectedEntity}
        count={10}
      />

      <button
        className="btn btn-primary btn-full"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? <Spinner /> : '⚡'} Predict {label}
      </button>

      {prediction && (
        <div className="prediction-result-wrapper">
          <PredictionResult
            prediction={prediction}
            modelId={modelId}
            taskType={taskType}
          />
        </div>
      )}
    </div>
  );
}
