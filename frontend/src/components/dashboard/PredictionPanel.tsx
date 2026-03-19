import { useState } from 'react';
import type { PredictionResponse } from '../../api/mlService';
import { StatusBadge } from '../ui/StatusBadge';
import { Spinner } from '../ui/Spinner';
import { CustomerSelector } from '../ui/CustomerSelector';
import { PredictionResult } from '../ui/PredictionResult';

interface PredictionPanelProps {
  onPredict: (entityId: string) => void;
  prediction: PredictionResponse | null;
  loading: boolean;
}

/**
 * PredictionPanel — Interactive credit risk inference panel.
 * SRP: Composes customer selection + predict action + result display.
 * DIP: Depends on callback interface (onPredict), not concrete service.
 */
export function PredictionPanel({ onPredict, prediction, loading }: PredictionPanelProps) {
  const [selectedCustomer, setSelectedCustomer] = useState('customer-0001');

  const handlePredict = () => onPredict(selectedCustomer);

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">🎯 Real-time Inference</h2>
        <StatusBadge variant="info">LIVE</StatusBadge>
      </div>

      <CustomerSelector
        onSelect={setSelectedCustomer}
        selected={selectedCustomer}
        count={10}
      />

      <button
        className="btn btn-primary"
        onClick={handlePredict}
        disabled={loading}
        style={{ width: '100%', justifyContent: 'center' }}
      >
        {loading ? <Spinner /> : '⚡'} Predict Credit Risk
      </button>

      {prediction && (
        <div style={{ marginTop: 20 }}>
          <PredictionResult prediction={prediction} />
        </div>
      )}
    </div>
  );
}
