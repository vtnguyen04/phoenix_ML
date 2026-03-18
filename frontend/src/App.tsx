import { useState, useEffect, useCallback } from 'react';
import { mlService } from './api/mlService';
import type { PredictionResponse, DriftReport, HealthResponse } from './api/mlService';
import { Sidebar } from './components/layout/Sidebar';
import { StatsGrid } from './components/dashboard/StatsGrid';
import { PredictionPanel } from './components/dashboard/PredictionPanel';
import { DriftPanel } from './components/dashboard/DriftPanel';
import { ServicesStatus } from './components/dashboard/ServicesStatus';
import './index.css';

/**
 * App — Root layout component.
 * Single Responsibility: orchestrates state and delegates rendering to children.
 */
export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [predictionCount, setPredictionCount] = useState(0);
  const [lastLatency, setLastLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [driftLoading, setDriftLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const h = await mlService.getHealth();
        setHealth(h);
        setError(null);
      } catch {
        setHealth(null);
        setError('Cannot connect to API');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handlePredict = useCallback(async (entityId: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await mlService.predict('credit-risk', entityId);
      setPrediction(res);
      setPredictionCount((c) => c + 1);
      setLastLatency(res.latency_ms);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleScanDrift = useCallback(async () => {
    setDriftLoading(true);
    setError(null);
    try {
      const res = await mlService.getDrift('credit-risk');
      setDrift(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Drift scan failed');
    } finally {
      setDriftLoading(false);
    }
  }, []);

  return (
    <div className="app-layout">
      <Sidebar health={health} />

      <main className="main-content">
        <header className="main-header">
          <h1>Production Dashboard</h1>
          {error && <span className="badge badge-danger">{error}</span>}
        </header>

        <StatsGrid
          predictionCount={predictionCount}
          lastLatency={lastLatency}
          modelAccuracy="78.5%"
        />

        <div className="grid-2">
          <PredictionPanel
            onPredict={handlePredict}
            prediction={prediction}
            loading={loading}
          />
          <DriftPanel
            drift={drift}
            onScan={handleScanDrift}
            loading={driftLoading}
          />
        </div>

        <ServicesStatus />
      </main>
    </div>
  );
}