import { useState, useEffect, useCallback } from 'react';
import { mlService } from './api/mlService';
import type { PredictionResponse, DriftReport, HealthResponse, ModelInfo } from './api/mlService';
import { Sidebar } from './components/layout/Sidebar';
import { StatsGrid } from './components/dashboard/StatsGrid';
import { ModelInfoCard } from './components/dashboard/ModelInfoCard';
import { GrafanaEmbed } from './components/dashboard/GrafanaEmbed';
import { PipelineStatus } from './components/dashboard/PipelineStatus';
import { PredictionPanel } from './components/dashboard/PredictionPanel';
import { DriftPanel } from './components/dashboard/DriftPanel';
import { ServicesStatus } from './components/dashboard/ServicesStatus';
import './index.css';

const REFRESH_INTERVAL = 15_000;

export default function App() {
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [driftReports, setDriftReports] = useState<DriftReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch available models on mount
  useEffect(() => {
    mlService.getModels().then((models) => {
      setAvailableModels(models);
      if (models.length > 0 && !selectedModelId) {
        setSelectedModelId(models[0].model_id);
      }
    }).catch(() => {
      setError('Cannot fetch model list');
    });
  }, [selectedModelId]);

  // Clear state when switching models
  useEffect(() => {
    if (!selectedModelId) return;
    setModel(null);
    setPrediction(null);
    setDrift(null);
    setDriftReports([]);
    setError(null);
    setDriftError(null);
  }, [selectedModelId]);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [h, m] = await Promise.all([
          mlService.getHealth(),
          mlService.getModel(selectedModelId),
        ]);
        setHealth(h);
        setModel(m);
        setError(null);
      } catch {
        setError(`Cannot fetch data for ${selectedModelId}`);
      }

      try {
        const reports = await mlService.getDriftReports(selectedModelId);
        setDriftReports(reports);
        if (reports.length > 0) {
          setDrift(reports[0]);
        }
      } catch {
        // Drift reports may not exist yet — that's OK
      }
    };

    fetchInitialData();
    const interval = setInterval(fetchInitialData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [selectedModelId]);

  const handlePredict = useCallback(async (entityId: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await mlService.predict(selectedModelId, entityId);
      setPrediction(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, [selectedModelId]);

  const handleScanDrift = useCallback(async () => {
    setDriftLoading(true);
    setDriftError(null);
    try {
      const res = await mlService.getDrift(selectedModelId);
      setDrift(res);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Drift scan failed';
      if (msg.includes('400') || msg.toLowerCase().includes('not enough data')) {
        setDriftError('Not enough prediction data yet. The monitoring loop will populate data automatically.');
      } else {
        setDriftError(msg);
      }
    } finally {
      setDriftLoading(false);
    }
  }, [selectedModelId]);

  const metrics = model?.metadata.metrics;

  return (
    <div className="app-layout">
      <Sidebar health={health} />

      <main className="main-content">
        <header className="main-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <h1>Production Dashboard</h1>
            <select 
              value={selectedModelId} 
              onChange={(e) => setSelectedModelId(e.target.value)}
              style={{ padding: '8px', borderRadius: '4px', background: '#1a1f36', color: '#fff', border: '1px solid #2a2f4c' }}
            >
              {availableModels.map(m => (
                <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
              ))}
            </select>
          </div>
          <div className="header-badges">
            {error && <span className="badge badge-danger">{error}</span>}
            <span className="badge badge-info">Auto-refresh: {REFRESH_INTERVAL / 1000}s</span>
          </div>
        </header>

        <StatsGrid
          accuracy={metrics?.accuracy ?? null}
          f1Score={metrics?.f1_score ?? null}
          precision={metrics?.precision ?? null}
          recall={metrics?.recall ?? null}
        />

        <div className="grid-2">
          <ModelInfoCard model={model} />
          <PipelineStatus
            modelType={metrics?.model_type}
            dataset={model?.metadata.dataset}
            featureCount={metrics?.n_features}
            hasMetrics={!!metrics}
            hasDrift={driftReports.length > 0}
          />
        </div>

        <GrafanaEmbed />

        <div className="grid-2">
          <DriftPanel
            drift={drift}
            onScan={handleScanDrift}
            loading={driftLoading}
            error={driftError}
          />
          <PredictionPanel
            onPredict={handlePredict}
            prediction={prediction}
            loading={loading}
          />
        </div>

        <ServicesStatus />
      </main>
    </div>
  );
}