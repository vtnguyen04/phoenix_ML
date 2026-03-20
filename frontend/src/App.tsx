import { useState, useEffect, useCallback } from 'react';
import { mlService } from './api/mlService';
import type { PredictionResponse, DriftReport, HealthResponse, ModelInfo } from './api/mlService';
import { Sidebar } from './components/layout/Sidebar';
import { StatsGrid } from './components/dashboard/StatsGrid';
import { ModelInfoCard } from './components/dashboard/ModelInfoCard';
import { GrafanaEmbed } from './components/dashboard/GrafanaEmbed';
import { ModelComparison } from './components/dashboard/ModelComparison';
import { PredictionPanel } from './components/dashboard/PredictionPanel';
import { DriftPanel } from './components/dashboard/DriftPanel';
import { PerformancePanel } from './components/dashboard/PerformancePanel';
import { ServicesStatus } from './components/dashboard/ServicesStatus';
import { REFRESH_INTERVAL_MS, detectTaskType } from './config';
import './index.css';

export default function App() {
  const [allModels, setAllModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Deduplicated unique model IDs (champion only)
  const uniqueModels = allModels.reduce<ModelInfo[]>((acc, m) => {
    if (!acc.find((x) => x.model_id === m.model_id)) {
      // Prefer champion version
      const champion = allModels.find(
        (x) => x.model_id === m.model_id && x.metadata.role === 'champion',
      );
      acc.push(champion ?? m);
    }
    return acc;
  }, []);

  // Challengers for the selected model
  const challengers = allModels.filter(
    (m) => m.model_id === selectedModelId && m.metadata.role === 'challenger',
  );

  // Fetch models on mount
  useEffect(() => {
    mlService.getModels().then((models) => {
      setAllModels(models);
      if (models.length > 0 && !selectedModelId) {
        // Pick first champion
        const firstChampion = models.find((m) => m.metadata.role === 'champion');
        setSelectedModelId(firstChampion?.model_id ?? models[0].model_id);
      }
    }).catch(() => setError('Cannot fetch model list'));
  }, [selectedModelId]);

  // Reset state on model switch
  useEffect(() => {
    if (!selectedModelId) return;
    setPrediction(null);
    setDrift(null);
    setError(null);
    setDriftError(null);
  }, [selectedModelId]);

  // Fetch model data + health
  useEffect(() => {
    if (!selectedModelId) return;

    const fetchData = async () => {
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
        if (reports.length > 0) setDrift(reports[0]);
      } catch {
        /* Drift may not exist yet */
      }
    };

    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [selectedModelId]);

  const handlePredict = useCallback(async (features: number[]) => {
    setLoading(true);
    setError(null);
    try {
      const res = await mlService.predictWithFeatures(selectedModelId, features);
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
      setDriftError(msg.includes('400') ? 'Not enough data yet.' : msg);
    } finally {
      setDriftLoading(false);
    }
  }, [selectedModelId]);

  const metrics = model?.metadata.metrics;
  const taskType = detectTaskType(metrics as Record<string, unknown> | undefined);
  const featureNames: string[] =
    metrics?.all_features as string[] ?? model?.metadata.features ?? [];

  return (
    <div className="app-layout">
      <Sidebar health={health} modelCount={uniqueModels.length} />

      <main className="main-content">
        <header className="main-header">
          <div className="header-left">
            <h1>Dashboard</h1>
            <select
              className="model-selector"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              {uniqueModels.map((m) => (
                <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
              ))}
            </select>
          </div>
          {error && <span className="badge badge-danger">{error}</span>}
        </header>

        <StatsGrid metrics={metrics as Record<string, number> | undefined} taskType={taskType} />

        <div className="grid-2">
          <ModelInfoCard model={model} />
          <ModelComparison
            champion={model}
            challengers={challengers}
            taskType={taskType}
          />
        </div>

        <div className="grid-3">
          <PerformancePanel modelId={selectedModelId} />
          <DriftPanel
            drift={drift}
            onScan={handleScanDrift}
            loading={driftLoading}
            error={driftError}
          />
          <PredictionPanel
            modelId={selectedModelId}
            taskType={taskType}
            featureNames={featureNames}
            onPredict={handlePredict}
            prediction={prediction}
            loading={loading}
          />
        </div>

        <GrafanaEmbed />
        <ServicesStatus />
      </main>
    </div>
  );
}