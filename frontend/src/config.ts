/**
 * Phoenix ML Dashboard — Centralized Configuration
 *
 * ALL hardcoded values extracted here. Change ports, URLs,
 * labels in ONE place — every component reads from this.
 */

// ── API ──────────────────────────────────────────────────
export const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';
export const REFRESH_INTERVAL_MS = 15_000;

// ── External Services ────────────────────────────────────
export const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL ?? 'http://localhost:3001';
export const GRAFANA_DASHBOARD_UID = 'phoenix-ml-prod';

export interface ServiceDef {
  name: string;
  port: number;
  icon: string;
  healthUrl?: string;
}

export const SERVICES: ServiceDef[] = [
  { name: 'API Server',  port: 8001,  icon: '🚀', healthUrl: '/api/health' },
  { name: 'PostgreSQL',  port: 5433,  icon: '🐘' },
  { name: 'Redis',       port: 6380,  icon: '⚡' },
  { name: 'Kafka',       port: 9094,  icon: '📨' },
  { name: 'Airflow',     port: 8080,  icon: '🌊', healthUrl: 'http://localhost:8080/health' },
  { name: 'MLflow',      port: 5000,  icon: '🧪', healthUrl: 'http://localhost:5000/health' },
  { name: 'Prometheus',  port: 9091,  icon: '🔥', healthUrl: 'http://localhost:9091/-/healthy' },
  { name: 'Grafana',     port: 3001,  icon: '📈', healthUrl: `${GRAFANA_URL}/api/health` },
  { name: 'Jaeger',      port: 16686, icon: '🔍', healthUrl: 'http://localhost:16686/' },
  { name: 'MinIO',       port: 9000,  icon: '📦', healthUrl: 'http://localhost:9000/minio/health/live' },
];

// ── Sidebar Navigation ───────────────────────────────────
export interface NavLink {
  label: string;
  icon: string;
  href: string;
  external?: boolean;
}

export const NAV_LINKS: NavLink[] = [
  { label: 'Dashboard',  icon: '📊', href: '#dashboard' },
  { label: 'Grafana',    icon: '📈', href: GRAFANA_URL,                    external: true },
  { label: 'Airflow',    icon: '🌊', href: 'http://localhost:8080',        external: true },
  { label: 'MLflow',     icon: '🧪', href: 'http://localhost:5000',        external: true },
  { label: 'Jaeger',     icon: '🔍', href: 'http://localhost:16686',       external: true },
  { label: 'Prometheus', icon: '🔥', href: 'http://localhost:9091',        external: true },
  { label: 'MinIO',      icon: '📦', href: 'http://localhost:9001',        external: true },
  { label: 'Kafka UI',   icon: '📨', href: 'http://localhost:8082',        external: true },
];

// ── Model Task Types ─────────────────────────────────────
export type TaskType = 'classification' | 'regression' | 'unknown';

export function detectTaskType(metrics: Record<string, unknown> | undefined): TaskType {
  if (!metrics) return 'unknown';
  if ('accuracy' in metrics || 'f1_score' in metrics) return 'classification';
  if ('rmse' in metrics || 'mae' in metrics || 'r2' in metrics) return 'regression';
  return 'unknown';
}

/** Metric display config — label, color, formatter */
export interface MetricDef {
  key: string;
  label: string;
  color: 'blue' | 'orange' | 'green' | 'red' | 'purple';
  format: (v: number) => string;
  sub: string;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const dec = (v: number) => v.toFixed(4);

export const CLASSIFICATION_METRICS: MetricDef[] = [
  { key: 'accuracy',  label: 'Accuracy',  color: 'blue',   format: pct, sub: 'Test set evaluation' },
  { key: 'f1_score',  label: 'F1 Score',  color: 'orange', format: pct, sub: 'Harmonic mean P/R' },
  { key: 'precision', label: 'Precision', color: 'green',  format: pct, sub: 'Positive predictive value' },
  { key: 'recall',    label: 'Recall',    color: 'red',    format: pct, sub: 'True positive rate' },
];

export const REGRESSION_METRICS: MetricDef[] = [
  { key: 'rmse', label: 'RMSE',  color: 'blue',   format: dec, sub: 'Root mean squared error' },
  { key: 'mae',  label: 'MAE',   color: 'orange', format: dec, sub: 'Mean absolute error' },
  { key: 'r2',   label: 'R²',    color: 'green',  format: pct, sub: 'Coefficient of determination' },
  { key: 'mse',  label: 'MSE',   color: 'red',    format: dec, sub: 'Mean squared error' },
];

export function getMetricsForTask(taskType: TaskType): MetricDef[] {
  return taskType === 'regression' ? REGRESSION_METRICS : CLASSIFICATION_METRICS;
}

// ── Prediction Display ───────────────────────────────────
export function formatPredictionVerdict(
  result: number,
  modelId: string,
  taskType: TaskType,
): { label: string; variant: 'good' | 'bad' | 'neutral' } {
  if (taskType === 'regression') {
    return { label: `Value: ${result.toFixed(4)}`, variant: 'neutral' };
  }
  // Classification: model-specific labels
  const classLabels: Record<string, [string, string]> = {
    'credit-risk':     ['🚨 BAD CREDIT',     '✅ GOOD CREDIT'],
    'fraud-detection': ['✅ LEGITIMATE',      '🚨 FRAUD DETECTED'],
  };
  const [label0, label1] = classLabels[modelId] ?? [`Class ${result}`, `Class ${result}`];
  const label = result === 1 ? label1 : label0;
  const variant = modelId === 'fraud-detection'
    ? (result === 1 ? 'bad' : 'good')
    : (result === 1 ? 'good' : 'bad');
  return { label, variant };
}

// ── Entity Input ─────────────────────────────────────────
export function getEntityPrefix(modelId: string): string {
  const prefixes: Record<string, string> = {
    'credit-risk': 'customer',
    'fraud-detection': 'txn',
    'house-price': 'house',
  };
  return prefixes[modelId] ?? 'entity';
}
