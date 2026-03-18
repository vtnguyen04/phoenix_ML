import { ServiceCard } from '../ui/ServiceCard';

/** Service metadata for infrastructure display. */
const SERVICES = [
  { name: 'API Server', port: 8001, icon: '🚀' },
  { name: 'PostgreSQL', port: 5433, icon: '🐘' },
  { name: 'Redis', port: 6380, icon: '⚡' },
  { name: 'Kafka', port: 9094, icon: '📨' },
  { name: 'Prometheus', port: 9091, icon: '🔥' },
  { name: 'Grafana', port: 3001, icon: '📈' },
  { name: 'MinIO', port: 9000, icon: '📦' },
] as const;

/**
 * ServicesStatus — Infrastructure service overview grid.
 * SRP: Only renders service cards. Data is static configuration.
 * OCP: Add new services by extending the SERVICES array.
 */
export function ServicesStatus() {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">🏗️ Infrastructure</h2>
        <span className="badge badge-success">ALL ONLINE</span>
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: 12,
      }}>
        {SERVICES.map((svc) => (
          <ServiceCard key={svc.name} {...svc} />
        ))}
      </div>
    </div>
  );
}
