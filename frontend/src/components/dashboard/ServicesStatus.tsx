import { ServiceCard } from '../ui/ServiceCard';

const SERVICES = [
  { name: 'API Server', port: 8001, icon: '🚀', url: 'http://localhost:8001/health' },
  { name: 'PostgreSQL', port: 5433, icon: '🐘' },
  { name: 'Redis', port: 6380, icon: '⚡' },
  { name: 'Kafka', port: 9094, icon: '📨' },
  { name: 'Prometheus', port: 9091, icon: '🔥', url: 'http://localhost:9091' },
  { name: 'Grafana', port: 3001, icon: '📈', url: 'http://localhost:3001' },
  { name: 'Jaeger', port: 16686, icon: '🔍', url: 'http://localhost:16686' },
  { name: 'MinIO', port: 9000, icon: '📦', url: 'http://localhost:9001' },
] as const;

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
