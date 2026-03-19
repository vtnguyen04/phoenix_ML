import type { HealthResponse } from '../../api/mlService';

interface SidebarProps {
  health: HealthResponse | null;
}

export function Sidebar({ health }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">⚡</div>
        <span className="sidebar-brand-text">PHOENIX ML</span>
      </div>

      <nav className="sidebar-nav">
        <a className="nav-item active" href="#dashboard">📊 Dashboard</a>
        <a className="nav-item" href="http://localhost:3001" target="_blank" rel="noreferrer">
          📈 Grafana
        </a>
        <a className="nav-item" href="http://localhost:16686" target="_blank" rel="noreferrer">
          🔍 Jaeger
        </a>
        <a className="nav-item" href="http://localhost:9091" target="_blank" rel="noreferrer">
          🔥 Prometheus
        </a>
        <a className="nav-item" href="http://localhost:9001" target="_blank" rel="noreferrer">
          📦 MinIO
        </a>
      </nav>

      <div className="sidebar-status">
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
          <span className={`status-dot ${health ? 'online' : 'offline'}`} />
          <span style={{ fontSize: 13, fontWeight: 600 }}>
            {health ? 'System Online' : 'Connecting...'}
          </span>
        </div>
        {health && (
          <span style={{
            fontSize: 11,
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
          }}>
            v{health.version}
          </span>
        )}
      </div>
    </aside>
  );
}
