import type { HealthResponse } from '../../api/mlService';

interface SidebarProps {
  health: HealthResponse | null;
}

/**
 * Sidebar — Navigation and system status.
 * SRP: Only handles navigation rendering and health status display.
 */
export function Sidebar({ health }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">⚡</div>
        <span className="sidebar-brand-text">PHOENIX ML</span>
      </div>

      <nav className="sidebar-nav">
        <a className="nav-item active" href="#dashboard">📊 Dashboard</a>
        <a className="nav-item" href="#inference">🎯 Inference</a>
        <a className="nav-item" href="#monitoring">🛡️ Monitoring</a>
        <a className="nav-item" href="http://localhost:3001" target="_blank" rel="noreferrer">
          📈 Grafana
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
