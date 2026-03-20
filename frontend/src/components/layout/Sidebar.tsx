import type { HealthResponse } from '../../api/mlService';
import { NAV_LINKS } from '../../config';

interface SidebarProps {
  health: HealthResponse | null;
  modelCount: number;
}

/**
 * Sidebar — Config-driven navigation.
 * Links read from config.ts, no hardcoded URLs.
 */
export function Sidebar({ health, modelCount }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">⚡</div>
        <span className="sidebar-brand-text">PHOENIX ML</span>
      </div>

      <nav className="sidebar-nav">
        {NAV_LINKS.map((link) => (
          <a
            key={link.label}
            className={`nav-item ${!link.external ? 'active' : ''}`}
            href={link.href}
            target={link.external ? '_blank' : undefined}
            rel={link.external ? 'noreferrer' : undefined}
          >
            {link.icon} {link.label}
          </a>
        ))}
      </nav>

      <div className="sidebar-status">
        <div className="sidebar-status-row">
          <span className={`status-dot ${health ? 'online' : 'offline'}`} />
          <span className="sidebar-status-label">
            {health ? 'System Online' : 'Connecting...'}
          </span>
        </div>
        <div className="sidebar-status-meta">
          {health && <span className="sidebar-version">v{health.version}</span>}
          <span className="sidebar-model-count">{modelCount} models</span>
        </div>
      </div>
    </aside>
  );
}
