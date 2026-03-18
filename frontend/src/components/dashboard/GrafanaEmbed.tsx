interface GrafanaEmbedProps {
  dashboardUid?: string;
  grafanaUrl?: string;
  height?: number;
}

export function GrafanaEmbed({
  dashboardUid = 'phoenix-ml-prod',
  grafanaUrl = 'http://localhost:3001',
  height = 350,
}: GrafanaEmbedProps) {
  const iframeSrc = `${grafanaUrl}/d/${dashboardUid}?orgId=1&theme=dark&kiosk`;

  return (
    <div className="card grafana-card">
      <div className="card-header">
        <h2 className="card-title">📈 Live Metrics — Grafana</h2>
        <a
          href={`${grafanaUrl}/d/${dashboardUid}`}
          target="_blank"
          rel="noreferrer"
          className="btn btn-sm"
        >
          Open Full ↗
        </a>
      </div>
      <div className="grafana-embed-container">
        <iframe
          src={iframeSrc}
          width="100%"
          height={height}
          style={{ border: 'none', borderRadius: 'var(--radius-sm)' }}
          title="Grafana Dashboard"
          loading="lazy"
        />
      </div>
    </div>
  );
}
