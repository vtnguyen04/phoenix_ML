interface ServiceCardProps {
  name: string;
  port: number;
  icon: string;
}

/**
 * ServiceCard — Displays a single infrastructure service status.
 * SRP: Only renders one service's name, port, and icon.
 */
export function ServiceCard({ name, port, icon }: ServiceCardProps) {
  return (
    <div style={{
      padding: '12px 16px',
      background: 'var(--bg-glass)',
      borderRadius: 'var(--radius-sm)',
      border: '1px solid var(--border-default)',
      display: 'flex',
      alignItems: 'center',
      gap: 10,
    }}>
      <span style={{ fontSize: 20 }}>{icon}</span>
      <div>
        <div style={{ fontSize: 12, fontWeight: 600 }}>{name}</div>
        <div style={{
          fontSize: 10,
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}>
          :{port}
        </div>
      </div>
    </div>
  );
}
