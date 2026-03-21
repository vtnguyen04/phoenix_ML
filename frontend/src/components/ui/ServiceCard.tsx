import { useEffect, useState } from 'react';

interface ServiceCardProps {
  name: string;
  port: number;
  icon: string;
  healthUrl?: string;
}

/**
 * ServiceCard — Shows service name, port and live health status.
 * Health is checked via fetch (proxy or direct) every 15 s.
 */
export function ServiceCard({ name, port, icon, healthUrl }: ServiceCardProps) {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>(healthUrl ? 'checking' : 'online');

  useEffect(() => {
    if (!healthUrl) return;

    const check = async () => {
      try {
        const res = await fetch(healthUrl, { mode: 'no-cors', signal: AbortSignal.timeout(3000) });
        // mode:'no-cors' returns opaque response (status 0) but no error = reachable
        setStatus(res.ok || res.type === 'opaque' ? 'online' : 'offline');
      } catch {
        setStatus('offline');
      }
    };

    check();
    const id = setInterval(check, 15_000);
    return () => clearInterval(id);
  }, [healthUrl]);

  const dot = status === 'online' ? '🟢' : status === 'offline' ? '🔴' : '🟡';

  return (
    <div className={`service-card ${status === 'online' ? 'service-online' : status === 'offline' ? 'service-offline' : ''}`}>
      <span className="service-card-icon">{icon}</span>
      <div>
        <div className="service-card-name">{name} {dot}</div>
        <div className="service-card-port">:{port}</div>
      </div>
    </div>
  );
}
