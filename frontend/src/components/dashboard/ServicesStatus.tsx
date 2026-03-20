import { SERVICES } from '../../config';
import { ServiceCard } from '../ui/ServiceCard';

/**
 * ServicesStatus — Config-driven infrastructure status grid.
 * Service list read from config.ts — zero hardcoded ports.
 */
export function ServicesStatus() {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Infrastructure</h2>
        <span className="badge badge-success">{SERVICES.length} SERVICES</span>
      </div>
      <div className="services-grid">
        {SERVICES.map((svc) => (
          <ServiceCard key={svc.name} name={svc.name} port={svc.port} icon={svc.icon} />
        ))}
      </div>
    </div>
  );
}
