interface ServiceCardProps {
  name: string;
  port: number;
  icon: string;
}

/**
 * ServiceCard — Single infrastructure service display.
 * All inline styles replaced with CSS classes.
 */
export function ServiceCard({ name, port, icon }: ServiceCardProps) {
  return (
    <div className="service-card">
      <span className="service-card-icon">{icon}</span>
      <div>
        <div className="service-card-name">{name}</div>
        <div className="service-card-port">:{port}</div>
      </div>
    </div>
  );
}
