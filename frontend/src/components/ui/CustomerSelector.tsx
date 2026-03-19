

interface CustomerSelectorProps {
  onSelect: (id: string) => void;
  selected: string;
  count?: number;
}

/**
 * CustomerSelector — Grid of customer ID buttons.
 * SRP: Only handles customer selection UI.
 * OCP: Configurable via `count` prop without modifying internals.
 */
export function CustomerSelector({ onSelect, selected, count = 10 }: CustomerSelectorProps) {
  const customerIds = Array.from(
    { length: count },
    (_, i) => `customer-${String(i).padStart(4, '0')}`,
  );

  return (
    <div className="customer-grid">
      {customerIds.map((id) => (
        <button
          key={id}
          className={`customer-btn ${selected === id ? 'active' : ''}`}
          onClick={() => onSelect(id)}
          aria-pressed={selected === id}
        >
          {id.replace('customer-', '#')}
        </button>
      ))}
    </div>
  );
}
