interface EntitySelectorProps {
  prefix: string;
  onSelect: (id: string) => void;
  selected: string;
  count?: number;
}

/**
 * EntitySelector — Generic entity ID picker.
 * Prefix adapts per model: "customer", "txn", "house", etc.
 */
export function EntitySelector({
  prefix,
  onSelect,
  selected,
  count = 10,
}: EntitySelectorProps) {
  const entityIds = Array.from(
    { length: count },
    (_, i) => `${prefix}-${String(i + 1).padStart(4, '0')}`,
  );

  return (
    <div className="customer-grid">
      {entityIds.map((id) => (
        <button
          key={id}
          className={`customer-btn ${selected === id ? 'active' : ''}`}
          onClick={() => onSelect(id)}
          aria-pressed={selected === id}
        >
          {id.replace(`${prefix}-`, '#')}
        </button>
      ))}
    </div>
  );
}
