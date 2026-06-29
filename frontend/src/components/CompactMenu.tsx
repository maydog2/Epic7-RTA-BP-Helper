type CompactMenuOption<T extends string> = {
  value: T;
  label: string;
  iconUrl?: string;
};

export function CompactMenu<T extends string>(props: {
  ariaLabel: string;
  value: T;
  options: CompactMenuOption<T>[];
  open: boolean;
  buttonClassName?: string;
  menuClassName?: string;
  iconOnly?: boolean;
  onToggle: () => void;
  onSelect: (value: T) => void;
}) {
  const selectedOption = props.options.find((option) => option.value === props.value);
  return (
    <>
      <button
        type="button"
        className={props.buttonClassName ?? "compact-menu-button"}
        aria-haspopup="menu"
        aria-expanded={props.open}
        aria-label={props.ariaLabel}
        onClick={props.onToggle}
      >
        <span>
          {selectedOption?.iconUrl ? (
            <img
              className="compact-menu-icon"
              src={selectedOption.iconUrl}
              alt={props.iconOnly ? selectedOption.label : ""}
              aria-hidden={props.iconOnly ? undefined : "true"}
            />
          ) : null}
          {props.iconOnly && selectedOption?.iconUrl ? null : (selectedOption?.label ?? props.value)}
        </span>
        <span aria-hidden="true">▾</span>
      </button>
      {props.open && (
        <div className={props.menuClassName ?? "compact-menu"} role="menu">
          {props.options.map((option) => (
            <button
              key={option.value}
              type="button"
              role="menuitemradio"
              aria-checked={props.value === option.value}
              className={props.value === option.value ? "active" : ""}
              title={props.iconOnly ? option.label : undefined}
              onClick={() => props.onSelect(option.value)}
            >
              {option.iconUrl ? (
                <img
                  className="compact-menu-icon"
                  src={option.iconUrl}
                  alt={props.iconOnly ? option.label : ""}
                  aria-hidden={props.iconOnly ? undefined : "true"}
                />
              ) : null}
              {props.iconOnly && option.iconUrl ? null : option.label}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
