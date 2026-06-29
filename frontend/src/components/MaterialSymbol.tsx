export function MaterialSymbol(props: { name: string; className?: string }) {
  return (
    <span
      className={`material-symbols-outlined${props.className ? ` ${props.className}` : ""}`}
      aria-hidden="true"
    >
      {props.name}
    </span>
  );
}
