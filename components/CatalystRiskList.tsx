export function CatalystRiskList({
  title,
  items,
  type,
}: {
  title: string;
  items: string[];
  type: "catalyst" | "risk";
}) {
  return (
    <div className="signal-card p-6">
      <h3 className="text-lg font-semibold mb-3">{title}</h3>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-3 text-sm">
            <span
              className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${
                type === "catalyst" ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-[var(--color-text-dim)] leading-relaxed">
              {item}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
