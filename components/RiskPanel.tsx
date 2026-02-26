import type { RiskMetric } from "@/lib/types";

export function RiskPanel({ metrics }: { metrics: RiskMetric[] }) {
  return (
    <div className="signal-card p-6">
      <h3 className="text-lg font-semibold mb-4">Risk Analytics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map((m) => (
          <div
            key={m.label}
            className="p-3 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)]"
          >
            <div className="text-xs text-[var(--color-text-dim)] mb-1">
              {m.label}
            </div>
            <div className="text-lg font-bold font-mono">{m.value}</div>
            <div className="text-xs text-[var(--color-text-dim)] mt-1">
              {m.detail}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
