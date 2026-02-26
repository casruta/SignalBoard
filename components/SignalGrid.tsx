import type { MLSignal } from "@/lib/types";
import { SignalBadge } from "./SignalBadge";

export function SignalGrid({ signals }: { signals: MLSignal[] }) {
  return (
    <div className="signal-card p-6">
      <h3 className="text-lg font-semibold mb-4">ML Signal Decomposition</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {signals.map((s) => (
          <div
            key={s.name}
            className="p-4 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)]"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-sm">{s.name}</span>
              <SignalBadge direction={s.direction} size="sm" />
            </div>
            <div className="flex items-center gap-4 mb-2">
              <div className="flex-1">
                <div className="h-2 rounded-full bg-[var(--color-bg)] overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${((s.value + 1) / 2) * 100}%`,
                      background:
                        s.value > 0.2
                          ? "var(--color-bullish)"
                          : s.value < -0.2
                            ? "var(--color-bearish)"
                            : "var(--color-neutral)",
                    }}
                  />
                </div>
              </div>
              <span className="font-mono text-sm w-12 text-right">
                {s.value > 0 ? "+" : ""}
                {s.value.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs text-[var(--color-text-dim)] mb-1">
              <span>Confidence: {(s.confidence * 100).toFixed(0)}%</span>
            </div>
            <p className="text-xs text-[var(--color-text-dim)] leading-relaxed mt-2">
              {s.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
