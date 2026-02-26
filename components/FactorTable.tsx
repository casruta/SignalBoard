import type { FactorExposure } from "@/lib/types";

export function FactorTable({ factors }: { factors: FactorExposure[] }) {
  return (
    <div className="signal-card p-6">
      <h3 className="text-lg font-semibold mb-4">Factor Exposure Analysis</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border)] text-[var(--color-text-dim)]">
              <th className="text-left py-2 pr-4">Factor</th>
              <th className="text-right py-2 px-4">Beta</th>
              <th className="text-right py-2 px-4">t-Stat</th>
              <th className="text-right py-2 px-4">Contribution (bps)</th>
              <th className="text-left py-2 pl-4">Significance</th>
            </tr>
          </thead>
          <tbody>
            {factors.map((f) => (
              <tr
                key={f.factor}
                className="border-b border-[var(--color-border)] hover:bg-[var(--color-surface-2)] transition-colors"
              >
                <td className="py-2 pr-4 font-medium">{f.factor}</td>
                <td className="text-right py-2 px-4 font-mono">
                  <span
                    className={
                      f.beta > 0 ? "text-green-400" : f.beta < 0 ? "text-red-400" : ""
                    }
                  >
                    {f.beta > 0 ? "+" : ""}
                    {f.beta.toFixed(3)}
                  </span>
                </td>
                <td className="text-right py-2 px-4 font-mono">
                  {f.tStat.toFixed(2)}
                </td>
                <td className="text-right py-2 px-4 font-mono">
                  <span
                    className={
                      f.contribution > 0
                        ? "text-green-400"
                        : f.contribution < 0
                          ? "text-red-400"
                          : ""
                    }
                  >
                    {f.contribution > 0 ? "+" : ""}
                    {f.contribution.toFixed(1)}
                  </span>
                </td>
                <td className="py-2 pl-4">
                  {Math.abs(f.tStat) > 2.576 ? (
                    <span className="text-green-400 text-xs">*** (p&lt;0.01)</span>
                  ) : Math.abs(f.tStat) > 1.96 ? (
                    <span className="text-yellow-400 text-xs">** (p&lt;0.05)</span>
                  ) : Math.abs(f.tStat) > 1.645 ? (
                    <span className="text-orange-400 text-xs">* (p&lt;0.10)</span>
                  ) : (
                    <span className="text-[var(--color-text-dim)] text-xs">n.s.</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
