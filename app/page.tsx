import Link from "next/link";
import { SignalBadge } from "@/components/SignalBadge";
import { loadAllReportsSync } from "@/lib/reports";
import { TICKER_SYMBOLS } from "@/lib/tickers";

export default function DashboardPage() {
  const reports = loadAllReportsSync();

  return (
    <div>
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-2">ML Research Dashboard</h2>
        <p className="text-[var(--color-text-dim)] text-sm">
          Quantitative signal analysis powered by ensemble ML models.
          Click any ticker for the full research report.
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="signal-card p-4">
          <div className="text-xs text-[var(--color-text-dim)] mb-1">
            Tickers Covered
          </div>
          <div className="text-2xl font-bold">{TICKER_SYMBOLS.length}</div>
        </div>
        <div className="signal-card p-4">
          <div className="text-xs text-[var(--color-text-dim)] mb-1">
            Bullish Signals
          </div>
          <div className="text-2xl font-bold text-green-400">
            {Object.values(reports).filter((r) => r.overallSignal === "bullish").length}
          </div>
        </div>
        <div className="signal-card p-4">
          <div className="text-xs text-[var(--color-text-dim)] mb-1">
            Neutral Signals
          </div>
          <div className="text-2xl font-bold text-yellow-400">
            {Object.values(reports).filter((r) => r.overallSignal === "neutral").length}
          </div>
        </div>
        <div className="signal-card p-4">
          <div className="text-xs text-[var(--color-text-dim)] mb-1">
            Bearish Signals
          </div>
          <div className="text-2xl font-bold text-red-400">
            {Object.values(reports).filter((r) => r.overallSignal === "bearish").length}
          </div>
        </div>
      </div>

      {/* Ticker Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {TICKER_SYMBOLS.map((symbol) => {
          const report = reports[symbol];
          if (!report) return null;
          const changeColor =
            report.priceChangePct > 0
              ? "text-green-400"
              : report.priceChangePct < 0
                ? "text-red-400"
                : "text-[var(--color-text-dim)]";

          return (
            <Link
              key={symbol}
              href={`/ticker/${symbol}`}
              className="signal-card p-6 block cursor-pointer"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-xl font-bold">{report.symbol}</div>
                  <div className="text-sm text-[var(--color-text-dim)]">
                    {report.companyName}
                  </div>
                </div>
                <SignalBadge direction={report.overallSignal} />
              </div>

              <div className="flex items-baseline gap-3 mb-4">
                <span className="text-2xl font-bold font-mono">
                  ${report.lastPrice.toFixed(2)}
                </span>
                <span className={`text-sm font-mono ${changeColor}`}>
                  {report.priceChangePct > 0 ? "+" : ""}
                  {report.priceChangePct.toFixed(2)}%
                </span>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-dim)]">
                    Composite Score
                  </span>
                  <span className="font-mono">
                    {report.compositeScore > 0 ? "+" : ""}
                    {report.compositeScore.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-dim)]">
                    Model Confidence
                  </span>
                  <span className="font-mono">
                    {(report.modelConfidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-dim)]">Sector</span>
                  <span>{report.sector}</span>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-[var(--color-border)] text-xs text-[var(--color-text-dim)]">
                {report.signals.length} ML signals | Updated{" "}
                {report.lastUpdated}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
