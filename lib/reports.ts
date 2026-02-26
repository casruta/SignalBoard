import type { TickerReport } from "./types";
import { TICKER_SYMBOLS } from "./tickers";

const reports: Record<string, TickerReport> = {};

// Dynamic imports for each ticker report
async function loadReport(symbol: string): Promise<TickerReport> {
  const mod = await import(`@/data/${symbol.toLowerCase()}`);
  return mod.default;
}

export function getReport(symbol: string): TickerReport {
  return reports[symbol.toUpperCase()];
}

export function getAllReports(): TickerReport[] {
  return TICKER_SYMBOLS.map((s) => reports[s]).filter(Boolean);
}

// Synchronous loading via require for server components
export function loadAllReportsSync(): Record<string, TickerReport> {
  for (const symbol of TICKER_SYMBOLS) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const mod = require(`@/data/${symbol.toLowerCase()}`);
      reports[symbol] = mod.default;
    } catch {
      // Report not yet generated
    }
  }
  return reports;
}
