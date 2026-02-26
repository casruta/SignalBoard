/** The 6 tickers derived from ML signal research */
export const TICKER_SYMBOLS = [
  "NVDA",
  "AAPL",
  "MSFT",
  "GOOGL",
  "AMZN",
  "TSLA",
] as const;

export type TickerSymbol = (typeof TICKER_SYMBOLS)[number];
