export type SignalDirection = "bullish" | "bearish" | "neutral";

export interface MLSignal {
  name: string;
  value: number;
  direction: SignalDirection;
  confidence: number;
  description: string;
}

export interface PricePoint {
  date: string;
  close: number;
  predicted: number;
  volume: number;
}

export interface FactorExposure {
  factor: string;
  beta: number;
  tStat: number;
  contribution: number;
}

export interface RiskMetric {
  label: string;
  value: string;
  detail: string;
}

export interface TickerReport {
  symbol: string;
  companyName: string;
  sector: string;
  industry: string;
  marketCap: string;
  lastPrice: number;
  priceChange: number;
  priceChangePct: number;
  overallSignal: SignalDirection;
  compositeScore: number;
  modelConfidence: number;
  lastUpdated: string;

  executiveSummary: string;
  modelMethodology: string;

  signals: MLSignal[];
  priceHistory: PricePoint[];
  factorExposures: FactorExposure[];
  riskMetrics: RiskMetric[];

  technicalAnalysis: string;
  fundamentalAnalysis: string;
  sentimentAnalysis: string;
  catalysts: string[];
  risks: string[];
  peerComparison: string;
  mlModelDetails: string;
}
