import Foundation

// MARK: - Signal List Response

struct SignalListResponse: Codable {
    let signals: [SignalSummary]
    let count: Int
}

struct SignalSummary: Codable, Identifiable {
    var id: String { ticker }
    let ticker: String
    let shortName: String
    let action: String  // "BUY" or "SELL"
    let confidence: Double
    let predictedReturn5d: Double
    let sector: String
    let generatedAt: String

    enum CodingKeys: String, CodingKey {
        case ticker
        case shortName = "short_name"
        case action
        case confidence
        case predictedReturn5d = "predicted_return_5d"
        case sector
        case generatedAt = "generated_at"
    }

    var isBuy: Bool { action == "BUY" }
    var confidencePercent: String { "\(Int(confidence * 100))%" }
}

// MARK: - Signal Detail Response

struct SignalDetail: Codable {
    let ticker: String
    let action: String
    let confidence: Double
    let predictedReturn5d: Double
    let entryPrice: Double
    let stopLoss: Double
    let takeProfit: Double
    let trailingStopTrigger: Double
    let timeStopDays: Int
    let positionSizePct: Double
    let sector: String
    let shortName: String
    let technical: ExplanationSection
    let fundamental: ExplanationSection
    let macro: ExplanationSection
    let mlInsight: MLInsight?
    let riskContext: RiskContext?
    let historicalContext: [String]
    let generatedAt: String

    enum CodingKeys: String, CodingKey {
        case ticker, action, confidence, sector
        case predictedReturn5d = "predicted_return_5d"
        case entryPrice = "entry_price"
        case stopLoss = "stop_loss"
        case takeProfit = "take_profit"
        case trailingStopTrigger = "trailing_stop_trigger"
        case timeStopDays = "time_stop_days"
        case positionSizePct = "position_size_pct"
        case shortName = "short_name"
        case technical, fundamental, macro
        case mlInsight = "ml_insight"
        case riskContext = "risk_context"
        case historicalContext = "historical_context"
        case generatedAt = "generated_at"
    }

    var isBuy: Bool { action == "BUY" }
}

struct ExplanationSection: Codable {
    let points: [String]
}

struct MLInsight: Codable {
    let predictedReturn: String
    let confidencePercentile: String
    let topFeatures: [String]

    enum CodingKeys: String, CodingKey {
        case predictedReturn = "predicted_return"
        case confidencePercentile = "confidence_percentile"
        case topFeatures = "top_features"
    }
}

struct RiskContext: Codable {
    let sectorExposureAfter: String
    let portfolioPositionsAfter: String
    let correlationNote: String

    enum CodingKeys: String, CodingKey {
        case sectorExposureAfter = "sector_exposure_after"
        case portfolioPositionsAfter = "portfolio_positions_after"
        case correlationNote = "correlation_note"
    }
}
