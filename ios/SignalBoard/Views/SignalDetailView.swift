import SwiftUI

struct SignalDetailView: View {
    let ticker: String
    @State private var detail: SignalDetail?
    @State private var isLoading = true
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            if isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity, minHeight: 200)
            } else if let detail {
                detailContent(detail)
            } else if let errorMessage {
                Text(errorMessage)
                    .foregroundStyle(.red)
                    .padding()
            }
        }
        .navigationTitle(ticker)
        .navigationBarTitleDisplayMode(.large)
        .task {
            await loadDetail()
        }
    }

    // MARK: - Detail Content

    @ViewBuilder
    private func detailContent(_ d: SignalDetail) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            // Header
            header(d)

            // Price levels
            priceLevels(d)

            // Explanation sections
            if !d.technical.points.isEmpty {
                explanationSection("Technical Analysis", icon: "chart.xyaxis.line", points: d.technical.points, color: .blue)
            }
            if !d.fundamental.points.isEmpty {
                explanationSection("Fundamental Analysis", icon: "building.2", points: d.fundamental.points, color: .purple)
            }
            if !d.macro.points.isEmpty {
                explanationSection("Macro Environment", icon: "globe", points: d.macro.points, color: .orange)
            }

            // ML Insight
            if let ml = d.mlInsight {
                mlInsightSection(ml)
            }

            // Historical Context
            if !d.historicalContext.isEmpty {
                historicalSection(d.historicalContext)
            }
        }
        .padding()
    }

    // MARK: - Header

    private func header(_ d: SignalDetail) -> some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading) {
                    Text(d.shortName)
                        .font(.headline)
                        .foregroundStyle(.secondary)
                    Text(d.sector)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text(d.action)
                        .font(.title2.bold())
                        .foregroundStyle(d.isBuy ? .green : .red)
                    Text("\(Int(d.confidence * 100))% confidence")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            // Predicted return bar
            HStack {
                Text("Predicted 5-day return")
                    .font(.subheadline)
                Spacer()
                Text(String(format: "%+.1f%%", d.predictedReturn5d * 100))
                    .font(.title3.bold())
                    .foregroundStyle(d.predictedReturn5d >= 0 ? .green : .red)
            }
            .padding()
            .background(Color(.systemGray6))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - Price Levels

    private func priceLevels(_ d: SignalDetail) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Trade Parameters")
                .font(.headline)

            HStack {
                priceBox("Entry", value: d.entryPrice, color: .blue)
                priceBox("Stop Loss", value: d.stopLoss, color: .red)
                priceBox("Target", value: d.takeProfit, color: .green)
            }

            HStack {
                Text("Time stop: \(d.timeStopDays) days")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("Trail trigger: $\(d.trailingStopTrigger, specifier: "%.2f")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func priceBox(_ label: String, value: Double, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("$\(value, specifier: "%.2f")")
                .font(.subheadline.bold())
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Explanation Sections

    private func explanationSection(
        _ title: String, icon: String, points: [String], color: Color
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.headline)
                .foregroundStyle(color)

            ForEach(points, id: \.self) { point in
                HStack(alignment: .top, spacing: 8) {
                    Circle()
                        .fill(color)
                        .frame(width: 6, height: 6)
                        .padding(.top, 6)
                    Text(point)
                        .font(.subheadline)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - ML Insight

    private func mlInsightSection(_ ml: MLInsight) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("ML Model Insight", systemImage: "brain")
                .font(.headline)
                .foregroundStyle(.indigo)

            HStack {
                Text("Predicted return:")
                Spacer()
                Text(ml.predictedReturn)
                    .bold()
            }
            .font(.subheadline)

            HStack {
                Text("Confidence percentile:")
                Spacer()
                Text(ml.confidencePercentile)
                    .bold()
            }
            .font(.subheadline)

            if !ml.topFeatures.isEmpty {
                Text("Top features: \(ml.topFeatures.joined(separator: ", "))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.indigo.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Historical Context

    private func historicalSection(_ context: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Historical Context", systemImage: "clock.arrow.circlepath")
                .font(.headline)
                .foregroundStyle(.teal)

            ForEach(context, id: \.self) { text in
                Text(text)
                    .font(.subheadline)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.teal.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Load

    private func loadDetail() async {
        isLoading = true
        defer { isLoading = false }
        do {
            detail = try await APIClient.shared.fetchSignalDetail(ticker: ticker)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
