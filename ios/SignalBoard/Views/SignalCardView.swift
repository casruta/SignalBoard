import SwiftUI

struct SignalCardView: View {
    let signal: SignalSummary

    var body: some View {
        HStack(spacing: 16) {
            // Action badge
            VStack {
                Text(signal.action)
                    .font(.caption.bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(actionColor)
                    .clipShape(Capsule())

                Text(signal.confidencePercent)
                    .font(.title2.bold())
                    .foregroundStyle(actionColor)
            }
            .frame(width: 70)

            // Ticker info
            VStack(alignment: .leading, spacing: 4) {
                Text(signal.ticker)
                    .font(.title3.bold())
                    .foregroundStyle(.primary)
                Text(signal.shortName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Text(signal.sector)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            // Predicted return
            VStack(alignment: .trailing, spacing: 4) {
                Text(formattedReturn)
                    .font(.headline)
                    .foregroundStyle(signal.predictedReturn5d >= 0 ? .green : .red)
                Text("5-day est.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding()
        .background(cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: actionColor.opacity(0.15), radius: 4, y: 2)
    }

    private var actionColor: Color {
        signal.isBuy ? .green : .red
    }

    private var cardBackground: some ShapeStyle {
        actionColor.opacity(0.06)
    }

    private var formattedReturn: String {
        let pct = signal.predictedReturn5d * 100
        return String(format: "%+.1f%%", pct)
    }
}
