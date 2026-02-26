import SwiftUI

struct SignalListView: View {
    @State private var signals: [SignalSummary] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showSettings = false
    @State private var lastUpdated: Date?

    var body: some View {
        NavigationStack {
            Group {
                if isLoading && signals.isEmpty {
                    ProgressView("Loading signals...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if signals.isEmpty {
                    emptyState
                } else {
                    signalList
                }
            }
            .navigationTitle("SignalBoard")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
            .refreshable {
                await loadSignals()
            }
            .task {
                await loadSignals()
            }
        }
    }

    // MARK: - Signal List

    private var signalList: some View {
        ScrollView {
            if let lastUpdated {
                Text("Updated \(lastUpdated, style: .relative) ago")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            }

            LazyVStack(spacing: 12) {
                ForEach(signals) { signal in
                    NavigationLink(destination: SignalDetailView(ticker: signal.ticker)) {
                        SignalCardView(signal: signal)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
            .padding(.top, 8)
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("No Active Signals")
                .font(.title2.bold())
            Text("Signals appear after the daily analysis\nruns at 4:30 PM ET.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Load

    private func loadSignals() async {
        isLoading = true
        defer { isLoading = false }
        do {
            signals = try await APIClient.shared.fetchSignals()
            lastUpdated = Date()
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
