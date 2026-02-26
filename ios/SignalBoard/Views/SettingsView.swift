import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage("backend_url") private var backendURL = "http://localhost:8000"
    @AppStorage("min_confidence") private var minConfidence = 70.0
    @State private var notificationsEnabled = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("Backend URL", text: $backendURL)
                        .textContentType(.URL)
                        .autocapitalization(.none)
                        .keyboardType(.URL)
                }

                Section("Filters") {
                    VStack(alignment: .leading) {
                        Text("Minimum confidence: \(Int(minConfidence))%")
                        Slider(value: $minConfidence, in: 50...95, step: 5)
                    }
                }

                Section("Notifications") {
                    Toggle("Push notifications", isOn: $notificationsEnabled)
                        .onChange(of: notificationsEnabled) { _, newValue in
                            if newValue {
                                NotificationManager.shared.requestPermission()
                            }
                        }
                }

                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Data sources")
                        Spacer()
                        Text("Yahoo Finance, FRED")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
