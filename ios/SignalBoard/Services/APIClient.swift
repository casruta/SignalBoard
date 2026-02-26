import Foundation

actor APIClient {
    static let shared = APIClient()

    private var baseURL: String {
        UserDefaults.standard.string(forKey: "backend_url") ?? "http://localhost:8000"
    }

    // MARK: - Fetch Signal List

    func fetchSignals() async throws -> [SignalSummary] {
        let url = URL(string: "\(baseURL)/signals")!
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.serverError
        }
        let decoded = try JSONDecoder().decode(SignalListResponse.self, from: data)
        return decoded.signals
    }

    // MARK: - Fetch Signal Detail

    func fetchSignalDetail(ticker: String) async throws -> SignalDetail {
        let url = URL(string: "\(baseURL)/signals/\(ticker)")!
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw APIError.serverError
        }
        if http.statusCode == 404 {
            throw APIError.notFound
        }
        guard http.statusCode == 200 else {
            throw APIError.serverError
        }
        return try JSONDecoder().decode(SignalDetail.self, from: data)
    }

    // MARK: - Register Device Token

    func registerDeviceToken(_ token: String) async throws {
        let url = URL(string: "\(baseURL)/device-token")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["token": token])
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.serverError
        }
    }
}

enum APIError: LocalizedError {
    case serverError
    case notFound

    var errorDescription: String? {
        switch self {
        case .serverError: return "Server error. Check your connection."
        case .notFound: return "Signal not found."
        }
    }
}
