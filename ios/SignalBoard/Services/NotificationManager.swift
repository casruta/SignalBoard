import Foundation
import UserNotifications
import UIKit

class NotificationManager {
    static let shared = NotificationManager()

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(
            options: [.alert, .sound, .badge]
        ) { granted, error in
            if granted {
                DispatchQueue.main.async {
                    UIApplication.shared.registerForRemoteNotifications()
                }
            }
            if let error = error {
                print("Notification permission error: \(error)")
            }
        }
    }

    func registerToken(_ token: String) async {
        do {
            try await APIClient.shared.registerDeviceToken(token)
            print("Device token registered with backend")
        } catch {
            print("Failed to register device token: \(error)")
        }
    }
}
