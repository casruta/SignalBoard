"""Apple Push Notification service (APNs) integration."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def send_push_notifications(
    recommendations: list[dict],
    device_tokens: list[str],
    config: dict,
) -> None:
    """Send push notifications for high-confidence signals.

    Parameters
    ----------
    recommendations : list of recommendation dicts
    device_tokens : list of APNs device token strings
    config : app config with apns section
    """
    if not device_tokens:
        logger.info("No device tokens registered, skipping push")
        return

    threshold = config["strategy"]["push_notification_threshold"]
    strong_signals = [r for r in recommendations if r["confidence"] >= threshold]

    if not strong_signals:
        logger.info("No signals above push threshold %.0f%%", threshold * 100)
        return

    apns_config = config.get("apns", {})
    key_path = apns_config.get("key_path")
    if not key_path or not Path(key_path).exists():
        logger.warning("APNs key not configured, skipping push notifications")
        return

    try:
        from aioapns import APNs, NotificationRequest

        apns = APNs(
            key=str(key_path),
            key_id=apns_config["key_id"],
            team_id=apns_config["team_id"],
            topic=apns_config["bundle_id"],
            use_sandbox=apns_config.get("use_sandbox", True),
        )

        for signal in strong_signals:
            payload = _build_payload(signal)
            for token in device_tokens:
                request = NotificationRequest(
                    device_token=token,
                    message=payload,
                )
                response = await apns.send_notification(request)
                if not response.is_successful:
                    logger.error(
                        "Push failed for %s: %s",
                        signal["ticker"],
                        response.description,
                    )
                else:
                    logger.info("Push sent: %s %s", signal["action"], signal["ticker"])

    except ImportError:
        logger.warning("aioapns not installed, skipping push notifications")
    except Exception as e:
        logger.error("Push notification error: %s", e)


def _build_payload(signal: dict) -> dict:
    """Build APNs payload from a recommendation."""
    ticker = signal["ticker"]
    action = signal["action"]
    confidence = signal["confidence"]
    predicted = signal.get("predicted_return_5d", 0)

    title = f"{action} Signal: {ticker}"
    body = f"{confidence:.0%} confidence | Predicted {predicted:+.1%} over 5 days"

    return {
        "aps": {
            "alert": {
                "title": title,
                "body": body,
            },
            "sound": "default",
            "badge": 1,
        },
        "ticker": ticker,
        "action": action,
    }
