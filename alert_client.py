"""
alert_client.py — IPC helper for sending alert triggers to alert_service.py
════════════════════════════════════════════════════════════════════════════
Import this in BOTH Audio.py (venv) and monkey_detector_flask.py (system).
No third-party dependencies — stdlib only.

Usage:
    from alert_client import trigger_alert
    trigger_alert(source="audio", confidence=87.5)
    trigger_alert(source="video", confidence=91.2, human=True)
"""

import json
import socket
import logging

log = logging.getLogger("MonkeyWarning")

# Must match SOCKET_PATH in alert_service.py
SOCKET_PATH = "/tmp/monkey_alert.sock"


def trigger_alert(
    source: str,
    confidence: float,
    human: bool = False,
) -> None:
    """
    Send a detection event to the alert service.
    Non-blocking and exception-safe — will never crash the caller.

    Args:
        source:     "audio" or "video"
        confidence: detection confidence as a percentage (0–100)
        human:      True if a human was also detected in the same frame
                    (video detector only)
    """
    try:
        msg = json.dumps({
            "source":     source,
            "confidence": round(float(confidence), 2),
            "human":      bool(human),
        }).encode()

        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(msg, SOCKET_PATH)

        log.debug(f"[AlertClient] Sent trigger: source={source} conf={confidence:.1f}%")

    except FileNotFoundError:
        log.warning("[AlertClient] alert_service not running (socket not found).")
    except Exception as exc:
        log.warning(f"[AlertClient] Could not send alert: {exc}")
