"""
Shared mutable state for the Fall Detection system.

Holds queues, locks, and globals that are accessed across
multiple route blueprints and the continuous monitoring thread.
"""
import queue
import threading

# SSE notification queue (continuous_monitoring -> /events SSE stream)
fall_notification_queue = queue.Queue()

# Polling notification storage (thread-safe)
poll_notifications = []
poll_lock = threading.Lock()

# Track the last exported CSV filepath so /fall_feedback can retroactively update it
last_exported_csv_path = None
csv_path_lock = threading.Lock()

# Global monitor instance (set by main.py on startup)
continuous_monitor = None


def add_poll_notification(fall_data: dict):
    """Store a fall notification for polling clients to pick up."""
    with poll_lock:
        poll_notifications.append(fall_data)
        # Keep max 50 to prevent unbounded growth
        if len(poll_notifications) > 50:
            poll_notifications.pop(0)
