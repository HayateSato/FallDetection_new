"""
Global state manager for recording session data
This allows continuous monitoring to access participant information and recording state
"""

import threading
from typing import Optional


class RecordingState:
    """Thread-safe singleton for managing recording session state"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize state variables"""
        self._state_lock = threading.Lock()
        self._recording_active = False
        self._participant_name = "unknown"
        self._participant_gender = "unknown"
        self._manual_truth_fall = 0
        self._pending_name = None
        self._pending_gender = None
        self._pending_manual_truth = None

    def set_recording_active(self, active: bool):
        """Set recording state and apply pending values when activated"""
        with self._state_lock:
            self._recording_active = active

            # Apply pending values when recording is activated
            if active:
                if self._pending_name is not None:
                    self._participant_name = self._pending_name
                    self._pending_name = None

                if self._pending_gender is not None:
                    self._participant_gender = self._pending_gender
                    self._pending_gender = None

                if self._pending_manual_truth is not None:
                    self._manual_truth_fall = self._pending_manual_truth
                    self._pending_manual_truth = None

    def is_recording_active(self) -> bool:
        """Check if recording is currently active"""
        with self._state_lock:
            return self._recording_active

    def update_participant_name(self, name: str):
        """Update participant name (will be applied on next recording activation)"""
        with self._state_lock:
            self._pending_name = name

    def update_participant_gender(self, gender: str):
        """Update participant gender (will be applied on next recording activation)"""
        with self._state_lock:
            if gender.lower() == 'male':
                self._pending_gender = 0
            elif gender.lower() == 'female':
                self._pending_gender = 1
            else:
                self._pending_gender = 0

    def update_manual_truth(self, ground_truth: int):
        """Update ground truth annotation"""
        with self._state_lock:
            self._pending_manual_truth = ground_truth

    def get_current_state(self) -> dict:
        """Get current recording state"""
        with self._state_lock:
            return {
                'recording_active': self._recording_active,
                'participant_name': self._participant_name,
                'participant_gender': self._participant_gender,
                'ground_truth_fall': self._manual_truth_fall
            }

    def get_active_values(self) -> tuple:
        """Get currently active values for CSV export"""
        with self._state_lock:
            return (
                self._participant_name,
                self._participant_gender,
                self._manual_truth_fall
            )


# Create singleton instance
recording_state = RecordingState()
