"""
Accelerometer signal processing implementation based on Jatesiktat & Ang (2017).

This module implements the accelerometer processing approach from the paper:
"An Elderly Fall Detection using a Wrist-worn Accelerometer and Barometer"

Processing pipeline:
1. Calculate acceleration magnitude (no filtering)
2. Detect high-impact events (>4g threshold)
3. Find 1g-crossing reference point (ground impact time)
4. Trigger barometer feature extraction

Reference:
    Jatesiktat, P., & Ang, W. T. (2017). An elderly fall detection using a
    wrist-worn accelerometer and barometer. IEEE EMBS.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class PaperMagnitudeAccelerometerConfig:
    """
    Configuration for paper-based accelerometer processing.

    Based on parameters from Jatesiktat & Ang (2017).

    Attributes
    ----------
    impact_threshold_g : float
        Minimum acceleration magnitude to trigger event detection.
        Default: 4.0g (from paper - filters most daily activities)

    crossing_threshold_g : float
        Threshold for finding reference point (1g-crossing).
        Default: 1.0g (from paper)

    sample_rate : float
        Accelerometer sampling rate in Hz.
        Default: 50.0 Hz (paper used 100 Hz)

    patch_duration_before : float
        Duration before reference point for patch extraction (seconds).
        Default: 3.0s (from paper)

    patch_duration_after : float
        Duration after reference point for patch extraction (seconds).
        Default: 3.0s (from paper)
    """
    impact_threshold_g: float = 4.0
    crossing_threshold_g: float = 1.0
    sample_rate: float = 50.0
    patch_duration_before: float = 3.0
    patch_duration_after: float = 3.0

    def __post_init__(self):
        if self.impact_threshold_g <= 0:
            raise ValueError(f"impact_threshold_g must be positive, got {self.impact_threshold_g}")
        if self.crossing_threshold_g <= 0:
            raise ValueError(f"crossing_threshold_g must be positive, got {self.crossing_threshold_g}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def patch_samples_before(self) -> int:
        """Number of samples before reference point."""
        return int(self.patch_duration_before * self.sample_rate)

    @property
    def patch_samples_after(self) -> int:
        """Number of samples after reference point."""
        return int(self.patch_duration_after * self.sample_rate)

    @property
    def total_patch_samples(self) -> int:
        """Total number of samples in a patch."""
        return self.patch_samples_before + self.patch_samples_after


@dataclass
class ImpactEvent:
    """
    Represents a detected high-impact event.

    Attributes
    ----------
    peak_index : int
        Index of the peak acceleration in the signal

    peak_magnitude : float
        Peak acceleration magnitude in g

    reference_index : int
        Index of the 1g-crossing point (ground impact reference)

    reference_time_ms : Optional[float]
        Timestamp of reference point in milliseconds

    peak_time_ms : Optional[float]
        Timestamp of peak in milliseconds
    """
    peak_index: int
    peak_magnitude: float
    reference_index: int
    reference_time_ms: Optional[float] = None
    peak_time_ms: Optional[float] = None


class PaperMagnitudeAccelerometerProcessor:
    """
    Batch processor implementing the paper's accelerometer processing approach.

    This processor implements the magnitude calculation and event detection
    as described in Jatesiktat & Ang (2017). The approach is designed to:
    1. Detect high-impact events that might be falls
    2. Find precise reference points for synchronized barometer analysis

    Key characteristics:
    - No filtering applied to accelerometer data
    - Simple magnitude calculation
    - 4g threshold for event detection
    - 1g-crossing for reference point detection

    Examples
    --------
    Basic usage:

    >>> config = PaperMagnitudeAccelerometerConfig(sample_rate=50.0)
    >>> processor = PaperMagnitudeAccelerometerProcessor(config)
    >>> magnitude = processor.compute_magnitude(acc_x, acc_y, acc_z)
    >>> events = processor.detect_impact_events(magnitude, timestamps)

    With event extraction:

    >>> for event in events:
    ...     print(f"Impact at index {event.peak_index}: {event.peak_magnitude:.2f}g")
    ...     print(f"Reference point: {event.reference_index}")
    """

    def __init__(self, config: PaperMagnitudeAccelerometerProcessor):
        """
        Initialize the paper-based accelerometer processor.

        Parameters
        ----------
        config : PaperMagnitudeAccelerometerConfig
            Configuration object with processing parameters
        """
        self.config = config

    def compute_magnitude(
        self,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceleration magnitude from 3-axis data.

        Formula: |a| = sqrt(ax² + ay² + az²)

        No filtering is applied - this follows the paper's approach
        of using raw magnitude for event detection.

        Parameters
        ----------
        acc_x : np.ndarray
            X-axis acceleration in g
        acc_y : np.ndarray
            Y-axis acceleration in g
        acc_z : np.ndarray
            Z-axis acceleration in g

        Returns
        -------
        np.ndarray
            Acceleration magnitude in g
        """
        return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    def find_1g_crossing(
        self,
        magnitude: np.ndarray,
        peak_index: int
    ) -> int:
        """
        Find the 1g-crossing point by tracing back from peak.

        The 1g-crossing point is where acceleration transitions from
        below 1g to above 1g. This marks approximately when the body
        starts hitting the ground.

        Parameters
        ----------
        magnitude : np.ndarray
            Acceleration magnitude array
        peak_index : int
            Index of the detected peak (>4g)

        Returns
        -------
        int
            Index of the 1g-crossing reference point

        Notes
        -----
        Algorithm from paper:
        1. Start at peak_index
        2. Trace backward in time
        3. Find where magnitude crosses from <1g to >1g
        """
        threshold = self.config.crossing_threshold_g

        # Trace back from peak to find 1g crossing
        for i in range(peak_index - 1, -1, -1):
            # Looking for transition from below threshold to above
            if magnitude[i] < threshold and magnitude[i + 1] >= threshold:
                return i + 1  # Return the index where it crosses above

        # If no crossing found, use the first sample below threshold
        for i in range(peak_index - 1, -1, -1):
            if magnitude[i] < threshold:
                return i

        # Fallback to start of signal
        return 0

    def detect_impact_events(
        self,
        magnitude: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        min_separation_samples: Optional[int] = None
    ) -> List[ImpactEvent]:
        """
        Detect high-impact events exceeding the threshold.

        Parameters
        ----------
        magnitude : np.ndarray
            Acceleration magnitude in g
        timestamps : np.ndarray, optional
            Timestamps in milliseconds for each sample
        min_separation_samples : int, optional
            Minimum samples between detected events.
            Default: half of patch duration

        Returns
        -------
        List[ImpactEvent]
            List of detected impact events with reference points

        Notes
        -----
        - Only events exceeding impact_threshold_g (4g) are detected
        - Each event includes the peak location and 1g-crossing reference
        - Events too close together are merged (only highest peak kept)
        """
        if min_separation_samples is None:
            min_separation_samples = self.config.patch_samples_before

        threshold = self.config.impact_threshold_g
        events = []

        # Find all samples exceeding threshold
        above_threshold = magnitude > threshold

        if not np.any(above_threshold):
            return events

        # Find contiguous regions above threshold
        # and identify the peak within each region
        in_event = False
        event_start = 0
        event_peak_idx = 0
        event_peak_val = 0

        for i in range(len(magnitude)):
            if above_threshold[i]:
                if not in_event:
                    # Start of new event
                    in_event = True
                    event_start = i
                    event_peak_idx = i
                    event_peak_val = magnitude[i]
                else:
                    # Continue event, track peak
                    if magnitude[i] > event_peak_val:
                        event_peak_idx = i
                        event_peak_val = magnitude[i]
            else:
                if in_event:
                    # End of event
                    in_event = False

                    # Check separation from last event
                    if events and (event_peak_idx - events[-1].peak_index) < min_separation_samples:
                        # Merge with previous - keep higher peak
                        if event_peak_val > events[-1].peak_magnitude:
                            events[-1] = self._create_event(
                                magnitude, event_peak_idx, event_peak_val, timestamps
                            )
                    else:
                        events.append(self._create_event(
                            magnitude, event_peak_idx, event_peak_val, timestamps
                        ))

        # Handle event at end of signal
        if in_event:
            if events and (event_peak_idx - events[-1].peak_index) < min_separation_samples:
                if event_peak_val > events[-1].peak_magnitude:
                    events[-1] = self._create_event(
                        magnitude, event_peak_idx, event_peak_val, timestamps
                    )
            else:
                events.append(self._create_event(
                    magnitude, event_peak_idx, event_peak_val, timestamps
                ))

        return events

    def _create_event(
        self,
        magnitude: np.ndarray,
        peak_idx: int,
        peak_val: float,
        timestamps: Optional[np.ndarray]
    ) -> ImpactEvent:
        """Create an ImpactEvent with reference point."""
        ref_idx = self.find_1g_crossing(magnitude, peak_idx)

        ref_time = timestamps[ref_idx] if timestamps is not None else None
        peak_time = timestamps[peak_idx] if timestamps is not None else None

        return ImpactEvent(
            peak_index=peak_idx,
            peak_magnitude=float(peak_val),
            reference_index=ref_idx,
            reference_time_ms=ref_time,
            peak_time_ms=peak_time
        )

    def process(
        self,
        acc_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[ImpactEvent]]:
        """
        Full processing pipeline for accelerometer data.

        Parameters
        ----------
        acc_data : np.ndarray
            Accelerometer data, shape (3, N) where rows are [x, y, z]
        timestamps : np.ndarray, optional
            Timestamps in milliseconds

        Returns
        -------
        Tuple[np.ndarray, List[ImpactEvent]]
            - magnitude: Acceleration magnitude array
            - events: List of detected impact events
        """
        # Compute magnitude
        magnitude = self.compute_magnitude(
            acc_data[0], acc_data[1], acc_data[2]
        )

        # Detect events
        events = self.detect_impact_events(magnitude, timestamps)

        return magnitude, events

    def get_config(self) -> PaperMagnitudeAccelerometerProcessor:
        """Get current configuration."""
        return self.config


class StreamingPaperMagnitudeAccelerometerProcessor:
    """
    Real-time streaming processor for accelerometer data.

    Maintains a buffer and detects events in real-time.
    """

    def __init__(self, config: PaperMagnitudeAccelerometerConfig):
        """Initialize streaming processor."""
        self.config = config
        self.buffer_size = config.total_patch_samples
        self.magnitude_buffer: List[float] = []
        self.timestamp_buffer: List[float] = []
        self.pending_event: Optional[ImpactEvent] = None
        self._in_event = False
        self._event_peak_idx = 0
        self._event_peak_val = 0.0

    def process_sample(
        self,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        timestamp_ms: Optional[float] = None
    ) -> Tuple[float, Optional[ImpactEvent]]:
        """
        Process a single accelerometer sample.

        Parameters
        ----------
        acc_x, acc_y, acc_z : float
            Acceleration values in g
        timestamp_ms : float, optional
            Timestamp in milliseconds

        Returns
        -------
        Tuple[float, Optional[ImpactEvent]]
            - magnitude: Current acceleration magnitude
            - event: Detected event if one just completed, None otherwise
        """
        # Compute magnitude
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        # Update buffers
        self.magnitude_buffer.append(magnitude)
        if timestamp_ms is not None:
            self.timestamp_buffer.append(timestamp_ms)

        # Maintain buffer size
        if len(self.magnitude_buffer) > self.buffer_size:
            self.magnitude_buffer.pop(0)
            if self.timestamp_buffer:
                self.timestamp_buffer.pop(0)

        # Event detection
        current_idx = len(self.magnitude_buffer) - 1
        detected_event = None

        if magnitude > self.config.impact_threshold_g:
            if not self._in_event:
                self._in_event = True
                self._event_peak_idx = current_idx
                self._event_peak_val = magnitude
            else:
                if magnitude > self._event_peak_val:
                    self._event_peak_idx = current_idx
                    self._event_peak_val = magnitude
        else:
            if self._in_event:
                # Event ended, create ImpactEvent
                self._in_event = False
                mag_array = np.array(self.magnitude_buffer)
                ref_idx = self._find_1g_crossing_in_buffer(mag_array, self._event_peak_idx)

                ts = self.timestamp_buffer if self.timestamp_buffer else None
                detected_event = ImpactEvent(
                    peak_index=self._event_peak_idx,
                    peak_magnitude=self._event_peak_val,
                    reference_index=ref_idx,
                    reference_time_ms=ts[ref_idx] if ts else None,
                    peak_time_ms=ts[self._event_peak_idx] if ts else None
                )

        return magnitude, detected_event

    def _find_1g_crossing_in_buffer(self, magnitude: np.ndarray, peak_idx: int) -> int:
        """Find 1g crossing in buffer."""
        threshold = self.config.crossing_threshold_g
        for i in range(peak_idx - 1, -1, -1):
            if magnitude[i] < threshold and magnitude[i + 1] >= threshold:
                return i + 1
        for i in range(peak_idx - 1, -1, -1):
            if magnitude[i] < threshold:
                return i
        return 0

    def reset(self) -> None:
        """Reset processor state."""
        self.magnitude_buffer.clear()
        self.timestamp_buffer.clear()
        self._in_event = False
        self._event_peak_idx = 0
        self._event_peak_val = 0.0

    def is_ready(self) -> bool:
        """Check if buffer is filled."""
        return len(self.magnitude_buffer) >= self.buffer_size
