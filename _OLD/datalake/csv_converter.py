"""
SmarKo CSV → inference window converter for the Fall Detection datalake.

Reads CSV files exported by the SmarKo mobile app (or by main.py recording sessions)
and converts the raw sensor time-series into a sequence of sliding windows, each
formatted as the dict expected by ml_server's internal run_window_inference() helper.

SmarKo CSV column reference (key columns used here):
  timestamp              — Unix timestamp in milliseconds (integer)
  is_accelerometer_bosch — 1 when this row carries valid ACC data
  bosch_acc_x/y/z        — raw LSB integers from Bosch sensor at ~25Hz
  is_pressure            — 1 when this row carries valid barometer data
  pressure_in_pa         — barometer reading in Pascals

Typical recording: continuous stream where ACC rows and pressure rows are
interleaved at different rates (25Hz ACC, ~1Hz barometer).
"""

import io
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse a SmarKo CSV from raw bytes into a DataFrame."""
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    return df


def extract_acc(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract accelerometer rows from the DataFrame.

    Filters to rows where is_accelerometer_bosch == 1, sorts by timestamp,
    and returns raw LSB values at the hardware sample rate (~25Hz).

    Returns:
        acc_data — shape (3, N), raw LSB integers [x, y, z]
        acc_time — shape (N,),  Unix timestamps in milliseconds
    """
    required = ["timestamp", "bosch_acc_x", "bosch_acc_y", "bosch_acc_z", "is_accelerometer_bosch"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    acc_rows = (
        df[df["is_accelerometer_bosch"] == 1][
            ["timestamp", "bosch_acc_x", "bosch_acc_y", "bosch_acc_z"]
        ]
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if acc_rows.empty:
        raise ValueError("No valid accelerometer rows found (is_accelerometer_bosch == 1)")

    acc_data = np.array([
        acc_rows["bosch_acc_x"].values,
        acc_rows["bosch_acc_y"].values,
        acc_rows["bosch_acc_z"].values,
    ], dtype=float)
    acc_time = acc_rows["timestamp"].values.astype(float)

    duration_s = (acc_time[-1] - acc_time[0]) / 1000.0
    logger.info(f"ACC: {acc_data.shape[1]} samples, {duration_s:.1f}s duration")
    return acc_data, acc_time


def extract_pressure(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract barometer rows from the DataFrame.

    Returns empty arrays if the CSV has no pressure data — the inference
    pipeline handles the missing barometer gracefully (ACC-only models work fine).

    Returns:
        pressure  — shape (M,), values in Pascals
        pres_time — shape (M,), Unix timestamps in milliseconds
    """
    if "is_pressure" not in df.columns or "pressure_in_pa" not in df.columns:
        logger.info("No barometer columns found — skipping pressure extraction")
        return np.array([]), np.array([])

    baro_rows = (
        df[df["is_pressure"] == 1][["timestamp", "pressure_in_pa"]]
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if baro_rows.empty:
        logger.info("No valid pressure rows found — barometer will be skipped")
        return np.array([]), np.array([])

    pressure  = baro_rows["pressure_in_pa"].values.astype(float)
    pres_time = baro_rows["timestamp"].values.astype(float)
    logger.info(f"Barometer: {len(pressure)} samples")
    return pressure, pres_time


def split_into_windows(
    acc_data: np.ndarray,
    acc_time: np.ndarray,
    pressure: np.ndarray,
    pres_time: np.ndarray,
    window_seconds: float = 9.0,
    step_seconds: float = 3.0,
    sample_rate: float = 25.0,
) -> List[dict]:
    """
    Slide a window over the full recording and produce one dict per window.

    Each dict matches the format expected by ml_server's internal
    run_window_inference() — same fields as the /predict request body.

    Args:
        acc_data       — (3, N) raw LSB values
        acc_time       — (N,)  timestamps in milliseconds
        pressure       — (M,)  Pa values (may be empty for ACC-only)
        pres_time      — (M,)  timestamps in milliseconds (may be empty)
        window_seconds — length of each window (default 9s = 450 samples @ 50Hz after resample)
        step_seconds   — advance between windows (default 3s → ~66% overlap)
        sample_rate    — hardware ACC rate in Hz (default 25Hz for SmarKo Bosch)

    Returns:
        List of window dicts, each containing:
          acc_x/y/z, timestamps_ms, pressure, pressure_timestamps_ms,
          acc_unit ("lsb"), sample_rate, window_start_ms, window_end_ms
    """
    window_samples = int(window_seconds * sample_rate)
    step_samples   = int(step_seconds * sample_rate)
    n_samples      = acc_data.shape[1]

    if n_samples < window_samples:
        raise ValueError(
            f"Recording has {n_samples} ACC samples but window requires {window_samples} "
            f"({window_seconds}s @ {sample_rate}Hz = {window_samples} samples). "
            f"Recording is too short — need at least {window_seconds}s of data."
        )

    windows = []
    start = 0
    while start + window_samples <= n_samples:
        end    = start + window_samples
        w_acc  = acc_data[:, start:end]
        w_time = acc_time[start:end]

        if len(pres_time) > 0:
            mask        = (pres_time >= w_time[0]) & (pres_time <= w_time[-1])
            w_pressure  = pressure[mask].tolist()
            w_pres_time = pres_time[mask].tolist()
        else:
            w_pressure  = []
            w_pres_time = []

        windows.append({
            "acc_x":                  w_acc[0].tolist(),
            "acc_y":                  w_acc[1].tolist(),
            "acc_z":                  w_acc[2].tolist(),
            "timestamps_ms":          w_time.tolist(),
            "pressure":               w_pressure,
            "pressure_timestamps_ms": w_pres_time,
            "acc_unit":               "lsb",
            "sample_rate":            sample_rate,
            "window_start_ms":        float(w_time[0]),
            "window_end_ms":          float(w_time[-1]),
        })
        start += step_samples

    logger.info(
        f"Produced {len(windows)} windows "
        f"({window_seconds}s window, {step_seconds}s step, {sample_rate}Hz)"
    )
    return windows
