"""
InfluxDB data fetcher — mock_app component.

Simulates what the real mobile app would do with raw wearable sensor data:
  1. The real app reads ACC data directly from the SmarKo wearable over BLE.
  2. This mock fetches the same data from InfluxDB (where the wearable already
     writes it), giving us a realistic data stream without needing the hardware
     running in a loop.

Data flow (mock):
  InfluxDB → fetch_raw_window() → MockAppPoller → POST /predict

Data flow (real app, future):
  SmarKo wearable (BLE) → buffer 9s window → POST /predict
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

from app.data_input.data_loader.influx_client_manager import _get_influxdb_client
from app.data_input.data_converter import (
    convert_acc_from_flux_to_numpy_array,
    convert_baro_from_flux_to_numpy_array,
)
from config.settings import (
    ACC_FIELD_X,
    ACC_FIELD_Y,
    ACC_FIELD_Z,
    BAROMETER_FIELD,
    INFLUXDB_BUCKET,
)

logger = logging.getLogger(__name__)


def _build_query(
    bucket:           str,
    lookback_seconds: int,
    uses_barometer:   bool,
    mac_address:      Optional[str] = None,
) -> str:
    fields_filter = (
        f'r["_field"] == "{ACC_FIELD_X}" or '
        f'r["_field"] == "{ACC_FIELD_Y}" or '
        f'r["_field"] == "{ACC_FIELD_Z}"'
    )
    if uses_barometer:
        fields_filter += f' or r["_field"] == "{BAROMETER_FIELD}"'

    q = f'''from(bucket: "{bucket}")
  |> range(start: -{lookback_seconds}s)
  |> filter(fn: (r) => r["_measurement"] == "SMART_DATA")
  |> filter(fn: (r) => {fields_filter})
'''
    if mac_address:
        q += f'  |> filter(fn: (r) => r["macAddress"] == "{mac_address}")\n'
    return q


def fetch_raw_window(
    patient_id:       str,
    lookback_seconds: int,
    uses_barometer:   bool,
    mac_address:      Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fetch the most recent sensor window for one patient from InfluxDB.

    Returns (acc_data[3,N], acc_time[N], pressure[M], pressure_time[M]).
    Pressure arrays are None when uses_barometer is False.
    Returns (None, None, None, None) if no data found or query fails.

    Note: no resampling is done here — the inference server handles that.
    """
    client = _get_influxdb_client()
    query  = _build_query(INFLUXDB_BUCKET, lookback_seconds, uses_barometer, mac_address)
    logger.debug(f"InfluxDB query:\n{query.strip()}")

    try:
        tables = client.query_api().query(query)
    except Exception as exc:
        logger.error(f"InfluxDB query failed (patient={patient_id}): {exc}")
        return None, None, None, None

    flux_records = [rec for table in tables for rec in table.records]
    logger.debug(f"InfluxDB returned {len(flux_records)} records for {patient_id}")
    if not flux_records:
        return None, None, None, None

    acc_data, acc_time = convert_acc_from_flux_to_numpy_array(
        flux_records,
        acc_field_x=ACC_FIELD_X,
        acc_field_y=ACC_FIELD_Y,
        acc_field_z=ACC_FIELD_Z,
    )
    if acc_data is None or acc_data.shape[1] == 0:
        return None, None, None, None

    pressure, pressure_time = None, None
    if uses_barometer:
        pressure, pressure_time = convert_baro_from_flux_to_numpy_array(
            flux_records, BAROMETER_FIELD
        )

    return acc_data, acc_time, pressure, pressure_time
