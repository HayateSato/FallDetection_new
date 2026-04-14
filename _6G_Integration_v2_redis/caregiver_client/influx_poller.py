"""
InfluxDB poller — runs in a background thread.

For each registered patient:
  1. Query InfluxDB for the last N seconds of ACC (+ optional barometer) data
  2. Convert FluxRecord → raw numpy arrays (NO resampling — server does that)
  3. POST to the inference server's /predict endpoint
  4. On the response, write a fall_history row to the local DB
  5. (Server publishes the fall to Redis itself; this poller does NOT touch Redis)

Patient identification:
  - PATIENT_IDS env var: comma-separated list of patient IDs
  - Each patient is mapped to an InfluxDB tag value via INFLUX_PATIENT_TAG (default 'patient_id')
  - If INFLUX_PATIENT_TAG is empty/unset, the poller queries the bucket without
    filtering and assigns all data to PATIENT_IDS[0] — useful for single-patient demos.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

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

from caregiver_client import db as cdb
from caregiver_client.inference_client import InferenceServerClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Influx query helpers
# ---------------------------------------------------------------------------

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
    Returns (acc_data[3,N], acc_time[N], pressure[M], pressure_time[M]).
    Pressure arrays are None when uses_barometer is False.
    Returns (None, None, None, None) on no data.
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
        pressure, pressure_time = convert_baro_from_flux_to_numpy_array(flux_records, BAROMETER_FIELD)

    return acc_data, acc_time, pressure, pressure_time


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

class InfluxPoller(threading.Thread):
    """
    Background thread that loops over the configured patients,
    fetches the latest sensor window from InfluxDB, calls the inference
    server, and stores the result locally.
    """

    def __init__(
        self,
        inference_client: InferenceServerClient,
        patient_ids:      List[str],
        mac_map:       Optional[dict] = None,
        poll_interval:    int = 10,
        lookback_seconds: int = 15,
        on_fall=None,
    ):
        super().__init__(daemon=True, name="InfluxPoller")
        self.client            = inference_client
        self.patient_ids       = patient_ids
        self.mac_map        = mac_map or {}
        self.poll_interval     = poll_interval
        self.lookback_seconds  = lookback_seconds
        self.on_fall           = on_fall          # optional callback(fall_dict)
        self._stop             = threading.Event()
        self._uses_barometer:  Optional[bool] = None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _ensure_baro_known(self) -> bool:
        if self._uses_barometer is None:
            self._uses_barometer = self.client.uses_barometer()
            logger.info(f"Server uses_barometer = {self._uses_barometer}")
        return self._uses_barometer

    # ------------------------------------------------------------------
    def _poll_one(self, patient_id: str) -> None:
        uses_baro = self._ensure_baro_known()
        mac_address = self.mac_map.get(patient_id)
        acc_data, acc_time, pressure, pressure_time = fetch_raw_window(
            patient_id       = patient_id,
            lookback_seconds = self.lookback_seconds,
            uses_barometer   = uses_baro,
            mac_address      = mac_address,
        )
        if acc_data is None:
            logger.info(f"No InfluxDB data for {patient_id} (lookback={self.lookback_seconds}s)")
            return

        device_id = self.mac_map.get(patient_id)
        result = self.client.predict(
            patient_id    = patient_id,
            device_id     = device_id,
            acc_x         = acc_data[0].tolist(),
            acc_y         = acc_data[1].tolist(),
            acc_z         = acc_data[2].tolist(),
            timestamps_ms = acc_time.tolist(),
            pressure                = pressure.tolist()      if pressure      is not None else None,
            pressure_timestamps_ms  = pressure_time.tolist() if pressure_time is not None else None,
        )
        if result is None:
            return

        inference = result.get("inference", {})
        is_fall   = bool(inference.get("fall_detected", False))
        det_time  = result.get("timestamp")
        try:
            det_time_dt = datetime.fromisoformat(det_time) if det_time else datetime.now(timezone.utc)
        except Exception:
            det_time_dt = datetime.now(timezone.utc)

        cdb.ensure_session(patient_id)
        fall_id = cdb.record_fall(
            patient_id        = patient_id,
            fall_detected     = is_fall,
            detection_time    = det_time_dt,
            patient_confirmed = "not_answered",
        )

        if is_fall:
            logger.info(f"FALL stored  patient={patient_id}  id={fall_id}  "
                        f"confidence={inference.get('confidence')}")
            if self.on_fall is not None:
                try:
                    self.on_fall({
                        "fall_id":        fall_id,
                        "patient_id":     patient_id,
                        "mac_id":         mac_address,
                        "fall_detected":  True,
                        "confidence":     inference.get("confidence"),
                        "model_version":  inference.get("model_version"),
                        "detection_time": det_time_dt.isoformat(),
                        "fhir_observation": result.get("fhir_observation"),
                    })
                except Exception as exc:
                    logger.warning(f"on_fall callback raised: {exc}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info(
            f"InfluxPoller started  patients={self.patient_ids}  "
            f"interval={self.poll_interval}s  lookback={self.lookback_seconds}s  "
            f"mac_map={self.mac_map or '(none)'}"
        )
        while not self._stop.is_set():
            for pid in self.patient_ids:
                if self._stop.is_set():
                    break
                try:
                    self._poll_one(pid)
                except Exception as exc:
                    logger.error(f"Poll error for {pid}: {exc}", exc_info=True)
            self._stop.wait(self.poll_interval)
        logger.info("InfluxPoller stopped")
