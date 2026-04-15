"""
Postgres writer for inference results — called as a FastAPI BackgroundTask.

The write happens AFTER the HTTP response is sent so it never adds latency
to the /predict call. All errors are caught and logged — inference results
are returned to the caller regardless of DB availability.

Two rows are written per call:
  1. inference_log  — one row with patient_id, fall_detected, confidence, etc.
  2. feature_snapshot — one row per feature (typically 16–22 rows)

The observation_id (UUID) is already in the HTTP response, so mock_app can
include it in the MQTT alert payload. caregiver_client stores it in
fall_history.observation_id, linking the two tables for the retraining query.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def write_inference_log(
    observation_id: str,
    patient_id:     str,
    device_id:      Optional[str],
    model_version:  str,
    fall_detected:  bool,
    confidence:     float,
    window_size:    int,
    latency_ms:     int,
    detection_time: datetime,
    features:       dict,
) -> None:
    """
    Write one inference_log row + N feature_snapshot rows to Postgres.

    Called via FastAPI BackgroundTasks — never raises, never blocks /predict.
    If DATABASE_URL is not set or DB is unreachable, logs a warning and returns.
    """
    try:
        from shared.db.session import SessionLocal
        from shared.db.models import InferenceLog, FeatureSnapshot

        if SessionLocal is None:
            logger.debug("DATABASE_URL not configured — skipping inference log write")
            return

        db = SessionLocal()
        try:
            log = InferenceLog(
                observation_id = observation_id,
                patient_id     = patient_id,
                device_id      = device_id,
                model_version  = model_version,
                fall_detected  = fall_detected,
                confidence     = confidence,
                window_size    = window_size,
                latency_ms     = latency_ms,
                detection_time = detection_time,
            )
            db.add(log)
            db.flush()  # assigns log.id before committing

            for name, value in features.items():
                try:
                    float_value = float(value)
                except (TypeError, ValueError):
                    float_value = None
                db.add(FeatureSnapshot(
                    inference_id  = log.id,
                    feature_name  = str(name),
                    feature_value = float_value,
                ))

            db.commit()
            logger.debug(
                f"DB write OK  observation_id={observation_id}  "
                f"fall={fall_detected}  features={len(features)}"
            )

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    except Exception as exc:
        logger.warning(f"Inference log DB write failed (non-fatal): {exc}")
