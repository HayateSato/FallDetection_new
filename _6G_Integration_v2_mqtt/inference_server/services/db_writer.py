"""
Postgres writer for inference results — called as FastAPI BackgroundTasks.

Writes happen AFTER the HTTP response is sent so they never add latency.
All errors are caught and logged — inference results are returned regardless
of DB availability.

write_inference_log  — called by /predict; writes inference_log + feature_snapshot rows
write_confirmation   — called by /inference/{observation_id}/confirm; updates
                       patient_confirmed and needs_help on the inference_log row
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
        from shared_db.db.session import SessionLocal
        from shared_db.db.models import InferenceLog, FeatureSnapshot

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


def write_confirmation(
    observation_id:    str,
    patient_confirmed: str,
    needs_help:        Optional[bool],
) -> None:
    """
    Update patient_confirmed and needs_help on an existing inference_log row.

    Called via FastAPI BackgroundTasks from POST /inference/{observation_id}/confirm.
    The mobile app calls this endpoint after the patient responds to the confirmation
    popup (or after the 10-second timeout).

    Never raises — if the row is not found or the DB is unreachable, logs a warning.
    """
    try:
        from shared_db.db.session import SessionLocal
        from shared_db.db.models import InferenceLog
        from sqlalchemy import select

        if SessionLocal is None:
            logger.debug("DATABASE_URL not configured — skipping confirmation write")
            return

        db = SessionLocal()
        try:
            row = db.scalar(
                select(InferenceLog).where(InferenceLog.observation_id == observation_id)
            )
            if row is None:
                logger.warning(
                    f"Confirmation received for unknown observation_id={observation_id} — "
                    "inference_log row not yet written or ID is wrong"
                )
                return
            row.patient_confirmed = patient_confirmed
            row.needs_help        = needs_help
            db.commit()
            logger.debug(
                f"Confirmation written  observation_id={observation_id}  "
                f"confirmed={patient_confirmed}  needs_help={needs_help}"
            )
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    except Exception as exc:
        logger.warning(f"Confirmation DB write failed (non-fatal): {exc}")
