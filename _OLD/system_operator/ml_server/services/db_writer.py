"""
PostgreSQL writer for inference results.

Called as a FastAPI BackgroundTask so the DB write never blocks the /predict response.
All errors are caught and logged — the inference result is returned regardless.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def write_inference_log(
    model_version: str,
    fall_detected: bool,
    confidence: float,
    window_size: int,
    inference_mode: str,
    latency_ms: int,
    participant: Optional[str],
    features: dict,
    step_seconds: Optional[float] = None,
    resampling_method: Optional[str] = None,
    acc_sensor_type: Optional[str] = None,
) -> Optional[int]:
    """
    Write one inference result to Postgres (inference_log + feature_snapshot rows).

    Returns the inference_log.id on success, None on failure.
    Never raises — errors are logged only.
    """
    try:
        from shared.db.session import SessionLocal
        from shared.db.models import InferenceLog, FeatureSnapshot

        if SessionLocal is None:
            logger.debug("DATABASE_URL not set — skipping DB write")
            return None

        db = SessionLocal()
        try:
            log = InferenceLog(
                timestamp=datetime.now(timezone.utc),
                model_version=model_version,
                fall_detected=fall_detected,
                confidence=confidence,
                window_size=window_size,
                inference_mode=inference_mode,
                latency_ms=latency_ms,
                participant=participant or "unknown",
                step_seconds=step_seconds,
                resampling_method=resampling_method,
                acc_sensor_type=acc_sensor_type,
            )
            db.add(log)
            db.flush()  # get auto-generated id before committing

            for name, value in features.items():
                try:
                    float_value = float(value)
                except (TypeError, ValueError):
                    float_value = None
                db.add(FeatureSnapshot(
                    inference_id=log.id,
                    feature_name=str(name),
                    feature_value=float_value,
                ))

            db.commit()
            log_id = log.id
            logger.debug(f"DB write OK — inference_log.id={log_id}, fall={fall_detected}")
            return log_id

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    except Exception as e:
        logger.error(f"DB write failed (non-fatal): {e}")
        return None


def write_inference_batch(rows: list) -> int:
    """
    Write multiple inference results to Postgres in a single transaction.

    Each item in `rows` must be a dict with the same keys as write_inference_log():
      model_version, fall_detected, confidence, window_size,
      inference_mode, latency_ms, participant, timestamp (datetime, optional)

    Returns the number of rows successfully written.
    Never raises — errors are logged only.
    """
    if not rows:
        return 0
    try:
        from shared.db.session import SessionLocal
        from shared.db.models import InferenceLog

        if SessionLocal is None:
            logger.debug("DATABASE_URL not set — skipping batch DB write")
            return 0

        db = SessionLocal()
        try:
            db.bulk_insert_mappings(InferenceLog, [
                {
                    "timestamp":         r.get("timestamp"),
                    "model_version":     r["model_version"],
                    "fall_detected":     r["fall_detected"],
                    "confidence":        r["confidence"],
                    "window_size":       r["window_size"],
                    "inference_mode":    r["inference_mode"],
                    "latency_ms":        r.get("latency_ms"),
                    "participant":       r.get("participant", "unknown"),
                    "step_seconds":      r.get("step_seconds"),
                    "resampling_method": r.get("resampling_method"),
                    "acc_sensor_type":   r.get("acc_sensor_type"),
                }
                for r in rows
                if r.get("fall_detected") is not None   # skip error windows
            ])
            db.commit()
            written = sum(1 for r in rows if r.get("fall_detected") is not None)
            logger.info(f"Batch DB write OK — {written} rows inserted")
            return written
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Batch DB write failed (non-fatal): {e}")
        return 0
