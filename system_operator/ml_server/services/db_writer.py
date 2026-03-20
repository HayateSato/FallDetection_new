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
