"""
Prometheus custom metrics for the ML inference server.

Metrics defined here:
  fall_detections_total       — Counter: total fall events, labelled by model_version + confidence_bucket
  inference_latency_seconds   — Histogram: full pipeline latency per predict call
  model_confidence            — Histogram: XGBoost confidence score distribution

Setup in server.py:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)   # auto-instruments HTTP routes

    from inference_server.services.metrics_collector import record_prediction

Why track confidence distribution?
  A healthy model returns scores bimodal (near 0 or near 1).
  If scores cluster near 0.5 (the threshold), the model may be seeing distribution shift —
  incoming sensor data no longer matches the training distribution.
"""

from prometheus_client import Counter, Histogram

fall_detections_total = Counter(
    "fall_detections_total",
    "Total number of fall detection events",
    ["model_version", "confidence_bucket"],
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "End-to-end inference pipeline latency in seconds",
    ["model_version"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

model_confidence = Histogram(
    "model_confidence",
    "XGBoost confidence score distribution (0.0 to 1.0)",
    ["model_version", "fall_detected"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def record_prediction(model_version: str, fall_detected: bool,
                      confidence: float, latency_seconds: float) -> None:
    """Update all Prometheus metrics for one prediction. Never raises."""
    try:
        inference_latency_seconds.labels(model_version=model_version).observe(latency_seconds)
        model_confidence.labels(
            model_version=model_version,
            fall_detected=str(fall_detected).lower(),
        ).observe(confidence)
        if fall_detected:
            fall_detections_total.labels(
                model_version=model_version,
                confidence_bucket=_confidence_bucket(confidence),
            ).inc()
    except Exception:
        pass  # metrics are best-effort, never block inference
