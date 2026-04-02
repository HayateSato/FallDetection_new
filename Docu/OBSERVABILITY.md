# Observability — What Is Stored Where

What Prometheus collects, what PostgreSQL stores, what overlaps, and why both exist.
Last updated: 2026-03-27

---

## Prometheus — operational metrics (transient, ~15-day retention)

Prometheus **scrapes** the `/metrics` endpoint of ml_server every 15 seconds.
It stores **time-aggregated counters and histograms**, not individual rows.

### Auto-instrumented by `prometheus_fastapi_instrumentator`

Every HTTP route on ml_server is automatically tracked:

| Metric name | Type | Labels | What it captures |
|-------------|------|--------|-----------------|
| `http_requests_total` | Counter | `method`, `status_code`, `handler` | Total request count per endpoint |
| `http_request_duration_seconds` | Histogram | `method`, `handler` | End-to-end HTTP response time |
| `http_requests_in_progress` | Gauge | `method`, `handler` | Concurrent requests right now |

### Custom metrics (defined in `services/metrics_collector.py`)

| Metric name | Type | Labels | What it captures |
|-------------|------|--------|-----------------|
| `fall_detections_total` | Counter | `model_version`, `confidence_bucket` | Count of falls detected; bucket = high/medium/low confidence |
| `inference_latency_seconds` | Histogram | `model_version` | XGBoost pipeline latency per prediction (p50/p95/p99 queryable) |
| `model_confidence` | Histogram | `model_version`, `fall_detected` | Confidence score distribution — the drift detector |

### Alert rules (`infrastructure/prometheus/alert_rules.yml`)

These fire → AlertManager → email/webhook:

| Alert | Condition | Fires after | Why it matters |
|-------|-----------|-------------|----------------|
| `HighInferenceLatency` | p95 latency > 2 s | 2 minutes | Hardware slowing down, model too large, or DB blocking inference |
| `ConfidenceDrift` | Median fall confidence < 0.6 | 30 minutes | **The model drift detector** — if confidence clusters near 0.5, sensor data no longer matches training distribution → retrain |
| `MlServerDown` | `up{job="ml_server"} == 0` | 1 minute | Fall detection is completely offline |
| `HighErrorRate` | >5% 5xx responses | 5 minutes | Something is broken in the inference pipeline |

### What Prometheus does NOT store

- Individual prediction records
- Patient names or participants
- `user_fall` / `need_help` feedback values
- Feature vectors
- Anything from before the last scrape interval

---

## PostgreSQL — permanent historical record

Every `/predict` call writes one row to `inference_log`. All rows persist indefinitely.

### `inference_log` — one row per prediction

| Column | Type | What it stores |
|--------|------|----------------|
| `id` | int PK | Auto-increment |
| `timestamp` | timestamptz | When the prediction was made |
| `model_version` | varchar | e.g. `v0`, `v3` |
| `fall_detected` | bool | XGBoost output |
| `confidence` | float | XGBoost probability (0.0–1.0) |
| `window_size` | int | Number of ACC samples in window |
| `inference_mode` | varchar | `remote` (live) or `replay` (CSV offline) |
| `latency_ms` | int | Pipeline latency in milliseconds |
| `participant` | varchar | Patient name (or CSV filename for replay) |
| `user_fall` | int | **Patient feedback**: 0=pending, 1=yes, 2=no, 3=no_answer |
| `need_help` | int | **Patient feedback**: same scale |
| `step_seconds` | float | Step between windows (replay only) |
| `resampling_method` | varchar | `linear`, `decimate`, or `average` |
| `acc_sensor_type` | varchar | `bosch` or `non_bosch` |

### `feature_snapshot` — one row per feature per prediction

Stores the full feature vector alongside each prediction.
Linked via `inference_id → inference_log.id`.
Used for: debugging unexpected predictions, and future retraining (query confirmed falls + their features).

### `participant_session` — one row per recording session

Patient name, gender, start/end time, fall count.
Written by `main.py` when a recording is started/stopped.
Read by caregiver_api `/patients` list.

### `api_request_log` — one row per HTTP request to ml_server

Client IP, endpoint, status code, response time, API key hash (SHA-256 — never raw).
Audit trail only.

---

## What is stored in BOTH (overlapping data)

| Data point | Prometheus | PostgreSQL |
|------------|-----------|-----------|
| Inference latency | `inference_latency_seconds` histogram (aggregated over time) | `latency_ms` per individual row |
| Confidence score | `model_confidence` histogram (distribution over time) | `confidence` per individual row |
| Fall detected | `fall_detections_total` counter (cumulative count) | `fall_detected` boolean per row |
| Model version | label on all custom metrics | column in `inference_log` |

**Key difference:** Prometheus stores *distributions and rates* (useful for alerting and dashboards). PostgreSQL stores *individual events* (useful for history, filtering, and retraining).

---

## What is ONLY in Prometheus

| Data | Why Postgres cannot replace it |
|------|-------------------------------|
| HTTP error rates and request rates | Prometheus auto-instruments every route; Postgres only gets inference rows |
| p95 / p99 latency over a sliding window | PromQL computes this on the fly; Postgres would need a slow aggregation query |
| **`ConfidenceDrift` alert** | Fires automatically when median confidence drops — you would have to query and check Postgres manually |
| `MlServerDown` alert | Prometheus can detect when the server is unreachable; Postgres gets no writes when the server is down |
| Real-time "is the server alive?" | `/metrics` scraped every 15s; Postgres writes are async and delayed |

---

## What is ONLY in PostgreSQL

| Data | Why Prometheus cannot replace it |
|------|----------------------------------|
| `user_fall` / `need_help` patient feedback | These arrive minutes after prediction — not a metric, it's event state |
| Individual prediction history with filters | Can query: "all falls by participant X in the last 7 days where user_fall=1" |
| Feature vectors (`feature_snapshot`) | 16–22 floats per prediction — not suitable for a time-series metric store |
| Participant session records | Structured relational data — not a metric |
| Replay results (CSV offline inference) | Needed for model comparison; labelled by filename |
| Ground truth labels for retraining | `user_fall=1` rows = confirmed falls — training data source |

---

## Summary: why both exist

| Need | Use |
|------|-----|
| "Is the system healthy right now?" | Prometheus → Grafana + alerts |
| "Detect model drift automatically" | Prometheus `ConfidenceDrift` alert |
| "What happened to patient X last Tuesday?" | PostgreSQL via caregiver dashboard |
| "Compare model v0 vs v3 on the same CSV" | PostgreSQL via model comparison page |
| "Which falls did the patient confirm?" | PostgreSQL `user_fall` column |
| "Build new training data from confirmed falls" | PostgreSQL `feature_snapshot` + `user_fall=1` |
