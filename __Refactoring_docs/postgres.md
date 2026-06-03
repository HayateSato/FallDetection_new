# Postgres Reference

Two separate Postgres instances — one per layer. Same Alembic migrations run on both (0001 → 0004), but each side only actively uses its own tables.

---

## Connect

**MCS layer** (`inference_posttraining_layer` — container `fall_postgres`):

```powershell
docker exec -it fall_postgres psql -U fall_user -d fall_detection
```

**Caregiver layer** (`caregiver_layer` — container `focus_postgres`):

```powershell
docker exec -it focus_postgres psql -U fall_user -d fall_detection
```

Once inside: `\dt` lists tables, `\q` exits.

---

## Tables

| Table | Active on | Written by | Purpose |
|-------|-----------|-----------|---------|
| `inference_log` | MCS | inference_server | One row per `/predict` call |
| `feature_snapshot` | MCS | inference_server | One row per feature per call (retraining input) |
| `participant_session` | Caregiver | fall_dashboard | One row per active patient session |

---

## `inference_log` — MCS only

| Column | Type | Description |
|--------|------|-------------|
| `id` | int PK | auto-increment |
| `observation_id` | string(36) | UUID per call — carried in MQTT payload and `/confirm` |
| `patient_id` | string | patient identifier |
| `device_id` | string | SmarKo MAC address |
| `model_version` | string(64) | e.g. `v0`, `mlflow:Production:v2(v0)` |
| `fall_detected` | bool | model output |
| `confidence` | float | model confidence score |
| `window_size` | int | ACC samples used (450) |
| `latency_ms` | int | inference latency |
| `detection_time` | datetime | UTC timestamp of the prediction |
| `patient_confirmed` | string(20) | `yes` / `no` / `not_answered` — set by mobile app via `/confirm` |
| `needs_help` | bool | whether patient requested help — set by `/confirm` |

---

## `feature_snapshot` — MCS only

| Column | Type | Description |
|--------|------|-------------|
| `id` | int PK | |
| `inference_id` | int FK → inference_log.id | links to parent inference |
| `feature_name` | string(50) | e.g. `acc_x_var`, `pressure_shift` |
| `feature_value` | float | pre-computed feature value used by the model |

Used by the retraining pipeline — avoids re-running feature extraction on raw sensor data.

---

## `participant_session` — Caregiver only

| Column | Type | Description |
|--------|------|-------------|
| `id` | int PK | |
| `participant_name` | string(100) | patient_id |
| `gender` | string(10) | optional |
| `start_time` | datetime | session start |
| `end_time` | datetime | null = session still active |

> `fall_count` column removed (migration 0004) — fall counts now come from InfluxDB `fall_events` at query time.

---

## Useful queries

```sql
-- Recent inferences (MCS)
SELECT id, patient_id, fall_detected, confidence, patient_confirmed, detection_time
FROM inference_log ORDER BY id DESC LIMIT 5;

-- Feature count per inference (MCS)
SELECT inference_id, COUNT(*) FROM feature_snapshot
GROUP BY inference_id ORDER BY inference_id DESC LIMIT 5;

-- Active patient sessions (Caregiver)
SELECT participant_name, start_time FROM participant_session WHERE end_time IS NULL;

-- Row counts across all tables
SELECT 'inference_log'      AS tbl, COUNT(*) FROM inference_log
UNION ALL
SELECT 'feature_snapshot',           COUNT(*) FROM feature_snapshot
UNION ALL
SELECT 'participant_session',        COUNT(*) FROM participant_session;
```
