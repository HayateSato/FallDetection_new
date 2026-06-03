`# MCS side (inference_posttraining_layer)
docker exec -it fall_postgres psql -U fall_user -d fall_detection

# Caregiver side (caregiver_layer)
docker exec -it focus_postgres psql -U fall_user -d fall_detection`

Once inside, `\dt` lists tables, `\q` exits.

---

**Tables in `fall_detection` database:**

| Table | Written by | Used on |
| --- | --- | --- |
| `inference_log` | inference_server | MCS |
| `feature_snapshot` | inference_server | MCS |
| `participant_session` | fall_dashboard | Caregiver |

> All three tables exist on **both** Postgres instances (same Alembic migrations run on both). Each side only actively uses its own subset.
> 

---

**`inference_log`** — one row per `/predict` call

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | auto-increment |
| `observation_id` | string(36) | UUID generated per call — cross-reference key for MQTT payload and `/confirm` |
| `patient_id` | string | patient identifier |
| `device_id` | string | SmarKo MAC address |
| `model_version` | string(64) | e.g. `v0`, `mlflow:Production:v2(v0)` |
| `fall_detected` | bool | model output |
| `confidence` | float | model confidence score |
| `window_size` | int | number of ACC samples used (450) |
| `latency_ms` | int | inference latency |
| `detection_time` | datetime | UTC timestamp of the prediction |
| `patient_confirmed` | string(20) | `yes` / `no` / `not_answered` — set by mobile app via `/confirm` |
| `needs_help` | bool | whether patient requested help — set by `/confirm` |

---

**`feature_snapshot`** — one row per feature per inference (typically 16–20 rows per call)

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK |  |
| `inference_id` | int FK → inference_log.id | links to the parent inference |
| `feature_name` | string(50) | e.g. `acc_x_var`, `pressure_shift` |
| `feature_value` | float | pre-computed feature value used in the model |

Used by retraining — avoids re-running feature extraction on raw sensor data.

---

**`participant_session`** — one row per active patient session

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK |  |
| `participant_name` | string(100) | patient_id |
| `gender` | string(10) | optional |
| `start_time` | datetime | session start |
| `end_time` | datetime | null = session still active |
| `fall_count` | int | running count of falls this session |