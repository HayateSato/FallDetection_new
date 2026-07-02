# FallDetection_new — Project Memory

## Project Overview
Fall detection system using XGBoost models + InfluxDB + SmarKo wearable.
Three separate use cases live in the same repo under different branches/folders.

## Three Use Cases — Branch / Folder Separation

### 1. Full System — branch `complete_system`
Research/development stack. 10-service Docker Compose. See CLAUDE.md for full architecture.
- `main.py` + `system_operator/ml_server/` with full MLOps: Postgres, Redis, Prometheus, model hot-swap, patient feedback loop, caregiver/emergency dashboards, Live Monitor, MinIO replay
- All development priorities complete as of 2026-03-27.

### 2. Internal Company Integration — folder `_EcoSystem_Integration/` (branch `6G-integration`)
Stripped-down, self-contained. No Docker, no Redis, no Postgres, no dashboards.
- Two components: `inference_server/server.py` (FastAPI :8001) + `trigger_client/client.py`
- Model selection **per-request**: client sends `model_version` in POST body; server caches lazily
- Integration point: `on_result` callback in `FallDetectionClient` — colleague replaces with alerting/DB/MQTT
- No FHIR output. Run from `_EcoSystem_Integration/` as cwd.

### 3. Charite/FOCUS Medical Partner — folders `_6G_Integration_v2_redis/` (frozen) + `_6G_Integration_v2_mqtt/` (active)
Multi-component stack with Redis event bus + FHIR output + patient feedback popup.
See detailed section below.

## Handover docs convention
- `handover_docs/` (legacy) — per-audience hand-offs (ADMIN_..., ISA_..., Tech_integrator.md). See [feedback_handover_docs.md](feedback_handover_docs.md).
- `handover_docs_2/` (active) — numbered topic files (`01_k8s.md` ... `08_user_flow_admin.md`) + running `Q&A.md`. New handover questions go into `Q&A.md`, newest at top. See [project_handover_docs_2.md](project_handover_docs_2.md).

## Code/doc indexes (jcodemunch, jdocmunch)
Code index scoped to `_6G_Integration_v2_mqtt/`; doc index covers the whole repo. Prefer over native Grep/Read. See [reference_munch_indexes.md](reference_munch_indexes.md).

## User Preferences
- Learning MLOps: explain architectural decisions, not just code
- Uses Windows laptops — all commands should be PowerShell syntax
- SmarKo hardware (full system / internal): ACC at 25Hz (Bosch) upsampled to 50Hz, barometer at 25Hz
- SmarKo hardware (6G / Charite integration): InfluxDB data already at 50Hz — `HARDWARE_ACC_SAMPLE_RATE=50` in `.env`; resampler code kept but server skips it when source == target rate
- No emojis in code/files unless asked
- Prefers `curl.exe` not `curl` (PowerShell alias conflict)
- Use `Select-String` or `findstr` not `grep` (not available in PowerShell)
- Use `$env:VAR = "value"` not `set VAR=value` for env vars in PowerShell
- **PowerShell 5.1 cannot parse non-ASCII characters (em-dash, smart quotes, etc.) in `.ps1` script files** when saved as UTF-8 without BOM. The 3rd byte of `—` (U+2014) is `0x94`, which Windows-1252 reads as `"` — terminating string literals prematurely. **Use plain ASCII (`-`, `--`, straight quotes) in `.ps1` files.** Markdown/Python files are fine — only `.ps1` execution is affected.

## Models
All XGBoost. Files in `model/` dir: v0, v0_lsb_int, v3 (BEST — ACC+BARO), v5_lsb.
Window: 9s @ 50Hz = 450 ACC samples.

## Key Known Issues / Gotchas (Full System)
- Docker Compose must always use `--env-file .env` flag (compose file is in `infrastructure/` subdirectory)
- Bcrypt hashes contain `$` — use `infrastructure/caregiver_secrets.env` with `$$` for Docker; `caregiver/api/.env` with `$` for dev mode
- Grafana `GF_SECURITY_ADMIN_PASSWORD` env var ignored after first startup — use `grafana cli admin reset-admin-password`
- nginx `proxy_pass http://grafana/;` (trailing slash) breaks Grafana sub-path — must be `http://grafana;`
- Grafana datasource UIDs fixed on creation: PostgreSQL=`PCC52D03280B7034C`, Prometheus=`PBFA97CFB590B2093`
- ml_server must run `--workers 1` — asyncio timer tasks are per-process; feedback POST on a different worker cannot cancel timer
- nginx SSE locations MUST be declared before their parent `/api/ml/` and `/api/caregiver/` blocks
- Alembic must run inside Docker container: `docker exec fall_ml_server alembic upgrade head`

## Patient Dashboard (caregiver-facing UI)
One Flutter web app shown to the caregiver, owned by **FOCUS DevOps team** (NOT Isa — corrected 2026-04-29). Combines patient info panel (FHIR demographics + InfluxDB biosignals) and fall panel (our fall_dashboard API). See [project_patient_dashboard.md](project_patient_dashboard.md). Isa = mobile app only.

## Unified dashboard URL with role split
Single ingress hostname routes 4 dashboards by path + role: caregiver sees Patient Dashboard, admin sees ml_dashboard + server health. Roles are mutually exclusive. See [project_unified_dashboard_rbac.md](project_unified_dashboard_rbac.md).

## K8s local testing — gotchas + diagnostic shortcuts
Smoke-test PASSED 2026-04-29; 10 pods Running. Full local-equivalent flow verified in K8s: MQTT alert → SSE, ml-dashboard retrain + hot-swap + rollback, server-health all 6 probes healthy, Grafana K8s dashboards loading. **fall-detection test.ps1: 8/8 probes PASS** (added 2026-04-29 alongside build/install/port-forward/teardown scripts at `helm/fall-detection/`). 16 numbered gotchas (helm upgrade ≠ pod restart on `latest`, K8s `<SVC>_PORT` URL injection, MLflow DNS rebinding protection, Grafana provisioning needs SUBDIRECTORY mounts, Windows IPv4/IPv6 localhost vs 127.0.0.1, BusyBox wget no `--user`/`--password`, etc.). See [project_k8s_local_testing.md](project_k8s_local_testing.md).

## 6G / Charite Integration — MQTT version active (2026-04-15)
Active folder: `_6G_Integration_v2_mqtt/`. Redis version frozen at `_6G_Integration_v2_redis/`.
**See [project_6g_mqtt.md](project_6g_mqtt.md) for current architecture, run commands, open steps, and gotchas.**

## FOCUS DevOps meeting decisions (2026-04-29)
Hardware budget ≤15 GB RAM, k3s + Traefik, single namespace, no NetworkPolicy, FHIR opt-out, registry `registry-smarko-health.de` + pull secret `mcs-labs`, new clinical 3-state dashboard requirement. Mohammed taking over K8s integration. See [project_focus_meeting_decisions.md](project_focus_meeting_decisions.md).

**See `REFACTOR_DOCUS/deployment_architecture.md` for two-namespace breakdown and data source table.**

**Architecture (2 MQTT clients only):**
```
mock_app → HTTP POST /predict → inference_server (:8001) → HTTP response (fall=True, observation_id=UUID)
mock_app → 10s patient confirmation → PUBLISH fall/alert/<pid> (includes observation_id, patient_confirmed, needs_help)
              → [broker] → fall_dashboard (:8002) → DB write (fall_history) + SSE fan-out
```
Inference server has NO MQTT client. `influx_marker_writer.py` deleted (colleague handles InfluxDB).

**Completed steps (as of 2026-04-29):** Steps 1–4, 6a, 6b, 6c, 7, 9 (templates), 11, 11.5.2, 11.5.3, 12.5 done. Steps 5, 8, 10, 11.5.4, 11.5.5 open. Helm chart now has 10 services (added ml-dashboard:8004 + server-health:8006 on 2026-04-29).

**Key new folders since last session:**
- `shared/db/` — shared SQLAlchemy ORM models + session factory + Alembic migrations
- `retrain/` — MLflow retraining pipeline: data_pipeline.py, retrain.py, seed_test_data.py

**observation_id cross-reference:** UUID generated per /predict, returned in HTTP response, included in MQTT payload, stored in both inference_log and fall_history. Enables retraining JOIN without synchronous DB write in the HTTP handler.

**Retraining data source: Postgres (not InfluxDB).** Features pre-computed and stored in feature_snapshot. seed_test_data.py --synthetic seeds Postgres without needing InfluxDB or Charite patients.
