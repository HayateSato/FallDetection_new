# User Flow — Admin / Tech

**Audience:** the engineer or ML admin who will operate the fall-detection backend after handover. This is whoever inherits Hayate's role at MCS, plus FOCUS DevOps who own the cluster.
**Scope:** what the admin sees, what they do, how they handle the day-to-day operations, retraining, monitoring, deployment, and incident response.

This is the human flow. Technical references are in `01_k8s.md`, `02_fall_detection_algorithm.md`, and `03_fall_detection_system.md`.

---

## 1. The admin in one paragraph

The admin keeps the fall-detection backend running, healthy, and accurate. They do **not** see patient data unless explicitly debugging an incident. They monitor system metrics, retrain the model on a cadence (or in response to specific signals), promote new model versions, and deploy the helm chart. They are mutually exclusive with the caregiver role — admins do not see the Patient Dashboard fall flags.

---

## 2. The admin's three dashboards

| Dashboard | Port | What it's for |
|-----------|:----:|---------------|
| **`ml_dashboard`** | :8004 | Trigger retrain, promote model versions, hot-swap, see MLflow runs at a glance |
| **`server_health`** | :8006 | Aggregate `/health` probes across all our pods — green/red status board |
| **Grafana** (in the chart) | :3000 | 3 pre-provisioned dashboards: ml_server_overview, model_performance, fall_events_timeline |

These are all admin-only. The Patient Dashboard (caregiver tool) does not link to them and vice versa. RBAC is planned via FOCUS SSO with mutually-exclusive roles (`admin` vs `caregiver`).

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ADMIN OVERVIEW (browser tabs the admin keeps open)                     │
│                                                                         │
│  Tab 1   ml_dashboard          → http://localhost:8004                  │
│            • current production model version                           │
│            • [Retrain] [Promote] [Hot-Swap] buttons                     │
│            • MLflow runs table                                          │
│                                                                         │
│  Tab 2   server_health         → http://localhost:8006                  │
│            • inference-server  ✓ green                                  │
│            • fall-dashboard    ✓ green                                  │
│            • postgres          ✓ green                                  │
│            • mqtt-broker       ✓ green                                  │
│            • mlflow            ✓ green                                  │
│            • minio             ✓ green                                  │
│                                                                         │
│  Tab 3   Grafana               → http://localhost:3000                  │
│            • request rate, latency, falls/hour (ml_server_overview)     │
│            • confidence distribution, drift flag (model_performance)    │
│            • SQL: falls today, recent events (fall_events_timeline)     │
│                                                                         │
│  Tab 4   MLflow UI (optional) → http://localhost:5000                   │
│            • Run comparison, artifact download                          │
│                                                                         │
│  Tab 5   Postgres (CLI on demand)                                       │
│            docker exec -it fall_postgres psql -U fall_user              │
│              -d fall_detection                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-end admin flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DAILY                                                                  │
│  • Glance at server_health — all green?                                 │
│  • Glance at Grafana ml_server_overview — request rate normal?          │
│    error rate ≈ 0?                                                      │
│  • Glance at fall_events_timeline — anomalous spikes?                   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  WEEKLY                                                                 │
│  • Skim model_performance — confidence distribution drifting?           │
│    KS-distance flag firing?                                             │
│  • Run a /api/falls SELECT to check FP rate per patient.                │
│  • If anomalies → consider retrain or threshold tuning.                 │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AS NEEDED — RETRAIN                                                    │
│  • Trigger via ml_dashboard "Retrain" button (or CLI).                  │
│  • Wait for the run; inspect in MLflow UI.                              │
│  • Compare metrics against current Production version.                  │
│  • If recall improved AND F1 didn't drop more than ~2pts → promote.     │
│  • Promote: None → Staging → smoke test → Production.                   │
│  • Hot-swap inference-server to load the new Production version.        │
│  • Verify on Grafana model_performance that confidence distribution     │
│    looks reasonable for ~30 minutes of live traffic.                    │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AS NEEDED — DEPLOY                                                     │
│  • Build new image tag (inference-server, fall-dashboard, or mlflow).   │
│  • Push to FOCUS registry.                                              │
│  • helm upgrade with the new tag.                                       │
│  • Watch rollout in kubectl. Verify migrate Job ran (if schema change). │
│  • Smoke test /health, /predict, SSE.                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ON INCIDENT                                                            │
│  • server_health goes red, or Grafana fires an alert, or a clinical     │
│    report comes in (missed fall, repeated false positives).             │
│  • Triage in this order: pod logs → DB consistency → model behaviour.   │
│  • If the model regressed: hot-swap to a known-good version (rollback). │
│  • If a pod is down: kubectl describe + logs; restart only after        │
│    diagnosing the root cause.                                           │
│  • Document the incident so the next admin doesn't relearn it.          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Daily routine — what the admin actually does

Day-to-day, the admin's job is mostly passive monitoring. Active work happens on weekly+ cadences (retrain, deploy) or on incidents.

### 4.1 The 5-minute morning glance

```powershell
# In the admin's laptop — 3 browser tabs already open

# Tab 1 — server_health (or:)
kubectl get pods -n mcs-fall-detection
# All Running / Completed?  Yes → next.  Anything else → triage.

# Tab 2 — Grafana ml_server_overview
# Request rate: should be roughly N_patients × (1 / 9s) per second.
#   If 0 → mobile apps are not reaching us. Check ingress.
#   If 10x normal → some test harness is loose. Investigate.
# Error rate: should be ≈ 0.
# Latency p99: should be < 200ms typically.

# Tab 3 — Grafana fall_events_timeline
# Falls today: roughly within historical bounds?
#   If 50× normal → false-positive storm; consider threshold or retrain.
#   If 0 in 24h with active patients → mobile-app or MQTT is silently broken.
```

If everything looks normal, that's the morning done. Go do other work.

### 4.2 Weekly review (≈30 minutes)

```powershell
# 1. Inspect Grafana model_performance — confidence distribution this week vs baseline.
#    KS-distance flag firing? Investigate.

# 2. Query Postgres for FP rate per patient
docker exec -it fall_postgres psql -U fall_user -d fall_detection -c "
  SELECT patient_id,
         COUNT(*) FILTER (WHERE patient_confirmed = 'yes') AS true_falls,
         COUNT(*) FILTER (WHERE patient_confirmed = 'no')  AS false_pos,
         COUNT(*) FILTER (WHERE patient_confirmed = 'not_answered') AS unanswered,
         COUNT(*) AS total
    FROM fall_history
   WHERE detection_time > NOW() - INTERVAL '7 days'
GROUP BY patient_id
ORDER BY false_pos DESC;
"

# 3. If false_pos / total > 0.4 for any patient — flag for retrain conversation.

# 4. Optional: MLflow UI to look at the most recent retrain runs and decide
#    whether a new training cycle is justified.
```

---

## 5. Retrain — the admin action

See `02_fall_detection_algorithm.md` Sections 4 and 6 for the full reference. In ml_dashboard the same thing is a button:

```
ml_dashboard (port 8004)

  Current Production model:  v3 (registered as fall_detector v124, deployed 2026-04-15)
  Active in inference-server: v3 (matches Production — good)

  ──────────────────────────────────────────────────────────
   [ Retrain now ]   ◄── triggers retrain.py as subprocess
                          progress streamed to a panel below
  ──────────────────────────────────────────────────────────

  Recent runs (from MLflow):
   #126  2026-04-29 09:14   v3-retrain   recall=0.92  f1=0.89  AUC=0.97   None
   #125  2026-04-22 11:00   v3-retrain   recall=0.91  f1=0.90  AUC=0.97   Staging
   #124  2026-04-15 08:30   v3-retrain   recall=0.91  f1=0.90  AUC=0.97   Production  ◄ active

  [ Promote #126 to Staging ]   [ Promote to Production ]   [ Hot-swap ]
```

Acceptance criteria when comparing a candidate to the current Production:
- **Recall must not regress.** A 2-pt recall drop is a reject regardless of other gains.
- **F1 must not drop more than 2 pts.**
- **AUC must not regress more than 1 pt.**
- **Per-patient breakdown** — recall on each individual patient should not regress catastrophically. A model that's better on average but worse on one specific high-risk patient is a regression.

If the candidate passes — promote to Staging, run a smoke test (Section 9.3), then promote to Production. Then hot-swap.

The promotion itself is mechanical:

```powershell
# Programmatically (subset of ml_dashboard internals):
import mlflow
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="fall_detector", version=126, stage="Production",
    archive_existing_versions=True,
)

# Hot-swap (or use ml_dashboard button):
.\local_dev\dev_scripts\switch_model.ps1 -Stage Production
```

---

## 6. Monitoring health — what to watch for

### 6.1 Pod-level (server_health + kubectl)

| Pod | Expected | Red flag |
|-----|----------|----------|
| inference-server | Running, /health 200 | CrashLoopBackOff → likely model load failure or DB connection |
| fall-dashboard | Running, /health 200 | Disconnect from MQTT broker (logs show retry loop) |
| postgres | StatefulSet, 1/1 ready | PVC unbound → StorageClass issue |
| mqtt-broker | Running | Logs show "auth failed" → check secret |
| minio | StatefulSet, 1/1 ready | Same as postgres for PVC |
| mlflow | Running | OOM (logs end with "Killed") → bump resource limit |
| prometheus, grafana | Running | Less critical; recoverable |
| Jobs (alembic-migrate, create-mlflow-bucket) | Completed | Failed → check job logs; first install only |

### 6.2 Application-level (Grafana)

| Metric | Normal | Red flag |
|--------|--------|----------|
| `fall_detection_requests_total` rate | ~1 req/9s × N patients | 0 → mobile apps disconnected; spike → test harness or replay |
| `fall_detection_request_latency_seconds` p99 | < 200 ms | > 500 ms → CPU starved, or model is mis-sized |
| `fall_detection_errors_total{type=*}` | 0 | non-zero → scrape the type label, check logs |
| `fall_detection_confidence` distribution | matches v3 baseline | KS-distance > 0.15 → input drift |
| `falls_today` (SQL panel) | within historical bounds | 50× normal → FP storm |

### 6.3 Database-level (psql)

```sql
-- Inference log size — should grow ~1 row per /predict
SELECT COUNT(*) FROM inference_log WHERE detection_time > NOW() - INTERVAL '1 hour';

-- Are observation_ids JOIN-able to fall_history? (sanity check the cross-reference)
SELECT COUNT(*)
  FROM inference_log il
  LEFT JOIN fall_history fh ON fh.observation_id = il.observation_id
 WHERE il.fall_detected = TRUE
   AND il.detection_time > NOW() - INTERVAL '1 day'
   AND fh.id IS NULL;
-- Should be near 0.  Non-zero means MQTT publishes are not reaching fall_dashboard.

-- Feature_snapshot count — N features × N inferences
SELECT COUNT(*) FROM feature_snapshot WHERE created_at > NOW() - INTERVAL '1 hour';
```

---

## 7. Deploy — admin action

See `01_k8s.md` Sections 6 and 9. In short:

```powershell
$REGISTRY = "registry.focus.example.com"
$NEW_TAG  = "v1.1.0"

# 1. Build + push the changed image(s)
docker build -f inference_server/Dockerfile -t $REGISTRY/inference-server:$NEW_TAG .
docker push $REGISTRY/inference-server:$NEW_TAG

# 2. Roll forward
helm upgrade fall-detection ./helm/fall-detection `
  --namespace mcs-fall-detection `
  --reuse-values `
  --set images.inferenceServer.tag=$NEW_TAG

# 3. Watch rollout
kubectl rollout status deployment/inference-server -n mcs-fall-detection

# 4. Verify
kubectl logs -n mcs-fall-detection job/alembic-migrate   # if schema change
curl https://<ingress>/predict                            # 405 expected
```

Helm's hook system runs Alembic on every upgrade automatically.

---

## 8. Incident playbook

### 8.1 inference-server is down

```powershell
# 1. What does kubectl say?
kubectl get pods -n mcs-fall-detection
kubectl describe pod inference-server-... -n mcs-fall-detection

# 2. Logs
kubectl logs inference-server-... -n mcs-fall-detection --tail=200

# Common causes:
#   - Model load failure (corrupt .pkl, missing feature_names.json)
#       → fix the file in MinIO or fall back to a bundled model/ version
#   - DB connection failure (Postgres restarted, password mismatch)
#       → check secret, restart Postgres
#   - OOM
#       → bump resource limit in values.yaml, helm upgrade
```

### 8.2 Confidence distribution drifted

The KS-distance flag fires in Grafana. Steps:

1. Confirm the drift is real — last 24h vs baseline.
2. Check if a wearable batch changed (BLE characteristic IDs, calibration). Talk to SmarKo.
3. If it's a real input drift: trigger retrain.
4. If it's a single bad patient: investigate that patient's wearable specifically.

### 8.3 False-positive storm

`falls_today` is 50× normal. Steps:

1. Check `fall_history` — is the spike across patients (system issue) or one patient (wearable / input issue)?
2. If across patients — recent model promotion? Hot-swap back to the previous Production version.
3. If one patient — investigate the wearable. Possibly disable the patient's mobile-app feed temporarily and add their data to the next retrain.

### 8.4 Missed fall (clinical report)

A caregiver reports the system did not flag a real fall. Steps:

1. Get the timestamp + patient_id from the report.
2. `SELECT * FROM inference_log WHERE patient_id=X AND detection_time BETWEEN ...` — was there a /predict call near that time?
3. If yes and `fall_detected=false` — look at the feature_snapshot. Was confidence near the threshold? Could a lower threshold have caught it?
4. If yes and confidence was very low — the model genuinely missed. Add this as a labelled positive in retraining.
5. If no — there was no /predict at that time, so the mobile app or wearable was disconnected. Investigate.

### 8.5 Pod restarted, model reverted

After an inference-server pod restart, hot-swapped model is gone. Steps:

1. The pod loaded `MODEL_VERSION` from the chart ConfigMap. That's the persistent default.
2. If the swap should have been permanent: `helm upgrade --set inferenceServer.modelVersion=<new>`.
3. If the swap was just for an experiment: re-hot-swap on the new pod.

### 8.6 Caregiver SSE not delivering

Symptoms: dashboard never flags, even though `fall_history` has rows.

1. `kubectl logs fall-dashboard-...` — does it log SSE connections?
2. CORS issue? Check browser DevTools network tab.
3. Ingress rule misconfigured for SSE? Traefik handles it natively, but if you switched to nginx, the SSE annotations may be missing — see `01_k8s.md` Section 7.

---

## 9. Rollback procedures

### 9.1 Rollback a model

Three options, fastest first:

```powershell
# 1 — Hot-swap to a bundled file version (instant)
.\local_dev\dev_scripts\switch_model.ps1 -Stage v3

# 2 — Hot-swap to a previous Registry version
curl -X POST http://<ingress>/model/switch `
  -H "X-API-Key: $env:INFERENCE_API_KEY" `
  -d '{"model_version": "<previous-stage-name>", "source": "registry"}'

# 3 — Full chart rollback (slowest but persistent)
helm rollback fall-detection <revision> -n mcs-fall-detection
```

### 9.2 Rollback a chart upgrade

```powershell
helm history fall-detection -n mcs-fall-detection
helm rollback fall-detection <revision> -n mcs-fall-detection
```

Migrations are NOT auto-reverted on rollback. If the bad version added a column, the column stays. Schema migrations should be backwards-compatible; we've kept Alembic migrations additive only.

### 9.3 Smoke test after any change

```powershell
# 1. Inference reachable
curl -X POST http://<ingress>/predict -H "X-API-Key: $key" -d '<known-fall-window>.json'
# Expect fall_detected=true on the known-fall window

# 2. MQTT reachable
mosquitto_pub -h <mqtt-host> -p 1883 -t fall/alert/test-patient -m '{"observation_id":"test","patient_id":"test","fall_detected":true,"patient_confirmed":"yes","needs_help":true,...}'

# 3. SSE delivers
curl -N http://<ingress>/api/stream
# Should print the test event within seconds
```

---

## 10. The admin's environment / tooling

| Tool | Purpose |
|------|---------|
| `kubectl` | All cluster operations |
| `helm` | Install / upgrade / rollback |
| `docker` | Build images |
| `python` 3.11+ + `retrain/requirements.txt` | Run retraining |
| `psql` (or `docker exec -it fall_postgres psql`) | DB queries |
| `mosquitto_pub` / `mosquitto_sub` | MQTT debugging |
| `curl` | API calls |
| `mlflow` CLI (optional) | UI is usually enough |

The admin's ml_dashboard runs locally on their laptop — it is NOT containerised. It calls the ingress URL of the cluster. The reasoning: ml_dashboard performs sensitive operations (retrain, promote, hot-swap), so we keep it off the production cluster and behind whatever local auth the admin's laptop has.

---

## 11. What the admin does NOT do

| Action | Why not |
|--------|---------|
| See or interact with the patient-facing popup | Admin role is mutually exclusive with caregiver role |
| Read patient demographics or biosignals from FOCUS systems | Out of scope; access only via FOCUS UI under their RBAC |
| Approve / dismiss caregiver alerts | Caregiver role |
| Manually edit `fall_history` rows | Read-only conventions; corrections via supported tooling only |
| Change MQTT topic structure or schema without coordinating | Mobile app + dashboard depend on the contract; coordinate first |

---

## 12. Step-by-step from the admin's perspective — a representative week

```
Mon 09:00  Morning glance: server_health all green. Grafana ml_server_overview shows
            request rate ≈ historical baseline. Falls today: 4. Within bounds.
            No action.
Tue 09:00  Same as Mon. server_health green.
Tue 14:00  Slack: clinical lead reports patient 005 had a fall at 14:32 yesterday
            that the system did not flag.
            Admin queries Postgres. Inference_log shows /predict calls every 9s
            for patient 005 around that time, but no fall_detected=true.
            Admin pulls the highest-confidence window in that 5-minute range:
            confidence = 0.42 — under the 0.5 threshold.
            Admin opens an issue: "consider retrain with patient 005's data and
            possibly threshold tuning". Seeds the data into Postgres for the
            next retrain.
Wed 09:00  Same as Mon. No action.
Wed 11:00  Weekly review. Per-patient FP query:
              patient_007 — 12 false positives this week.
            Admin investigates wearable 007 (Slack to SmarKo). Turns out a
            firmware update was rolled out to that wearable; the calibration
            constants drifted.
            Admin disables patient_007's data temporarily by adding a deny
            rule to the mobile-app config (out of scope here — mobile-app side).
Thu 09:00  server_health: alembic-migrate shows in red.
            kubectl describe job/alembic-migrate — last upgrade re-ran the
            migration job and it didn't clean up. Job status is "Failed" but
            schema is correct.
            Admin deletes the job manually.
              kubectl delete job/alembic-migrate -n mcs-fall-detection
            Server_health goes green.
Thu 15:00  Triggers a retrain (on the data-set that includes patient 005's
            missed fall). MLflow run #127 completes. recall = 0.93 (+0.02
            over current Production), F1 = 0.90 (no regression).
            Promotes #127 to Staging.
Fri 09:00  Smoke test on Staging via a copy of the historical labelled set.
            Looks good. Promotes to Production.
            Hot-swaps inference-server.
Fri 11:00  Watches Grafana model_performance for 30 min. Confidence distribution
            looks normal. Done.
Fri 17:00  Documents the week's actions in the team's running notes.
```

---

## 13. Open admin decisions

Things we'd like the next admin to decide / inherit:

1. **Retrain cadence** — currently manual. Do we want a quarterly minimum even with no signals? A nightly auto-retrain (CronJob)? My preference: quarterly + on-signal until you have enough data for nightly to be meaningful.
2. **Alert routing** — Grafana alerts are not wired to a pager (PagerDuty / on-call). Wiring them is a small change. Decide on the channel (Slack? PagerDuty? email?) and configure.
3. **Acknowledge feature for caregiver dashboard** — small server-side state change. Adds value. Currently flags auto-expire by time only.
4. **Per-patient threshold** — extension. If a patient has chronic FPs, we could raise their personal threshold. Not implemented.
5. **Auto-rollback on regression** — currently manual. Could be a CronJob that compares latest Production confidence distribution to baseline and rolls back if drifted >X%. Risky; deliberate.
6. **MQTT broker auth** — currently anonymous. Production must have user/pass or mTLS before go-live.
7. **API key rotation policy** — how often, who issues, who revokes. TBD.

---

## 14. Cross-references

- [`01_k8s.md`](01_k8s.md) — install / upgrade / rollback commands
- [`02_fall_detection_algorithm.md`](02_fall_detection_algorithm.md) — model + retraining + version mgmt
- [`03_fall_detection_system.md`](03_fall_detection_system.md) — system architecture
- [`05_web_app_integration.md`](05_web_app_integration.md) — what the caregiver dashboard does
- [`07_user_flow_caregiver.md`](07_user_flow_caregiver.md) — the role admin hand-offs to
- Existing handover docs: `handover_docs/ADMIN_ml_ops_related.md`, `handover_docs/ADMIN_runbook_retrain_and_hotswap.md`, `handover_docs/ADMIN_data_storage_map.md` — earlier drafts; still accurate

---

## 15. Contact

| For | Reach out to |
|-----|--------------|
| Anything ML-related (training, MLflow, hot-swap) | Hayate (MCS) — primary handover contact |
| Cluster, ingress, networking | FOCUS DevOps |
| Mobile app upstream behaviour | Isa (SmarKo) |
| Clinical / patient-population questions | Charite |
