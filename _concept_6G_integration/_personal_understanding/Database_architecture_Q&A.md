## 1. Do you need MinIO in addition to Postgres?

Not right now. MinIO is only needed when the **MLflow artifact store needs to be shared across machines** — specifically when you want the inference server to load a model directly from the MLflow registry (`mlflow.sklearn.load_model("models:/FallDetector/Production")`) rather than from a local file.

For the current phase:

- **Local dev**: MLflow stores `.pkl` files in `./mlruns/` on disk. No MinIO needed.
- **Production (Step 11)**: If the inference server and the MLflow tracking server are separate pods in Kubernetes, they can't share a local disk — that's when MinIO becomes necessary as shared object storage.

MinIO is a Step 11 concern, not something needed before end-to-end testing.

---

## 2. Who writes to which DB?

Under the current design:

**Postgres `inference_log`** ← written by **inference_server** inside `/predict` (via `BackgroundTasks` after the HTTP response is sent — so it doesn't add latency to the mobile app).

**SQLite `fall_history`** ← written by **fall_dashboard** when a confirmed alert arrives via MQTT.

So a single fall event produces two writes in two different processes. That's intentional — each DB has a different owner and a different purpose.

---

## 3. If Isa writes all this to InfluxDB, can you remove the caregiver DB?

What Isa agreed to write:

| Data | Written by |
| --- | --- |
| Raw bio-data | mobile app (already) |
| Fall detection result (confidence, model_version, timestamp) | mobile app (from HTTP response) |
| Patient confirmation (fall T/F, help_required T/F, no_response T/F, timestamp, patient UID) | mobile app |
| Window start/end time | mobile app |

On paper this covers everything `fall_history` stores. So technically yes, you could remove the caregiver DB.

**But there's a structural problem:** InfluxDB is a time-series database, not a relational one. It is optimised for `"give me all readings between T1 and T2"`, not for `"give me all confirmed falls grouped by patient with a count"`. To answer that second query from InfluxDB you'd need to:

1. Query the fall detection measurement
2. Query the patient confirmation measurement separately
3. Join them in Python by patient UID + timestamp
4. Aggregate fall counts per patient

And the dashboard does this on every page refresh. That's significantly more code and slower than one SQL query: `SELECT patient_id, COUNT(*) FROM fall_history WHERE patient_confirmed='yes' GROUP BY patient_id`.

---

## 4. Pre-joined DB vs. querying multiple sources at read time

You're describing a real architectural pattern called a **materialized view** — a DB that holds pre-joined, pre-aggregated data purely for fast reading, populated by subscribing to events.

That's actually what the caregiver DB already is:

- The mobile app writes raw data to InfluxDB (source of truth)
- The fall_dashboard subscribes to the MQTT alert → writes a joined row to its own SQLite
- The dashboard reads from SQLite with one query

The argument for keeping it:

`InfluxDB (source of truth, owned by Isa)
    │
    │  MQTT fall/alert event
    ▼
fall_dashboard
    │  one INSERT per confirmed fall
    ▼
SQLite fall_history  (read model, owned by you)
    │
    │  SELECT * FROM fall_history JOIN participant_session
    ▼
dashboard`

The argument for removing it:

`InfluxDB (source of truth)
    │
    │  3 separate measurement queries + Python join
    ▼
dashboard`

**Recommendation: keep the caregiver DB**, for three reasons:

1. **Query simplicity** — relational queries (counts, joins, filters) are SQL's strength, not InfluxDB's
2. **Decoupling** — if Isa changes her InfluxDB measurement names or field names, your dashboard doesn't break. The caregiver DB is a stable interface you control.
3. **You already have it** — the fall_dashboard already subscribes to MQTT events and writes on arrival. The sync problem essentially doesn't exist because the write happens in the same callback that receives the event.

The only real downside is that InfluxDB and the caregiver DB can theoretically diverge if the fall_dashboard is down when an alert fires. That's an acceptable trade-off for a local dashboard. In production you'd add a reconciliation query on startup — compare the last N events in InfluxDB against the caregiver DB and fill any gaps.