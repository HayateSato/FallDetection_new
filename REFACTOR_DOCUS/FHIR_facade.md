# FHIR Output — Design Decision

## Decision: no active FHIR push, no FHIR column in Postgres

**Decided: 2026-04-17**

We will not implement active FHIR pushes from the inference server to an external FHIR server,
and we will not store FHIR JSON as a column in Postgres alongside the normalized fall data.

Postgres (`inference_log`, `fall_history`, `feature_snapshot`) is the single source of truth.
FHIR is an output format, not a storage format.

---

## Why not store FHIR JSON in Postgres (rejected option)

Storing a `fhir_json` blob column in `fall_history` would mean keeping the same data twice —
once normalized across multiple tables, and once serialized as a FHIR JSON blob. Problems:

- Any schema change (new column, renamed field) must be applied in two places
- The JSON blob goes stale if the Postgres rows are updated
- Wastes storage and makes queries more complex
- Provides no benefit over generating FHIR on the fly from Postgres

**Decision: won't do. See facade pattern below instead.**

---

## Why not push to FOCUS FHIR server on every /predict (deferred, not rejected)

The inference server already supports this. If `FHIR_SERVER_URL` is set in `.env`,
it fires a BackgroundTask after each `/predict` response:

```
mobile app → POST /predict → inference_server
                                    │
                          HTTP response sent immediately
                                    │ (BackgroundTask — does NOT block response)
                                    ▼
                          POST FHIR Observation → FOCUS FHIR server
```

Because it uses `BackgroundTask`, the mobile app's latency is not affected.

However, this adds an outbound HTTP call per prediction to an external server. At scale
(many patients sending predictions in parallel) this becomes a reliability concern:
- If the FHIR server is slow or down, background tasks queue up
- Retries and error handling add complexity
- Network failures need to be monitored separately from inference failures

**Decision: `FHIR_SERVER_URL` stays blank until FOCUS confirms they have a running FHIR server
and explicitly need us to push to it. The FHIR Observation is already included in the
`/predict` HTTP response body — FOCUS can read it from there without a push.**

---

## If FOCUS later needs historical FHIR data: FHIR facade

If FOCUS wants to query past fall detections in FHIR format, implement a facade endpoint
in `fall_dashboard` that reads from Postgres and serializes to FHIR R4 on the fly:

```
FOCUS queries → GET /fhir/Observation?patient={id}
                        │
                        ▼
              fall_dashboard reads fall_history from Postgres
                        │
                        ▼
              serializes rows → FHIR R4 Bundle of Observations
                        │
                        ▼
              returns JSON to FOCUS
```

No schema change needed. No duplicate storage. FHIR format can evolve independently
of the Postgres schema. This is the standard pattern in healthcare IT for systems
that use a relational DB internally but expose FHIR externally.

**Implementation when needed:**
- Add `GET /fhir/Observation` route to `fall_dashboard/web.py`
- Read from `fall_history` via `db.list_falls()`
- Serialize each row to a FHIR R4 `Observation` resource (same format already
  generated inside `inference_server/server.py` per `/predict` call)
- Return as a FHIR `Bundle`

---

## Current state of FHIR in the codebase

| Component | What it does |
|-----------|-------------|
| `inference_server/server.py` | Generates a FHIR R4 Observation per prediction; returns it in the `/predict` response body. Optionally POSTs to `FHIR_SERVER_URL` via BackgroundTask if the env var is set. |
| `mock_focus/fhir_server.py` | Mock FHIR server (port 8003) for local dev. Serves synthetic Patient + Observation resources. Never ships to K8s — replace with real FOCUS FHIR URL. |
| `.env` | `FHIR_SERVER_URL=` (blank — push disabled). `FHIR_PUSH_ON_FALL_ONLY=true`. |

---

## Open questions for FOCUS (todo.md Step 5)

- Does FOCUS have a running FHIR server? What is the URL?
- Do they want us to push Observations to it, or is reading from `/predict` response sufficient?
- Does LOINC code `72514-3` (fall risk) pass their FHIR validator?
