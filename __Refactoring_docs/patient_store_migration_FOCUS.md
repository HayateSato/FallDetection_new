# Migrating the Patient Store to FOCUS's System (MySQL or FHIR)

**Audience:** FOCUS engineers integrating the Fall Dashboard into their existing
infrastructure.
**Goal:** make the Fall Dashboard read patient information (id, name, age,
gender, device/MAC) from FOCUS's own patient database — either **MySQL** or an
**FHIR** server — instead of the bundled SQLite file.

> For MCS local testing you do **not** need a real FOCUS database. This document
> only explains *what to change*. You can develop against the existing SQLite
> store and swap the backend at deployment.

---

## 1. The key idea: one module is the only thing that touches patient storage

The dashboard talks to patient storage through a **single module**:

```
_6G_Integration_v2_mqtt/fall_dashboard/patient_store.py
```

Nothing else in the dashboard issues a patient query. `db.py`, `web.py` and
`main.py` only call the **functions** that module exposes. So migrating the
backend = **rewrite `patient_store.py` to talk to MySQL or FHIR, keeping the
same function signatures and return shapes.** No other file needs to change.

This is the contract you must preserve. If your replacement returns the same
shapes, the rest of the system (API, frontend, MQTT/SSE, InfluxDB fall history)
keeps working unchanged.

### Functions the dashboard depends on

| Function | Called by | Must return / do |
|----------|-----------|------------------|
| `init_db()` | `main.py` at startup | Prepare the connection / verify reachability. May be a no-op for a remote DB. **Must not raise** if the backend is simply empty. |
| `sync_from_env()` | `main.py` at startup | Optional in production. With an external source of truth (FOCUS DB), this can be a **no-op** — you no longer seed from `PATIENT_IDS`. See §4. |
| `list_patients()` | `db.py` → `/api/patients` | `list[dict]`, one per patient, with **exactly** these keys (see §2). |
| `get_mac_map()` | `main.py` | `dict` of `{patient_id: mac_id}`. |
| `upsert_patient(...)` | future admin/runtime use | Insert/update one patient. Can raise `NotImplementedError` if FOCUS owns writes (read-only integration). |

---

## 2. The data contract (do not change these keys)

`list_patients()` MUST return a list of dicts shaped like this — the frontend
(`dashboard/app.js`) and `db.py` read these exact keys:

```python
{
    "patient_id":     "patient_test_4",  # str  — STABLE ID. Must equal the id used
                                         #        in MQTT topic fall/alert/<patient_id>
                                         #        and the InfluxDB `patient_id` tag.
    "name":           "Jane Doe",        # str | None
    "age":            81,                # int | None
    "gender":         "female",          # str | None
    "mac_id":         "6c:1d:eb:04:a9:d9",# str  ("" if unknown) — sensor device id
    "session_active": True,              # bool — whether to treat as currently monitored
}
```

`db.py` adds `"fall_count"` afterwards (from InfluxDB) — **you do not provide
it.** Provide the five identity fields above; leave counts to InfluxDB.

### ⚠️ The single most important field: `patient_id`

`patient_id` is the join key across the whole system:
- MQTT alert topic: `fall/alert/<patient_id>`
- InfluxDB tag: `patient_id`
- The mobile/mock app's `PATIENT_IDS`

When you migrate, **the id you expose from FOCUS's DB must be the same id the
sensor/mobile-app side uses.** If FOCUS's primary key is a different value
(e.g. a numeric DB id or an FHIR resource UUID), you must map FOCUS's id to the
sensor-side `patient_id`. Decide this mapping early — it is the thing most
likely to break the integration. Options:
- Store the sensor `patient_id` as a column / FHIR identifier on the patient
  record, and expose **that** as `patient_id`; or
- Keep a small mapping table (FOCUS id ↔ sensor patient_id).

---

## 3. Option A — MySQL backend

MySQL is the closest swap to today's SQLite. Steps:

### 3.1 Add the driver
In `fall_dashboard/requirements.txt` add one of:
```
PyMySQL            # pure-python, simplest
# or mysqlclient   # C extension, faster, needs build deps in the Dockerfile
```

### 3.2 Rewrite `patient_store.py` connection + queries
Replace the `sqlite3` connection with a MySQL connection. Read connection info
from env vars (add them to `docker-compose.yml` under `fall-dashboard`):

```
MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
```

Map each function to a SQL query against FOCUS's patient table. Example skeleton
(adjust table/column names to FOCUS's schema):

```python
import os, pymysql

def _connect():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DB"],
        cursorclass=pymysql.cursors.DictCursor,
    )

def init_db():
    # Just verify connectivity; FOCUS owns the schema. Do NOT CREATE TABLE.
    with _connect() as c:
        c.ping(reconnect=True)

def list_patients():
    with _connect() as c, c.cursor() as cur:
        cur.execute("""
            SELECT  focus_patient_key  AS patient_id,   -- <-- map to sensor id!
                    full_name          AS name,
                    age,
                    gender,
                    device_mac         AS mac_id,
                    is_active          AS session_active
            FROM    patients
        """)
        rows = cur.fetchall()
    return [{
        "patient_id":     str(r["patient_id"]),
        "name":           r.get("name"),
        "age":            r.get("age"),
        "gender":         r.get("gender"),
        "mac_id":         r.get("mac_id") or "",
        "session_active": bool(r.get("session_active", 1)),
    } for r in rows]

def get_mac_map():
    return {p["patient_id"]: p["mac_id"] for p in list_patients()}

def sync_from_env():
    pass            # FOCUS DB is the source of truth; no env seeding

def upsert_patient(*args, **kwargs):
    raise NotImplementedError("Patients are managed in FOCUS's system")
```

### 3.3 Remove the SQLite volume
In `caregiver_layer/docker-compose.yml`, the `patient_db` named volume and the
`/app/data` mount + `PATIENT_DB_PATH` env are no longer needed — delete them and
add the `MYSQL_*` env vars instead. Ensure the dashboard container can reach the
MySQL host on the network.

### 3.4 What you do NOT change
`db.py`, `web.py`, `main.py`, `dashboard/*`, MQTT, InfluxDB — untouched.

---

## 4. Option B — FHIR backend

FHIR is a REST API, not SQL. The patient demographics map to the **`Patient`**
resource; the sensor/device maps to **`Device`** (and optionally a link between
them). You replace the SQLite calls with HTTP calls to the FHIR server.

### 4.1 Add the client
In `requirements.txt`:
```
requests            # or fhirclient / fhir.resources for typed models
```

### 4.2 Field mapping: FHIR `Patient` → our contract

| Our key | FHIR source |
|---------|-------------|
| `patient_id` | An `identifier` on the `Patient` whose `system` is the **sensor id namespace** (preferred), OR `Patient.id`. **Must equal the sensor-side id** — see §2 warning. |
| `name` | `Patient.name[0]` → `given` + `family` (join into a display string) |
| `age` | Derive from `Patient.birthDate` (compute years), or read an Age extension if FOCUS uses one |
| `gender` | `Patient.gender` (`male` / `female` / `other` / `unknown`) |
| `mac_id` | From the linked `Device` — `Device.identifier` (MAC) where `Device.patient` references this Patient, or a `Device.udiCarrier`. |
| `session_active` | `Patient.active`, or presence of an active `Encounter` / `CarePlan` if FOCUS models sessions that way |

### 4.3 Rewrite `patient_store.py` against FHIR

```python
import os, requests
from datetime import date

FHIR_BASE  = os.environ["FHIR_BASE_URL"]          # e.g. https://focus.example/fhir
FHIR_TOKEN = os.getenv("FHIR_BEARER_TOKEN", "")    # if OAuth/SMART is required

def _headers():
    h = {"Accept": "application/fhir+json"}
    if FHIR_TOKEN:
        h["Authorization"] = f"Bearer {FHIR_TOKEN}"
    return h

def init_db():
    requests.get(f"{FHIR_BASE}/metadata", headers=_headers(), timeout=10).raise_for_status()

def _age_from_birthdate(bd):
    if not bd: return None
    y = date.today().year - int(bd[:4])
    return y

def list_patients():
    # Adjust the search params to FOCUS's cohort (e.g. by organization/careteam)
    r = requests.get(f"{FHIR_BASE}/Patient", params={"active": "true", "_count": "200"},
                     headers=_headers(), timeout=15)
    r.raise_for_status()
    bundle = r.json()
    out = []
    for entry in bundle.get("entry", []):
        p = entry["resource"]
        name = ""
        if p.get("name"):
            n = p["name"][0]
            name = " ".join(n.get("given", []) + [n.get("family", "")]).strip()
        out.append({
            "patient_id":     _sensor_id_from_patient(p),     # <-- you implement the mapping
            "name":           name or None,
            "age":            _age_from_birthdate(p.get("birthDate")),
            "gender":         p.get("gender"),
            "mac_id":         _mac_for_patient(p["id"]),       # <-- query linked Device
            "session_active": p.get("active", True),
        })
    return out
```

You implement two helpers:
- `_sensor_id_from_patient(patient_resource)` — extract the identifier that
  equals the sensor-side `patient_id` (see §2).
- `_mac_for_patient(fhir_patient_id)` — `GET /Device?patient=<id>` and pull the
  MAC identifier; return `""` if none.

`get_mac_map`, `sync_from_env`, `upsert_patient` follow the MySQL pattern
(sync = no-op; upsert = `NotImplementedError`, or a `PUT /Patient` if FOCUS
allows writes).

### 4.4 Env / compose
Add `FHIR_BASE_URL` (and auth vars) to the `fall-dashboard` service env. Remove
the SQLite `patient_db` volume + `PATIENT_DB_PATH`. Ensure network egress to the
FHIR server.

### 4.5 Performance note
`list_patients()` is called on `/api/patients` (page load) and after each
caregiver-alerting fall event. For FHIR, avoid N extra `Device` calls per
patient on every request — either fetch Devices in one bundled search and join
in memory, or add a short in-process cache (e.g. 30 s TTL). Not needed for SQLite/MySQL.

---

## 5. Migration checklist

- [ ] Decide the **`patient_id` mapping** (sensor id ↔ FOCUS id). This is the
      critical one.
- [ ] Pick backend: MySQL (§3) or FHIR (§4).
- [ ] Add the driver/client to `fall_dashboard/requirements.txt`.
- [ ] Rewrite `fall_dashboard/patient_store.py` preserving the function
      signatures and the dict shape in §2.
- [ ] Add backend connection env vars to `caregiver_layer/docker-compose.yml`
      under `fall-dashboard`.
- [ ] Remove the SQLite bits: `patient_db` volume, `/app/data` mount,
      `PATIENT_DB_PATH` env (only relevant once you fully cut over).
- [ ] `sync_from_env()` → no-op (FOCUS DB is source of truth).
- [ ] Decide write policy: read-only (`upsert_patient` raises) vs. dashboard may
      create patients.
- [ ] Verify against a real FOCUS endpoint at deployment (out of scope for MCS
      local testing).

## 6. What stays the same regardless of backend

- The HTTP API (`/api/patients`, `/api/falls`, `/api/stream`) and the frontend.
- Fall **events** and **counts** — still InfluxDB, never moved into the patient DB.
- MQTT alert flow and SSE fan-out.
- `db.py`, `web.py`, `main.py` — no edits needed if `patient_store.py` honours
  the contract.

---

## Related docs

- `__Refactoring_docs/patient_store_sqlite.md` — the current SQLite
  implementation, schema, and the who/how-to-add-users discussion.
- `__Refactoring_docs/postgres.md` — the earlier Postgres setup (MCS inference
  layer still uses Postgres for `inference_log` / `feature_snapshot`; unaffected).
