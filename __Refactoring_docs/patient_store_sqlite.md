# Patient Store — SQLite (replaces Postgres `participant_session`)

**Date:** 2026-06-04
**Layer:** caregiver_layer (FOCUS) — `fall_dashboard` service only
**Status:** implemented, not yet committed at time of writing

---

## Why this change

Postgres was removed from the caregiver layer (commits `8c2d202`, `2e2670d`,
`1ac0d21`). The dashboard then fell back to reading the patient list from the
`PATIENT_IDS` env var, parsed **once at module import**. Two problems:

1. The list was frozen at process start, so editing `.env` and `docker restart`
   did **not** update the displayed patients (only a full container recreate
   re-injected env, and even then the env-var design carried no patient
   metadata).
2. There was no place to store per-patient data (name, age, gender) that the
   old `participant_session` table used to hold.

The FOCUS partner runs low-power hardware, so a full Postgres server is overkill.
**SQLite** is the right weight: a single file, no server process, no extra
container, no network port, and `sqlite3` is in the Python stdlib (zero new
dependency). It comfortably handles a single-writer dashboard at this scale.

This replaces *patient identity* storage only. Fall **events** and fall **counts**
still live in InfluxDB — the SQLite store never duplicates them.

---

## What changed (files)

| File | Change |
|------|--------|
| `fall_dashboard/patient_store.py` | **New.** SQLite layer: schema, `init_db()`, `sync_from_env()`, `list_patients()`, `get_mac_map()`, `upsert_patient()`. |
| `fall_dashboard/db.py` | `list_patients()` now reads live from SQLite (joining InfluxDB fall counts) instead of a frozen `PATIENT_IDS` global. Fixes the stale-list bug. |
| `fall_dashboard/main.py` | On startup calls `patient_store.init_db()` + `sync_from_env()`; MAC map sourced from the store. |
| `fall_dashboard/web.py` | `/api/patients` treats the store's `mac_id` as source of truth (env `mac_map` is fallback only). |
| `caregiver_layer/docker-compose.yml` | Added `PATIENT_DB_PATH=/app/data/patients.db` env + named volume `patient_db` mounted at `/app/data`. |

The dashboard **frontend was not changed** — `app.js` already reads
`patient_id` / `fall_count` / `mac_id` from `/api/patients`.

---

## Schema

Table `patients` in `patients.db`:

| Column | Type | Now | Notes |
|--------|------|-----|-------|
| `patient_id` | TEXT PK | used | Stable ID; matches MQTT topic + InfluxDB `patient_id` tag |
| `name` | TEXT | NULL | Reserved for deployment |
| `age` | INTEGER | NULL | Reserved for deployment |
| `gender` | TEXT | NULL | Reserved for deployment |
| `mac_id` | TEXT | used | Sensor MAC (from `MAC_IDS` env, positional) |
| `session_active` | INTEGER | always 1 | 1 = monitored. Currently never set to 0 (see policy) |
| `created_at` | TEXT | auto | UTC first-seen timestamp |

`patient_confirmed` int encoding (carried in InfluxDB / SSE, not stored here):
`1` = yes, `0` = no, `-1` = not answered.

---

## Where the DB lives

- **In container:** `/app/data/patients.db` (set by `PATIENT_DB_PATH`).
- **On host:** Docker **named volume** `patient_db` (project-prefixed:
  `caregiver_layer_patient_db`). Survives `docker compose down` and
  `--force-recreate`. Wiped only by `docker compose down -v` or
  `docker volume rm caregiver_layer_patient_db`.

---

## Current behaviour (design decisions taken)

- **Add/update patients:** edit `PATIENT_IDS` (+ `MAC_IDS`) in `.env`, then
  recreate the container. `sync_from_env()` upserts them. ("Config file,
  restart to apply.")
- **Removal policy — "keep, never touch":** a patient dropped from
  `PATIENT_IDS` is **left in the DB, still active**. Sync only ever inserts or
  refreshes; it never deactivates or deletes. To actually remove a patient you
  must edit the DB directly (today).
- **List filter — "all patients":** `/api/patients` returns every row, active
  or not.

### Apply changes

```powershell
# from caregiver_layer/
docker compose up -d --build --force-recreate fall-dashboard
docker logs focus_fall_dashboard | Select-String "Patient store ready|upserted"
```

### Inspect / edit the DB directly

```powershell
# open a shell in the running container
docker exec -it focus_fall_dashboard python -c "import sqlite3,json; c=sqlite3.connect('/app/data/patients.db'); c.row_factory=sqlite3.Row; print(json.dumps([dict(r) for r in c.execute('SELECT * FROM patients')], indent=2))"

# delete a patient (the only current way to truly remove one)
docker exec -it focus_fall_dashboard python -c "import sqlite3; c=sqlite3.connect('/app/data/patients.db'); c.execute('DELETE FROM patients WHERE patient_id=?',('patient_test_5',)); c.commit()"
```

---

## WHO and HOW we add users — current vs. future

This is the part to revisit. Right now "adding a user" = adding a `patient_id`,
done by **whoever edits `.env`** (an operator with shell + compose access). No
auth, no UI, no demographics, restart required. That is fine for local testing
but not for deployment.

### Open questions to decide before deployment

1. **Who is a "user"?** Two distinct roles are currently conflated:
   - **Patient** — the monitored person (`patient_id`, name, age, gender, MAC).
     This is what the store holds today.
   - **Caregiver / operator** — the human viewing the dashboard. There is
     **no caregiver account model at all** yet (no login, dashboard is open on
     port 8002). If FOCUS needs per-caregiver access or audit, this is a new
     table + auth layer, not part of `patients`.

2. **How are patients added in production?** Options, lightest → richest:
   - **A. Env/config seed (today).** Edit `PATIENT_IDS` + recreate. No names.
     Operator-only. Good enough only if patient set is small and static.
   - **B. `patients.json` seed file** on the `patient_db` volume, read by
     `sync_from_env()`'s sibling `sync_from_file()`. Lets you specify
     name/age/gender per patient without code. Still restart-to-apply, still
     operator-edited. **Recommended next step** — low effort, unlocks
     demographics.
   - **C. Runtime admin API + form** (`POST /api/patients`, `DELETE
     /api/patients/{id}`). `upsert_patient()` already supports the write; needs
     endpoints + a small admin page. No restart. Requires deciding auth (who
     may call it) — otherwise anyone on the WiFi can add/remove patients.

3. **Removal semantics in production.** "Keep, never touch" preserves history
   but means the active list grows forever. For deployment likely want either:
   - soft-deactivate (`session_active=0`) when a session ends, list filtered to
     active by default; or
   - explicit delete via admin action.
   Decide this alongside option B/C.

4. **Multi-tenancy.** If one dashboard serves multiple care sites/homes, add a
   `site_id` / `org_id` column and scope queries. Not needed for single-site.

### Already in place to support the above

- `patient_store.upsert_patient(patient_id, name, age, gender, mac_id,
  session_active)` — full write path, COALESCE semantics (only non-None fields
  overwrite). Ready for an admin endpoint or a JSON seed loader.
- `name` / `age` / `gender` columns exist and are NULL-safe end to end.

### Suggested path

1. **Now:** ship SQLite as-is (operator edits `.env`, restart).
2. **Before deployment:** add option **B** (`patients.json` seed) to get
   name/age/gender without a UI.
3. **If FOCUS needs live management or access control:** add option **C** (admin
   API) *and* a caregiver auth model — treat that as a separate design.

---

## Related docs

- `__Refactoring_docs/postgres.md` — the Postgres setup this replaces (still
  describes the MCS-side `inference_log` / `feature_snapshot` tables, which are
  unaffected and remain in Postgres on the inference layer).
- `__Refactoring_docs/deployment_architecture.md` — overall layer split.
