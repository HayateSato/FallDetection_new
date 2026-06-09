"""
Patient store — lightweight SQLite backing for the caregiver dashboard.
=======================================================================

Replaces the old Postgres `participant_session` table. SQLite needs no server
process, no extra container and no network port — it is a single file on a
mounted volume, read/written via Python's stdlib `sqlite3`. This is a good fit
for the low-power hardware the FOCUS partner runs.

What lives here
---------------
Patient identity + (future) demographics. Fall *events* and fall *counts* still
live in InfluxDB — this store never duplicates them.

Schema (forward-compatible with deployment):
    patient_id     TEXT PRIMARY KEY   — stable ID, matches MQTT topic + InfluxDB tag
    name           TEXT               — NULL for now; populated in deployment
    age            INTEGER            — NULL for now
    gender         TEXT               — NULL for now
    mac_id         TEXT               — sensor MAC (from MAC_IDS env var)
    session_active INTEGER DEFAULT 1  — 1 = currently monitored
    created_at     TEXT               — first-seen timestamp (UTC)

Config-file workflow ("edit .env, recreate container to apply")
---------------------------------------------------------------
On startup the dashboard calls ``sync_from_env()``. It UPSERTS every id in
PATIENT_IDS (with its positional MAC from MAC_IDS). Patients already in the DB
are never deactivated or deleted by sync — they are kept untouched. To change a
patient's MAC, edit MAC_IDS and recreate the container; to drop a patient,
remove the row manually.
"""

import logging
import os
import sqlite3
from typing import List, Optional

logger = logging.getLogger(__name__)

# DB path is configurable so the container can point it at a mounted volume.
# Default lives under /app/data which the compose file mounts as a named volume.
DB_PATH = os.getenv("PATIENT_DB_PATH", os.path.join("data", "patients.db"))


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Open a connection with row access by column name and FK enforcement."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the patients table if it does not exist. Idempotent."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id     TEXT PRIMARY KEY,
                name           TEXT,
                age            INTEGER,
                gender         TEXT,
                mac_id         TEXT,
                session_active INTEGER NOT NULL DEFAULT 1,
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
    logger.info(f"Patient store ready at {DB_PATH}")


# ---------------------------------------------------------------------------
# Sync from env (config-file workflow)
# ---------------------------------------------------------------------------

def sync_from_env() -> None:
    """
    Upsert PATIENT_IDS / MAC_IDS into the patients table.

    - New ids are inserted (session_active=1).
    - Existing ids keep their row; only mac_id is refreshed when MAC_IDS
      provides a (non-empty) value, so editing MAC_IDS in .env takes effect on
      the next container recreate.
    - Patients absent from PATIENT_IDS are left untouched (never deactivated or
      deleted) — per the chosen "keep, never touch" policy.
    """
    patient_ids = [p.strip() for p in os.getenv("PATIENT_IDS", "").split(",") if p.strip()]
    mac_list    = [m.strip() for m in os.getenv("MAC_IDS", "").split(",")     if m.strip()]
    mac_map     = {pid: mac_list[i] for i, pid in enumerate(patient_ids) if i < len(mac_list)}

    if not patient_ids:
        logger.info("sync_from_env: PATIENT_IDS empty — nothing to sync")
        return

    with _connect() as conn:
        for pid in patient_ids:
            mac = mac_map.get(pid, "")
            # Insert new patient, or refresh mac_id for an existing one.
            # COALESCE keeps the existing mac when the new value is empty.
            conn.execute(
                """
                INSERT INTO patients (patient_id, mac_id)
                VALUES (?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    mac_id = COALESCE(NULLIF(excluded.mac_id, ''), patients.mac_id)
                """,
                (pid, mac),
            )
    logger.info(f"sync_from_env: upserted {len(patient_ids)} patient(s) from PATIENT_IDS")


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def list_patients() -> List[dict]:
    """
    Return all patients in the store (active and inactive), newest first.

    Fall counts are NOT included here — db.list_patients() joins them from
    InfluxDB so this store stays free of event data.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT patient_id, name, age, gender, mac_id, session_active
            FROM patients
            ORDER BY created_at DESC, patient_id ASC
            """
        ).fetchall()
    return [
        {
            "patient_id":     r["patient_id"],
            "name":           r["name"],
            "age":            r["age"],
            "gender":         r["gender"],
            "mac_id":         r["mac_id"] or "",
            "session_active": bool(r["session_active"]),
        }
        for r in rows
    ]


def get_mac_map() -> dict:
    """Return {patient_id: mac_id} for all patients (used by the SSE layer)."""
    with _connect() as conn:
        rows = conn.execute("SELECT patient_id, mac_id FROM patients").fetchall()
    return {r["patient_id"]: (r["mac_id"] or "") for r in rows}


# ---------------------------------------------------------------------------
# Write API (for future runtime management / deployment use)
# ---------------------------------------------------------------------------

def upsert_patient(
    patient_id: str,
    name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    mac_id: Optional[str] = None,
    session_active: bool = True,
) -> None:
    """Insert or update a single patient. Only non-None fields overwrite."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO patients (patient_id, name, age, gender, mac_id, session_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                name           = COALESCE(excluded.name,   patients.name),
                age            = COALESCE(excluded.age,    patients.age),
                gender         = COALESCE(excluded.gender, patients.gender),
                mac_id         = COALESCE(NULLIF(excluded.mac_id, ''), patients.mac_id),
                session_active = excluded.session_active
            """,
            (patient_id, name, age, gender, mac_id or "", 1 if session_active else 0),
        )


def delete_patient(patient_id: str) -> bool:
    """Delete a patient from the store. Returns True if a row was deleted."""
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info(f"Deleted patient {patient_id!r}")
    else:
        logger.warning(f"delete_patient: no row found for {patient_id!r}")
    return deleted
