## Requirements files — what each is for

| File | Used by | When |
| --- | --- | --- |
| **`requirements.dev.txt`** (new) | Your Windows laptop (venv) | Local development — **this is the one to install** |
| `requirements.txt` | Docker `ml_server` container | Production — Flask client only |
| `requirements.server.txt` | Referenced by `system_operator/ml_server/requirements.txt` | Production — base for ML server container |
| `system_operator/ml_server/requirements.txt` | Docker `ml_server` container | Production — ML server + new packages |
| `caregiver/api/requirements.txt` | Docker `caregiver_api` container | Production — caregiver service only |
| `emergency/notification_service/requirements.txt` | Docker `emergency_svc` container | Production — emergency service only |

---

## What to do right now

You only need **one venv** for everything on this laptop:

`# Make sure your venv is activated, then from the project root:
pip install -r requirements.dev.txt`

---

## Why so many files then?

When running with Docker Compose, each service runs in its **own isolated container** — a container is like a mini-computer that only has what it needs. So:

- The `caregiver_api` container only installs `caregiver/api/requirements.txt` (no XGBoost, no InfluxDB — things it doesn't need)
- The `ml_server` container installs `system_operator/ml_server/requirements.txt` (no InfluxDB, no Flask)

This keeps containers small and fast. On your laptop in development, you don't need that separation — one venv with everything is fine.

**`requirements.server.txt`** — you can keep it. It's still referenced by `system_operator/ml_server/requirements.txt` via `-r ../../requirements.server.txt`. Don't delete it.