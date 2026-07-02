---
name: Patient feedback loop implementation
description: Architecture and gotchas for the patient popup, two-channel Redis, asyncio timer, and caregiver dashboard rewrite
type: project
---

Implemented 2026-03-26. Covers ml_server changes, patient dashboard, and caregiver dashboard rewrite.

**Two Redis channels:**
- `patient_alerts` — published on every real-time fall immediately after detection
- `fall_events` — published only when an alert must reach emergency contact (conditional)

**ml_server asyncio timer pattern:**
- `_pending_emergency_tasks: Dict[int, asyncio.Task]` — module-level dict, keyed by inference_id
- On fall detection: start `asyncio.create_task(_delayed_emergency_alert(..., delay_seconds=12))`
- `POST /patient/feedback/{id}` cancels the task via `task.cancel()` before updating DB
- **Gotcha:** asyncio tasks are per-process. With `--workers > 1`, the feedback POST may hit a different worker than the one holding the timer. Use `--workers 1` for ml_server until timer state is moved to Redis.

**Feedback values stored in inference_log:**
- `user_fall`: 0=pending, 1=yes (fell), 2=no (didn't fall), 3=no_answer (12s timeout)
- `need_help`: 0=pending, 1=yes, 2=no, 3=no_answer (10s step-2 client timeout)

**Patient dashboard SSE:**
- Connects to `GET /api/ml/patient/stream?participant=<name>` (optional name filter)
- nginx has a dedicated SSE location block before the generic `/api/ml/` block (proxy_buffering off, 1h timeout)
- Patient name stored in `localStorage` — persists across page refreshes

**Caregiver dashboard routing fix:**
- `/patients/stream` must be declared BEFORE `/patients/{patient_name}/...` in FastAPI
- Without this, "stream" is matched as a patient_name path parameter

**CSS hidden bug fixed:**
- Old code used HTML `hidden` attribute on divs that also had `display: flex` in CSS — CSS wins
- Fixed by using `.hidden { display: none !important; }` class throughout

**Why:** Closes the false-positive loop. Emergency contact is NOT alerted when the patient is fine. Ground truth labels stored for future retraining use.
