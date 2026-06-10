# Grafana Dashboard — Known Issues / TODO

Tracking notes for the v3 mcs stack Grafana dashboards
(`_6G_integration_v3_docker_mcs/infrastructure/grafana/`).

---

## OPEN — "Falls per Patient" panel shows empty

**Dashboard:** Fall Events Timeline (`fall-events-timeline`)
**Panel:** "Falls per Patient (last 24h)" (panel id 7, barchart)
**Status:** open — noticed 2026-06-09, not yet fixed.

**Symptom:** the bar chart renders empty even though other panels (which read the
same `inference_log` table) show data.

**Current query:**
```sql
SELECT patient_id, COUNT(*) AS falls
FROM inference_log
WHERE fall_detected = true
  AND detection_time >= NOW() - INTERVAL '24 hours'
GROUP BY patient_id
ORDER BY falls DESC
```

**Likely causes to check when fixing:**
1. **Time window** — the only fall data so far is from 2026-06-08 (~74 min window).
   If "now" is more than 24h after that, `NOW() - INTERVAL '24 hours'` excludes it
   → empty. The stat panels using the same 24h filter would also be empty then;
   if they show data but this doesn't, it's not the time window.
2. **Barchart field mapping** — Grafana barchart needs a clear category (string)
   + value (number) field. May need `format: table` (already set) plus an explicit
   x-field / value-field in panel options. Verify the panel's `options` map
   `patient_id` → x axis and `falls` → value.
3. **`$patient_id` template filter** — this panel does NOT currently apply the
   `$patient_id` variable, so that's not the cause, but confirm if the panel is
   later edited to add it.

**Quick diagnosis when picking this up** (raw query works at the DB level — proven
2026-06-09 returning `patient_test_50`=6, `patient_test_500`=4 over a 720h window):
```powershell
docker exec mcs_fall_postgres psql -U fall_user -d fall_detection -c "SELECT patient_id, COUNT(*) AS falls FROM inference_log WHERE fall_detected = true AND detection_time >= NOW() - INTERVAL '24 hours' GROUP BY patient_id ORDER BY falls DESC"
```
If that returns rows but the panel is empty → it's a panel field-mapping issue, not
data. If it returns nothing → it's the 24h time window (data is older than 24h).

---

## RESOLVED — dashboard fully broken (fixed 2026-06-09)

Three stacked bugs, all fixed:
1. **Dropped tables** — panels queried `fall_history` / `participant_session`
   (removed in the architecture migration). Rewritten to use `inference_log`
   (which now carries `patient_confirmed` / `needs_help`). Session panel
   repurposed to "Patients Seen (24h)". Added two confidence-drift panels.
2. **Wrong datasource password** (the real cause of every query `400`) — the
   Postgres datasource hardcoded `password: fall_pass` but Postgres was
   initialised with `POSTGRES_PASSWORD=CHANGE_ME`. Changed datasource to read
   `${POSTGRES_PASSWORD}` and passed that env var into the Grafana container.
   Health now reports "Database Connection OK".
3. **Datasource type + template-variable format** — type was the legacy
   `postgres` instead of `grafana-postgresql-datasource` (Grafana 10.4); the
   `patient_id` variable needed the minimal string-query form (no `regex`/object
   `query`). After the fix the `re.replace is not a function` error required
   clearing stale browser state (incognito / fresh navigation).

Files: `infrastructure/grafana/dashboards/fall_events_timeline.json`,
`infrastructure/grafana/provisioning/datasources/datasources.yaml`,
`docker-compose.yml` (grafana service env).
