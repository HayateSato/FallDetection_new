# Handover Q&A

Running list of questions raised during handover discussions, with answers traced back to the code.
Add new entries at the top so the most recent question is easiest to find.

---

## Q: How is `latency_ms` in `inference_log` calculated? From what point until what point?

**Short answer:** It measures only the inference server's internal processing time — from "request parsed, about to compute" to "model returned a prediction". It does NOT include network transit, DB writes, FHIR push, or the mobile app's confirmation popup.

### Start point

[`inference_server/server.py:403`](../_6G_Integration_v2_mqtt/inference_server/server.py#L403):

```python
t_start = time.monotonic()
```

This runs **after**:
- TCP / HTTP / TLS handshake
- FastAPI receives the request body
- Pydantic validates the schema
- Model lookup / lazy load (cached after first call)
- `obs_id = uuid.uuid4()` is generated

### End point

[`inference_server/server.py:455`](../_6G_Integration_v2_mqtt/inference_server/server.py#L455):

```python
latency_ms = int((time.monotonic() - t_start) * 1000)
```

This runs **immediately after** `engine.predict()` returns and the result fields (`is_fall`, `confidence`) are extracted.

### What IS included in the measurement

1. NumPy array construction from the request payload
2. Hardware-rate to 50 Hz resampling — currently a no-op since `HARDWARE_ACC_SAMPLE_RATE=50` already equals `ACC_SAMPLE_RATE=50`
3. LSB to g unit conversion
4. DataFrame conversion
5. Detection-window slice (last 9 s = 450 samples)
6. **The actual XGBoost feature extraction + prediction**

### What is NOT included

- Network transit (mobile app -> ingress -> inference_server)
- HTTP / Pydantic request parsing
- Model lazy-load on first request (one-time cost, not in steady state)
- FHIR push to the FOCUS FHIR server
- Postgres write of the `inference_log` row
- Prometheus metric emit
- Response logging
- Network return path back to the client
- The mobile app's 10 s patient-confirmation popup
- MQTT publish + `fall_dashboard`'s `fall_history` write (separate flow over the broker)

### Clock used

`time.monotonic()`, not `time.time()` — immune to NTP / wall-clock jumps, but only valid as a delta within the same process.

### What this means in practice

A row showing `latency_ms = 5` means the model finished in 5 ms once we had the data in hand. The user-perceived round-trip (mobile-app POST -> mobile-app receives response) is this number plus network + DB write + FHIR push, which is **not** captured anywhere today.

### Full end-to-end latency is NOT implemented today

If we want a true end-to-end metric (e.g. for an SLA: "fall is detected and acknowledged within X ms of the sensor sample"), we'd need to add:

| Layer | What to measure | How |
|---|---|---|
| Mobile app -> server | Wall-clock at client send vs. client receive | Mobile app records its own round-trip and sends it as a header or a separate metric. Cannot be done server-side alone — server has no view of network transit. |
| Full server-side handler | First line of `/predict` to last line before `return` | Move `t_start` to the very top of the handler and capture a second timestamp after the response body is built. Add a new column `latency_full_ms` to `inference_log`. |
| FHIR push duration | Around the `httpx.post(fhir_url, ...)` call | Wrap the FHIR push in its own timer and store as `latency_fhir_ms`. Today it's fire-and-forget after `t_start` was captured, so it's invisible. |
| DB write duration | Around the `INSERT INTO inference_log` | Wrap `db_writer.write_inference_log` in its own timer. |
| MQTT round-trip (client confirmation) | Server publish -> client publish back on `fall/alert/<pid>` | Already partly observable: `fall_history.detection_time` vs. `fall_history.confirmation_time`. Could be exposed as a Prometheus histogram. |

**Recommended minimum if the question comes up:** add `latency_full_ms` as a sibling column. It's a 5-minute change in `server.py` + a one-line Alembic migration, and it answers "how long did the server take from receive to respond" without needing client-side cooperation.

The current `latency_ms` should then be renamed to `latency_inference_ms` so the two are unambiguous in the schema.
