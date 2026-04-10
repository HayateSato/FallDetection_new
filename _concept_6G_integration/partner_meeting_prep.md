# Partner Integration Meeting — Preparation Document
**Audience:** Software developers / DevOps at FOCUS  
**Your role:** Data scientist presenting the fall detection system  
**Goal:** Align on what you deliver, what they need to provide, and how the two sides connect

---

## 1. What You Are Providing (30-second pitch)

> "We have a trained machine learning model that detects falls in real time from wrist-worn accelerometer data. We wrap it in a REST API that your system can call. When a fall is detected, the API returns the result in FHIR R4 format — the standard your FHIR server likely already speaks. We also have a reference client implementation that polls InfluxDB, calls the API, and stores the fall history."

The deliverable is **two components**:

| Component | What it does | Port |
|-----------|-------------|------|
| **Inference Server** | Receives sensor data → runs ML model → returns FHIR Observation | 8001 |
| **Caregiver Client** | Polls InfluxDB → calls Inference Server → stores history → serves dashboard + SSE alerts | 8002 |

They can integrate at whichever boundary fits their stack. If they already have an InfluxDB poller and dashboard, they may only need the Inference Server and call it directly.

---

## 2. How the Model Was Trained

### What data was used?

- **918 labeled windows** collected from wrist-worn SmarKo IMU sensors
  - 420 fall windows
  - 498 ADL (Activities of Daily Living) windows — walking, sitting, gestures, etc.
- Train/test split: **734 training / 184 test** (80/20), stratified by class
- 5-fold cross-validation used for model selection

> **If asked about data source:** Training data was collected using the same SmarKo wearable hardware that Charite participants will use. Falls were simulated/labeled in a controlled study. ADL data represents everyday movement.

### What algorithm?

- **XGBoost** binary classifier (gradient boosted decision trees)
- Input: **16 statistical features** extracted from a 9-second accelerometer window
- Output: probability 0–1; threshold 0.5 → fall / no fall

### What features does the model use?

Each prediction takes a **9-second window** of accelerometer data (3 axes: X, Y, Z) and computes:

| Feature group | Features |
|--------------|---------|
| Per-axis (X, Y, Z) | min, max, mean, variance — 12 features |
| Magnitude (√X²+Y²+Z²) | min, max, mean, variance — 4 features |
| **Total** | **16 features** |

The top 3 most important features are: `acc_mag_max`, `acc_x_max`, `acc_y_min` — essentially, the peak magnitude of the impact is the strongest signal.

### Why XGBoost and not a neural network?

- Interpretable: you can inspect feature importance
- Fast inference: ~1–5ms per window, no GPU needed
- Small model file: ~100KB, runs on a laptop or small VM
- 918 samples is too small for deep learning to generalize reliably

### Performance numbers (test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| v0 (ACC only) | 99.5% | 100% | 98.8% | 99.4% | 0.994 |
| v3 (ACC + barometer) | 99.5% | 100% | 98.8% | 99.4% | 0.993 |

Confusion matrix on 184 test samples:
- True negatives (correctly no-fall): 100
- True positives (correctly fall): 83
- False positives: 0
- False negatives: 1

> **Honest caveat to mention if asked:** These numbers are from a controlled lab setting. Real-world performance may differ — elderly participants move differently, sensors may have different noise profiles, and edge cases (near-falls, sitting down quickly) are harder. We recommend a calibration period at the start of the trial.

---

## 3. Data Pipeline — What Happens on Every Prediction

```
Wearable sensor (SmarKo watch, 25Hz)
    │  Bluetooth
    ▼
SmarKo mobile app
    │  Wi-Fi / HTTPS
    ▼
InfluxDB  [measurement: SMART_DATA, tag: macAddress]
    │  Query: last 15–30 seconds, filtered by macAddress
    ▼
Caregiver Client (influx_poller.py)
    │  Build JSON payload with raw ACC arrays + timestamps
    ▼
POST /predict  (Inference Server, port 8001)
    │
    ├── Resample: 25Hz → 50Hz (linear interpolation)
    ├── Convert: LSB integers → g units  (÷ 16384)
    ├── Window: take last 9 seconds = 450 samples
    ├── Extract: 16 statistical features
    └── XGBoost: predict probability → threshold 0.5
    │
    ▼
Response (FHIR R4 Observation JSON)
    │
    ├── If fall: publish to Redis → alert dashboard in real time
    └── Always: store in local DB (SQLite or Postgres)
```

### Key numbers to remember:
- Sensor rate: 25 Hz (Bosch accelerometer in SmarKo)
- Model input rate: 50 Hz (we upsample)
- Detection window: **9 seconds** (one prediction per polling cycle)
- Polling cycle: every 10–15 seconds (configurable)
- Latency from fall to alert: **~10–15 seconds** (dominated by polling interval, not inference)

---

## 4. The API Interface

### POST /predict

**Request body:**
```json
{
  "acc_x": [1024, 1028, ...],      // raw ADC integers, 25Hz, ~375 values (15s)
  "acc_y": [512, 510, ...],
  "acc_z": [4096, 4100, ...],
  "timestamps_ms": [1700000000000, 1700000040, ...],
  "patient_id": "Patient/patient-001",
  "model_version": "v0"            // optional, defaults to server .env setting
}
```

**Response (FHIR R4 Observation):**
```json
{
  "resourceType": "Observation",
  "id": "fall-detection-...",
  "status": "final",
  "code": { "coding": [{ "system": "http://snomed.info/sct", "code": "217082002", "display": "Fall" }] },
  "subject": { "reference": "Patient/patient-001" },
  "valueBoolean": true,
  "component": [
    { "code": { "coding": [{ "code": "72514-3" }] }, "valueDecimal": 0.994 }
  ]
}
```

The `is_fall` and `confidence` are also available as plain fields in a wrapper around the FHIR object if needed.

### Other endpoints
- `GET /health` — is server alive, which model is loaded
- `GET /model/info?version=v0` — feature count, uses barometer or not
- `GET /docs` — interactive Swagger UI (auto-generated)

---

## 5. What We Need From Them — Questions to Ask

Prepare to ask these in the meeting. Frame them as "we need these to configure the integration correctly."

### About their InfluxDB
- [ ] What is the InfluxDB URL and bucket name?
- [ ] What measurement name stores the accelerometer data? (We assume `SMART_DATA`)
- [ ] What tag identifies each patient/device? (We use `macAddress`)
- [ ] What field names are used for ACC axes? (We use `bosch_acc_x/y/z`)
- [ ] Is barometer data also stored? If yes, what field name? (We use `bmp_pressure`)
- [ ] What is the sample rate of the accelerometer sensor they use?

### About their FHIR server
- [ ] Which FHIR server are they running? (HAPI FHIR? Azure FHIR? Firely?)
- [ ] Do they want us to **push** the Observation to their FHIR server automatically, or just **return** it in the API response?
- [ ] What is the base URL if push is needed? (e.g. `https://fhir.focus-partner.de/fhir`)
- [ ] Is authentication required? (Bearer token, client cert?)

### About their dashboard / monitoring system
- [ ] How is it built? (React? Vue? Angular? Plain JS?)
- [ ] How does it currently receive real-time updates? (WebSocket? Polling? SSE?)
- [ ] Do they want to consume our Redis events directly, or should we expose an SSE endpoint they can subscribe to?
- [ ] Do they want the fall history API, or will they read from our Postgres/their own DB?

### About the deployment environment
- [ ] Where will this run? (Their server? Cloud VM? On-premise?)
- [ ] Docker is fine — do they have a specific Docker registry or do we just provide a `docker-compose.yml`?
- [ ] What OS / architecture? (Linux x86? ARM?)
- [ ] Network: can our container reach their InfluxDB? Is there a firewall/VPN?
- [ ] Do they need the inference server to be publicly reachable (via ngrok/Cloudflare), or is everything on the same internal network?

### About the trial
- [ ] How many patients will be monitored simultaneously?
- [ ] Is monitoring 24/7 or only during specific hours?
- [ ] Do they want a feedback mechanism (patient confirms fall yes/no)?

---

## 6. Integration Options — Three Levels

Depending on what they already have, offer the right integration level:

### Option A — Drop-in Inference Server only (minimal footprint)
They keep their existing InfluxDB poller and dashboard. We just give them the inference server.
- They `POST /predict` from their existing code
- We return FHIR — they push it to their FHIR server themselves
- No Redis, no caregiver client needed

**Good if:** They already have an integration layer and just need the ML endpoint.

### Option B — Full client + server (reference implementation)
We run both components. They point our client at their InfluxDB. We handle polling, inference, FHIR push, fall history DB, and real-time SSE alerts.
- Their dashboard subscribes to our SSE stream for live alerts
- Their dashboard can query our `/api/falls` for history

**Good if:** They want a working end-to-end system with minimal effort.

### Option C — Dockerized microservice (production)
We provide a `docker-compose.yml` that adds our services alongside their existing stack.
- Uses their existing Redis/Postgres if they have one
- Communicates via standard REST + SSE
- Full environment variable configuration

**Good if:** They want production-grade deployment integrated into their infrastructure.

---

## 7. Configuration — What They Need to Set

If they run our system, these are the only things that need to be configured in `.env`:

```env
# Which sensor hardware (bosch = standard SmarKo)
ACC_SENSOR_TYPE=bosch
HARDWARE_ACC_SAMPLE_RATE=25

# Their InfluxDB
INFLUXDB_URL=https://...
INFLUXDB_TOKEN=...
INFLUXDB_ORG=...
INFLUXDB_BUCKET=...

# Patient mapping (comma-separated, 1:1 positional)
PATIENT_IDS=Patient/001,Patient/002
MAC_IDS=aa:bb:cc:dd:ee:ff,11:22:33:44:55:66

# Their FHIR server (optional — if empty, FHIR is returned in API response only)
FHIR_SERVER_URL=https://fhir.focus-partner.de/fhir
FHIR_AUTH_TOKEN=...

# Redis (optional — for real-time dashboard alerts)
REDIS_URL=redis://localhost:6379/0
```

---

## 8. Expected Questions and How to Answer

### "What is your model's false positive rate in real-world conditions?"

> "In our test set, we had 0 false positives out of 100 non-fall windows — but that was a controlled dataset. In practice, aggressive movements like sports or quick sitting down could trigger false positives. We don't have real-world data yet from this specific trial population (elderly). The feedback mechanism (patient confirms yes/no) is designed to catch this — we can use that data to retrain and improve the model over time."

### "Can the model run on-premise without internet?"

> "Yes, completely. The inference server is a single Docker container with no external dependencies. All ML happens locally. The only external calls are to your InfluxDB (to fetch sensor data) and optionally to your FHIR server (to push results) — both of which are on your network anyway."

### "What happens if InfluxDB is unavailable?"

> "The polling cycle fails silently — we log the error and retry on the next cycle. No fall events are generated during the outage. No data is lost from InfluxDB's side since it received and stored the sensor data from the wearable independently. We just can't run inference until the connection is restored."

### "Can we use our own Postgres instead of your SQLite?"

> "Yes, just set `DATABASE_URL=postgresql://user:pass@host:5432/dbname` in the `.env` file. SQLite is the default for zero-setup development, but we support any SQLAlchemy-compatible database."

### "How do we add new patients?"

> "Add their patient ID and MAC address to the `PATIENT_IDS` and `MAC_IDS` environment variables (comma-separated, same position), then restart the caregiver client. That's it — new polling loops start automatically."

### "Why are you polling every 10 seconds? Can we get real-time alerts?"

> "The 10-second polling interval is the main latency driver. The actual ML inference takes under 5ms. Polling interval is configurable — lower it to 5 seconds for faster alerts. True real-time would require the SmarKo app to push directly to us on each sample, which is a change on the hardware/app side outside our scope. Once a fall IS detected, the Redis alert reaches the dashboard in under 1 second."

### "Is the model specific to SmarKo hardware?"

> "The model was trained on SmarKo Bosch ACC data. We also support non-Bosch sensors via a calibration matrix, but it needs to be validated for each new hardware type. If FOCUS is using a different wearable, we would need sample data to calibrate and potentially retrain."

### "What FHIR version and which resources do you support?"

> "We output FHIR R4 Observation resources. The coding uses SNOMED CT (217082002 = Fall event), LOINC (72514-3 = severity), and SNOMED CT (246075003 = confidence). If they need a different coding system or a different resource type (e.g. QuestionnaireResponse for patient feedback), we can adjust that — it's just a configuration in our converter."

### "How do we update the ML model if we retrain it?"

> "Drop a new `.pkl` model file into the `model/model_vX/` directory, add a `config.json` describing its features, then either restart the server or call `POST /model/switch` with the new version name — no redeployment needed."

### "What is your SLA / uptime guarantee?"

> "This is a research prototype, not a production service. We don't have SLA commitments. If you need production SLAs, the inference server should be run with a process manager (systemd, Kubernetes) and health-check monitoring — the `/health` endpoint is there for that. We can discuss what monitoring setup makes sense for your infrastructure."

### "Do we need to store the raw sensor data on your side?"

> "No. We only need a short lookback window (15 seconds) to run each prediction — we never persist raw sensor data. Your InfluxDB remains the single source of truth for all raw sensor data."

### "Why Redis? Can we use something else?"

> "Redis is used for real-time pub/sub — when a fall is detected, we publish a message that dashboard clients subscribe to for instant alerts. If you already have RabbitMQ, Kafka, or MQTT in your stack, we could adapt the publisher. Or if your dashboard already has a mechanism for real-time updates, we can skip Redis entirely — the fall history DB is always there as a fallback polling source."

---

## 9. Things You Don't Know — Honest Answers

It is fine to say these things directly:

- **"I don't know the specifics of your FHIR server setup."** → "Can you share documentation or a test endpoint so I can verify compatibility?"

- **"I haven't tested with more than 1–2 simultaneous patients."** → "We'll need to test concurrent polling under your expected patient count. The design should scale but we haven't benchmarked it."

- **"I haven't built the Docker Compose for your full stack yet."** → "I can have a `docker-compose.yml` ready once I know what other services you're running so I can wire the network correctly."

- **"I'm not sure which FHIR resource type is most appropriate for your workflow."** → "Can you tell me how you currently use FHIR resources in your system? We can match that."

---

## 10. What You Want Out of This Meeting

By the end, you need:

1. Their InfluxDB connection details (or a test environment)
2. Confirmation: do they want inference-server-only, or full client+server?
3. Their FHIR server URL and auth method (if push is needed)
4. Their dashboard technology (so we know how to connect the alert stream)
5. Deployment environment (OS, Docker version, network layout)
6. A follow-up contact for technical integration questions
