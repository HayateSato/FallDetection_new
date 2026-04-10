# Fall Detection — 6G / Charite Integration

Minimal inference server that returns fall detection results in **FHIR R4 Observation** format,
designed for integration into the FOCUS/Charite monitoring stack.

```
_6G_Integration/
├── .env                        ← all configuration (annotated with owner)
├── README.md                   ← you are here
├── fhir_converter.py           ← builds FHIR R4 Observation from inference result
├── app/                        ← shared ML pipeline (do not edit)
├── config/                     ← settings.py reads .env
├── model/                      ← XGBoost .pkl files (v0, v3, v0_lsb_int, v5_lsb)
└── inference_server/
    ├── server.py               ← FastAPI server
    └── requirements.txt
```

---

## What is fixed vs. configurable

| Parameter | Where set | Can change at runtime? |
|---|---|---|
| Model version | `.env` → `MODEL_VERSION` | No — restart required |
| Sensor type (bosch / non_bosch) | `.env` → `ACC_SENSOR_TYPE` | No |
| Hardware sample rate | `.env` → `HARDWARE_ACC_SAMPLE_RATE` | No |
| **ACC input unit** | **Always LSB** (server converts internally) | No |
| **Target sample rate** | **Always 50 Hz** (fixed — what XGBoost was trained on) | No |
| **Detection window** | **Always 9 s × 50 Hz = 450 samples** | No |
| FHIR server URL | `.env` → `FHIR_SERVER_URL` | No |
| API key | `.env` → `API_KEYS` | No |

---

## Quick start

```bash
cd _6G_Integration

# Install dependencies
pip install -r inference_server/requirements.txt

# Edit .env — set MODEL_VERSION, API_KEYS, and optionally FHIR_SERVER_URL
# Then start:
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001
```

Verify:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/model/info    # check uses_barometer field
curl http://localhost:8001/docs         # Swagger UI
```

---

## API — `POST /predict`

**Header:** `X-API-Key: <your key>`

### Request body

```json
{
  "patient_id":              "charite-patient-007",
  "device_id":               "smarko-wearable-42",
  "acc_x":                   [-512, -498, ...],
  "acc_y":                   [128, 134, ...],
  "acc_z":                   [16300, 16280, ...],
  "timestamps_ms":           [1712345678000, 1712345678040, ...],
  "pressure":                [101325.0, 101322.5, ...],
  "pressure_timestamps_ms":  [1712345678000, 1712345678040, ...]
}
```

**Notes on input:**
- `acc_x/y/z` — raw **LSB integers** directly from InfluxDB (`bosch_acc_x/y/z` for Bosch sensor). The server converts to g and resamples to 50 Hz internally. Do **not** pre-convert.
- `timestamps_ms` — Unix epoch in milliseconds (same as InfluxDB timestamps).
- `pressure` — required when model is `v3` (uses barometer). Not required for `v0`.
- Minimum samples: `9s × 25Hz = 225` per axis (server will return 422 if fewer).

### Response body

```json
{
  "patient_id":  "charite-patient-007",
  "device_id":   "smarko-wearable-42",
  "timestamp":   "2026-04-07T10:23:45.123456+00:00",

  "inference": {
    "fall_detected": true,
    "confidence":    0.8721,
    "threshold":     0.5,
    "result":        "High confidence fall",
    "model_version": "v3",
    "window_size":   450
  },

  "fhir_observation": {
    "resourceType": "Observation",
    "id":           "b3f7c2a1-...",
    "status":       "final",
    "category": [{ "coding": [{ "system": "...observation-category", "code": "activity" }] }],
    "code": {
      "coding": [{ "system": "http://snomed.info/sct", "code": "217082002", "display": "Fall (event)" }]
    },
    "subject":          { "reference": "Patient/charite-patient-007" },
    "device":           { "reference": "Device/smarko-wearable-42" },
    "effectiveDateTime": "2026-04-07T10:23:45.123456+00:00",
    "valueBoolean":     true,
    "component": [
      {
        "code": { "coding": [{ "system": "http://loinc.org", "code": "72514-3", "display": "Fall risk assessment score" }] },
        "valueQuantity": { "value": 0.8721, "unit": "probability", "system": "http://unitsofmeasure.org", "code": "1" }
      },
      {
        "code": { "coding": [{ "system": "http://snomed.info/sct", "code": "246075003", "display": "Causative agent" }] },
        "valueString": "SmarKo-FallDetection-v3"
      }
    ]
  },

  "fhir_pushed": true
}
```

**`fhir_pushed`** — `true` if the server successfully POSTed the Observation to `FHIR_SERVER_URL`.
If `FHIR_SERVER_URL` is not configured, this is always `false` — but `fhir_observation` is always present in the response for the caller to forward manually.

---

## FHIR server integration

### Option A — server pushes automatically

Set `FHIR_SERVER_URL` in `.env`:
```env
FHIR_SERVER_URL=https://fhir.focus-partner.de/fhir
FHIR_AUTH_TOKEN=Bearer eyJhbGci...    # if the FHIR server requires auth
FHIR_PUSH_ON_FALL_ONLY=true           # only push when fall_detected=true
```

The server will `POST /Observation` to the FHIR server after every inference where `fall_detected=true`.
The main `/predict` response is always returned to the caller even if the FHIR push fails.

### Option B — caller forwards manually

Leave `FHIR_SERVER_URL` empty. Extract `fhir_observation` from the response and POST it yourself:

```python
result = requests.post("http://localhost:8001/predict", json=payload, headers=headers).json()

if result["inference"]["fall_detected"]:
    fhir_resp = requests.post(
        "https://fhir.focus-partner.de/fhir/Observation",
        json=result["fhir_observation"],
        headers={"Content-Type": "application/fhir+json", "Authorization": "Bearer ..."},
    )
```

---

## FHIR Observation — codes used

| Field | System | Code | Display |
|---|---|---|---|
| Category | `observation-category` | `activity` | Activity |
| Observation type | SNOMED CT | `217082002` | Fall (event) |
| Confidence score | LOINC | `72514-3` | Fall risk assessment score |
| Algorithm version | SNOMED CT | `246075003` | Causative agent |

---

## Minimum data per request

The server uses the **last 9 seconds** of data from whatever you provide.
You must provide at least 9 seconds worth:

| Sensor config | Min samples per axis |
|---|---|
| Bosch 25 Hz → after resampling to 50 Hz | 225 raw samples (9s × 25Hz) |
| Non-Bosch 100 Hz → after downsampling to 50 Hz | 900 raw samples (9s × 100Hz) |

Typical InfluxDB lookback: **15 seconds** → 375 raw Bosch samples → sufficient.

---

## What is NOT included

Compared to the full system and `_EcoSystem_Integration`:

- No model switching at runtime (restart required)
- No InfluxDB trigger client (caller is responsible for fetching sensor data and calling `/predict`)
- No Redis / patient feedback / emergency alert service
- No PostgreSQL inference history (first step — add later)
- No Prometheus / Grafana
- No patient dashboard / caregiver dashboard
