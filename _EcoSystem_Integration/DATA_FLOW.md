# Data Processing Flow
## Fall Detection — InfluxDB → Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SmarKo Wearable  ──BT──►  SmarKo App  ──Wi-Fi──►  InfluxDB (cloud)    │
│                                                                         │
│  Fields written per measurement:                                        │
│    bosch_acc_x / y / z   (LSB integers, 25 Hz)    ← if bosch sensor    │
│    acc_x / y / z         (LSB integers, 100 Hz)   ← if non-bosch       │
│    bmp_pressure          (Pa, 25 Hz)               ← barometer         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                │  influxdb-client query
                                │  lookback window: last ~15 s
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  trigger_client/client.py  (runs wherever InfluxDB is reachable)        │
│                                                                         │
│  fetch_and_preprocess_sensor_data()                                     │
│    • Flux query: _measurement filters for configured field names        │
│    • Returns acc_data[3×N] (raw LSB integers), acc_time[N] (ms),       │
│      pressure[M] (Pa), pressure_time[M] (ms)                           │
│                                                                         │
│  Minimum data check: need ≥ window_seconds × hardware_rate samples     │
│  (e.g. 9s × 25Hz = 225 raw samples before resampling)                  │
│                                                                         │
│  Builds JSON payload:                                                   │
│    { acc_x[], acc_y[], acc_z[], timestamps_ms[],                        │
│      pressure[], pressure_timestamps_ms[],                              │
│      acc_unit: "lsb",  sample_rate: 25 }                               │
│                                                                         │
│  POST /predict + X-API-Key  ─────────────────────────────────────────► │
└─────────────────────────────────────────────────────────────────────────┘
                                │
          (over network / localhost depending on deployment)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  inference_server/server.py  (FastAPI :8001)                            │
│                                                                         │
│  Step 1 — Input validation                                              │
│    All four arrays (acc_x/y/z, timestamps_ms) must have equal length.  │
│                                                                         │
│  Step 2 — Resampling  (if hardware_rate ≠ 50 Hz)                       │
│    AccelerometerResampler                                               │
│      25 Hz  → linear interpolation → 50 Hz  (×2 upsample)             │
│      100 Hz → decimate or average   → 50 Hz  (×2 downsample)          │
│    Result: acc_data[3 × N′] at 50 Hz, new timestamps                   │
│                                                                         │
│  Step 3 — Unit conversion                                               │
│    convert_lsb_to_g():  raw_LSB ÷ 16384 → acceleration in g            │
│    (skipped for v0_lsb_int and v5_lsb models which expect raw LSB)     │
│                                                                         │
│  Step 4 — DataFrame                                                     │
│    convert_acc_nparray_to_df()                                          │
│    Columns:  Device_Timestamp_[ms] | Acc_X[g] | Acc_Y[g] | Acc_Z[g]   │
│                                                                         │
│  Step 5 — Detection window                                              │
│    compose_detection_window()                                           │
│    Extracts the LAST  window_seconds × 50 = 450 rows (default 9 s)     │
│    from the DataFrame (earlier rows are discarded)                      │
│    Barometer is aligned to the same time span.                          │
│                                                                         │
│  Step 6 — Feature extraction   PipelineSelector.extract_features()     │
│                                                                         │
│    ACC features (v1_features — all current models except v2_paper):    │
│      per-axis statistical features over the 450-sample window:         │
│        acc_x_min, acc_x_max, acc_x_mean, acc_x_var                     │
│        acc_y_min, acc_y_max, acc_y_mean, acc_y_var                     │
│        acc_z_min, acc_z_max, acc_z_mean, acc_z_var                     │
│        acc_mag_min, acc_mag_max, acc_mag_mean, acc_mag_var             │
│      magnitude = sqrt(x²+y²+z²)                          → 16 features │
│                                                                         │
│    Barometer features (v2_paper — used by v3):                         │
│      slope-limit filter → moving average → feature extraction:         │
│        pressure_shift     (Pa delta over window)                        │
│        middle_slope       (slope at midpoint)                           │
│        post_fall_slope    (slope in second half)                        │
│        filtered_pressure_var                              → 4 features  │
│                                                                         │
│    Total features by model version:                                     │
│      v0           16 (ACC only)                                         │
│      v0_lsb_int   16 (ACC only, raw LSB input)                         │
│      v3           20 (ACC + BARO)   ← recommended                      │
│      v5_lsb       22 (raw features, LSB input)                         │
│                                                                         │
│  Step 7 — XGBoost inference                                             │
│    model.predict_proba(X)[1]  → fall_probability  ∈ [0.0, 1.0]        │
│    threshold (default 0.5)    → is_fall: bool                           │
│                                                                         │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
                             JSON response (synchronous):
                             {
                               "fall_detected":  true | false,
                               "confidence":     0.0 – 1.0,
                               "threshold":      0.5,
                               "result":         "High confidence fall detection",
                               "model_version":  "v3",
                               "model_name":     "...",
                               "num_features":   20,
                               "window_size":    450,
                               "features":       { "acc_x_min": ..., ... }
                             }
                                         │
                                         ▼
                          trigger_client on_result() callback
                          (integrate here: push to your system,
                           write marker to InfluxDB, trigger alert, etc.)
```

---

## Configurable Parameters

All parameters can be set via `.env` (loaded at startup) **or** updated at runtime via `POST /config`:

| Parameter | `.env` key | Default | Effect |
|-----------|-----------|---------|--------|
| Model version | `MODEL_VERSION` | `v0` | Which XGBoost model to use (`v0`, `v3`, `v5_lsb`, …) |
| Hardware ACC rate | `HARDWARE_ACC_SAMPLE_RATE` | `25` | Hz of sensor; triggers resampling to 50 Hz |
| ACC sensor type | `ACC_SENSOR_TYPE` | `bosch` | `bosch` → `bosch_acc_x/y/z` fields; `non_bosch` → `acc_x/y/z` |
| Resampling method | `RESAMPLING_METHOD` | `linear` | `linear` (upsample), `decimate` or `average` (downsample) |
| Window size | `WINDOW_SIZE_SECONDS` | `9` | Detection window in seconds (9 s × 50 Hz = 450 samples) |
| InfluxDB bucket | `INFLUXDB_BUCKET` | — | Which bucket to query |
| Monitoring interval | `MONITORING_INTERVAL_SECONDS` | `5` | How often the client polls InfluxDB |
| Lookback | `MONITORING_LOOKBACK_SECONDS` | `15` | How far back the InfluxDB query fetches |
| API key | `API_KEYS` (server) / `REMOTE_API_KEY` (client) | — | Shared secret for `X-API-Key` header |

### Runtime config change (no restart needed)

```bash
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"hardware_sample_rate": 25, "window_seconds": 9, "acc_sensor_type": "bosch"}'
```

### Switch model at runtime

```bash
curl -X POST http://localhost:8001/model/switch \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"version": "v3"}'
```

---

## Data Volume Estimates

| Parameter | Value |
|-----------|-------|
| InfluxDB query cadence | every 5 s (default) |
| Lookback window fetched | 15 s |
| Raw ACC samples fetched | 15 s × 25 Hz = 375 samples per axis |
| After resampling (50 Hz) | 750 samples per axis |
| Detection window used | last 450 samples (9 s × 50 Hz) |
| Payload size (JSON) | ~50 KB per request |
| Inference latency | typically 5–30 ms (XGBoost on CPU) |
