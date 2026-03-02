# Data Processing Journey

Traces the full transformation of sensor data from InfluxDB query to model inference and CSV export.

---

## Overview

```
InfluxDB
  └─► List[FluxRecord]        (data_fetcher)
        └─► numpy (3, N)       (data_processor → convert_acc_from_flux_to_numpy_array)
              └─► numpy (3, N) (acc_resampler   → optional resample)
                    └─► numpy (3, N) (nonbosch_calibration → optional calibrate)
                          └─► pandas DataFrame (N, 4)  (data_exporter → convert_to_dataframe)
                                └─► pandas DataFrame (450, 4) (data_exporter → compose_detection_window)
                                      └─► dict[str, float]  (inference_engine → extract_features)
                                            └─► numpy (1, 16) → XGBoost → result dict
                                                                              │
                                                                   (flux_records, original)
                                                                              └─► CSV file
```

---

## Stage 1 — InfluxDB → `List[FluxRecord]`

**File:** `data_loader/data_fetcher.py`

A Flux query is built in `influx_data_fetcher.fetch_and_preprocess_sensor_data()` and executed
against InfluxDB. The response is a list of `FluxTable` objects, each containing a list of
`FluxRecord` objects. These are flattened into a single Python list.

```
flux_records: List[FluxRecord]
```

Each `FluxRecord` exposes:
| Method | Returns | Example |
|---|---|---|
| `get_field()` | field name string | `"bosch_acc_x"` |
| `get_value()` | raw sensor value | `3842` (ADC count) |
| `get_time()` | Python `datetime` | `2026-02-20 10:00:01.040+00:00` |

All axes, barometer, and annotation fields (`manual_truth_marker`, `user_feedback`) arrive
in the **same flat list** — one record per measurement per timestamp.

**This list is kept unmodified all the way to Stage 9 (CSV export).**

---

## Stage 2 — `List[FluxRecord]` → numpy arrays

**File:** `data_processor.py` — `convert_acc_from_flux_to_numpy_array()`, `convert_baro_from_flux_to_numpy_array()`

Records are separated by field name and converted to numpy arrays.

**Accelerometer:**
```
acc_data  : numpy (3, N)  — raw hardware ADC counts (LSB integers)
acc_time  : numpy (N,)    — timestamps in milliseconds (float64)
                            converted via: record.get_time().timestamp() * 1000
```
Shape `(3, N)` means row 0 = X axis, row 1 = Y axis, row 2 = Z axis.
X-axis timestamps are used as the reference time array (Y and Z share the same cadence).

**Barometer (only when model uses it):**
```
pressure      : numpy (M,)  — pressure values in Pa
pressure_time : numpy (M,)  — timestamps in milliseconds
```

Units at this stage: **raw ADC integers** for ACC (e.g. `4096 ≈ 1g` when sensitivity = 4096 LSB/g), **Pa** for barometer.

---

## Stage 3 — Resampling *(conditional)*

**File:** `accelerometer_processor/acc_resampler.py` — `AccelerometerResampler`

Only executed when `HARDWARE_ACC_SAMPLE_RATE != MODEL_ACC_SAMPLE_RATE` (i.e. `RESAMPLING_ENABLED=True`).

| Hardware rate | Model rate | Direction | Default method |
|---|---|---|---|
| 25 Hz | 50 Hz | Upsample | `linear` interpolation |
| 100 Hz | 50 Hz | Downsample | `decimate` or `average` |
| 50 Hz | 50 Hz | No-op | — |

```
acc_data  (3, N_hw)    →  acc_data  (3, N_model)
acc_time  (N_hw,)      →  acc_time  (N_model,)
```

Units unchanged (still raw ADC counts, now possibly float from interpolation).

---

## Stage 4 — Calibration *(conditional)*

**File:** `accelerometer_processor/nonbosch_calibration.py` — `transform_acc_array()`

Only executed when `ACC_SENSOR_TYPE=non_bosch` (`SENSOR_CALIBRATION_ENABLED=True`).
Applies a pre-computed affine matrix to transform non-Bosch axis orientation and scale to
Bosch-equivalent values, so a single model can serve both hardware types.

```
acc_data  (3, N)  →  acc_data  (3, N)    (shape unchanged, values adjusted)
```

---

## Stage 5 — numpy → pandas DataFrame

**File:** `../data_output/data_exporter.py` — `convert_to_dataframe(acc_data, acc_time, acc_scale_factor)`

The 2-D numpy array is pivoted into a row-per-sample DataFrame and the scale factor is applied.

```
DataFrame columns: Device_Timestamp_[ms] | Acc_X[g] | Acc_Y[g] | Acc_Z[g]
DataFrame shape:   (N, 4)
```

**Scale factor is model-dependent** (determined by `PipelineSelector.get_acc_scale_factor()`):

| Model | `acc_in_lsb` | Scale factor | Values after conversion |
|---|---|---|---|
| `v0_lsb_int` | `True` | `1.0` | Raw ADC integers, e.g. `3842` |
| All others | `False` | `1 / ACC_SENSOR_SENSITIVITY` | g units, e.g. `0.937` |

> Note: the column is named `Acc_X[g]` in both cases for compatibility; for `v0_lsb_int`
> the unit is actually LSB, not g.

---

## Stage 6 — Full DataFrame → windowed DataFrame

**File:** `../data_output/data_exporter.py` — `compose_detection_window(df, required_samples)`

Takes the **last 450 rows** (`WINDOW_SIZE_SECONDS × MODEL_ACC_SAMPLE_RATE = 9 s × 50 Hz`)
and slices the barometer arrays to the matching timestamp range.

```
window_df            : DataFrame (450, 4)   — 9-second detection window
window_pressure      : numpy (K,)           — barometer samples inside the window
window_pressure_time : numpy (K,)           — timestamps for those samples
```

---

## Stage 7 — DataFrame → feature dict

**File:** `../core/inference_engine.py` — `PipelineSelector.extract_features()`

Column arrays are extracted from the DataFrame and statistical features are computed.

```python
acc_x = window_df['Acc_X[g]'].values   # numpy (450,)
acc_y = window_df['Acc_Y[g]'].values
acc_z = window_df['Acc_Z[g]'].values
acc_mag = sqrt(acc_x² + acc_y² + acc_z²)
```

For `v1_features` models (v0, v0_lsb_int, v1, v3): min, max, mean, var for each of
X, Y, Z and magnitude → **16 ACC features**.

Output:
```python
features: dict[str, float]   # 16 entries, e.g.:
# { 'acc_x_min': ..., 'acc_x_max': ..., 'acc_x_mean': ..., 'acc_x_var': ...,
#   'acc_y_...', 'acc_z_...', 'acc_mag_...' }
```

---

## Stage 8 — feature dict → prediction

**File:** `../core/inference_engine.py` — `PipelineSelector._predict_generic()`

Features are ordered to match the training column order and packed into a matrix:

```python
X = np.array([[features_dict[name] for name in feature_names]])  # shape (1, 16)
```

XGBoost inference returns a probability scalar which is thresholded:

```python
result = {
    'is_fall':    bool,   # True if confidence > threshold (default 0.5)
    'confidence': float,  # raw XGBoost probability 0–1
    'features':   dict,   # same features dict from Stage 7
}
```

---

## Stage 9 — CSV export

**File:** `../data_output/data_exporter.py` — `save_detection_window_to_csv(flux_records, ...)`

The **original unmodified `flux_records` list from Stage 1** is used here.
Each FluxRecord becomes one row in a long-format sensor data block:

```
columns: time | field | value | measurement | prediction
```

A metadata header block is prepended containing:
- Detection timestamp (UTC + local)
- Model version, confidence, fall/no-fall
- Participant name, gender
- `manual_truth_marker` (set at recording time)
- `user_feedback` (initially `-1`, updated later via `/fall_feedback` endpoint)

The CSV therefore contains the **full raw lookback window** (e.g. 30 s), not just the
9-second detection window — preserving all context for offline analysis.

---

## Key design notes

- **flux_records passes through untouched** — the raw InfluxDB objects are never modified;
  they are only read at the very end for CSV export.
- **LSB vs g branching happens only at Stage 5** — all stages before and after are
  unaffected; the model registry flag `acc_in_lsb` controls the scale factor.
- **Barometer data stays as numpy throughout** — it is never put into the pandas DataFrame;
  it is passed separately into feature extraction.
- **Two entry points, one pipeline** — both `continuous_monitoring.py` (background thread)
  and `detection.py` (`/trigger` route) call the same `fetch_and_preprocess_sensor_data()`
  function from `influx_data_fetcher.py`.
