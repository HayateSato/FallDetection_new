# App File Structure - Layered Architecture

## Key Decisions

| File | Previous Location | New Location | Rationale |
|------|-------------------|--------------|-----------|
| `data_processing_registry.py` | `app/` | `app/core/inference_engine.py` | It's the inference engine, not a "registry". Name should match the class (`PreprocessMethodSelecter`) purpose |
| `model_registry.py` | `app/` | `app/core/model_registry.py` | Model config/enum definitions are core business logic |
| `flask_csv_recording_state.py` | `app/` | `app/core/recording_state.py` | Thread-safe state singleton is core logic. Drop the `flask_csv_` prefix - it's not Flask-specific |
| `continuous_monitoring.py` | `app/` | `app/services/continuous_monitoring.py` | It's a background service/worker, distinct from request-handling code |
| `api_security.py` | `app/` | `app/middleware/api_security.py` | Auth/rate-limiting are request middleware concerns |

## Layered Architecture Pattern

The `app/` directory follows a **layered architecture** where each folder represents a distinct responsibility:

```
app/
  core/           --> What the app DOES (business logic, model config, state)
  services/       --> Background workers (monitoring thread)
  middleware/     --> Request interceptors (auth, rate limiting, CORS)
  routes/         --> HTTP endpoints (Flask blueprints)
  data_input/     --> Inbound I/O (InfluxDB fetching, preprocessing, resampling)
  data_output/    --> Outbound I/O (CSV export, InfluxDB marker writing)
  utils/          --> Shared helpers (logging, shared state)
  static/         --> Frontend assets (HTML)
```

### Layer Descriptions

- **`core/`** - Core business logic that is framework-agnostic. Contains the inference engine (`PreprocessMethodSelecter`), model configuration registry (`ModelName`, `ModelConfig`), and session state management (`RecordingState` singleton). These modules define *what* the system does.

- **`services/`** - Long-running background workers. Currently holds `ContinuousMonitor`, which runs in a daemon thread and periodically fetches sensor data + runs inference. Services consume `core/` modules but are not directly tied to HTTP requests.

- **`middleware/`** - Request-level concerns that wrap route handlers. API key validation (`require_api_key` decorator), rate limiting, and CORS setup. These are cross-cutting concerns applied to routes.

- **`routes/`** - Flask Blueprint definitions that map HTTP endpoints to handler functions. Each file groups related endpoints (`detection`, `recording`, `monitoring`). Routes orchestrate calls to `core/`, `data_input/`, and `data_output/`.

- **`data_input/`** - Everything related to getting data *into* the system: InfluxDB fetching, raw data preprocessing, accelerometer/barometer processing pipelines, resampling, sensor calibration, and CSV file loading.

- **`data_output/`** - Everything related to getting data *out* of the system: CSV export of detection results, InfluxDB marker injection (manual truth, user feedback).

- **`utils/`** - Shared utilities that don't fit into a specific layer: `shared_state.py` (global queues, locks, and mutable state accessed across modules), `model_logger.py` (structured logging helper).

### Dependency Flow

```
routes/ --> core/, data_input/, data_output/, middleware/
services/ --> core/, data_input/
middleware/ --> config/
core/ --> data_input/ (preprocessors loaded dynamically)
data_output/ --> utils/ (shared_state for CSV path tracking)
```

No circular dependencies. Lower layers never import from higher layers.
