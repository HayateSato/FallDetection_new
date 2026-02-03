# Fall Detection API Documentation

## Overview

This API provides real-time fall detection inference using machine learning. Send accelerometer and barometer sensor data, and receive fall detection predictions.

## Authentication

All requests to protected endpoints require an API key.

**Provide the API key via:**
- **Header (recommended):** `X-API-Key: <your-api-key>`
- **Query parameter:** `?api_key=<your-api-key>`

## Rate Limits

- **30 requests per minute** per IP address
- Exceeding the limit returns `429 Too Many Requests`

---

## Endpoints

### 1. Trigger Fall Detection

Analyze sensor data and detect falls.

```
POST /trigger
```

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Your API key |
| `Content-Type` | Yes | `application/json` |

#### Request Body

```json
{
  "participant_name": "John Doe",
  "participant_gender": "male",
  "ground_truth_fall": 0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `participant_name` | string | No | Name of the participant (default: "unknown") |
| `participant_gender` | string | No | Gender: "male", "female", or "other" (default: "unknown") |
| `ground_truth_fall` | integer | No | 1 if this is a known fall event, 0 otherwise (default: 0) |

#### Response

**Success (200 OK):**

```json
{
  "message": "Fall detection completed (V3).",
  "data_source": "influx",
  "fall_detected": true,
  "model_version": "v3",
  "model_name": "V3",
  "result": "High confidence fall detection",
  "confidence": 0.8234,
  "threshold": 0.5,
  "sampling_rate": 50.0,
  "window_size": 450,
  "window_duration_seconds": 9,
  "num_features": 20,
  "acc_features": 16,
  "baro_features": 4,
  "baro_samples": 225,
  "participant_name": "John Doe",
  "participant_gender": "male",
  "ground_truth_fall": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fall_detected` | boolean | `true` if a fall was detected |
| `confidence` | float | Confidence score (0.0 to 1.0) |
| `threshold` | float | Decision threshold used |
| `result` | string | Human-readable result description |
| `model_version` | string | Model version used (v0-v5) |

#### Error Responses

| Code | Description |
|------|-------------|
| 401 | Missing API key |
| 403 | Invalid API key |
| 404 | No sensor data available |
| 429 | Rate limit exceeded |
| 500 | Server error |
| 503 | Database connection error |

---

### 2. Get Model Info

Get information about the currently loaded model.

```
GET /model/info
```

#### Response

```json
{
  "name": "V3",
  "version": "v3",
  "description": "ACC (statistical) + BARO (paper slope-limit)",
  "uses_barometer": true,
  "num_features": 20,
  "acc_features": 16,
  "baro_features": 4,
  "acc_preprocessing": "v1_features",
  "baro_preprocessing": "v2_paper"
}
```

---

### 3. Get Monitoring Status

Check the current system status.

```
GET /monitoring/status
```

#### Response

```json
{
  "monitoring_enabled": true,
  "is_running": true,
  "data_source": "influx",
  "model_version": "v3",
  "model_name": "V3",
  "uses_barometer": true,
  "acc_sample_rate": 50,
  "baro_sample_rate": 25,
  "features": 20
}
```

---

## Code Examples

### Python

```python
import requests

API_URL = "https://your-ngrok-url.ngrok.io"
API_KEY = "your-api-key-here"

# Trigger fall detection
response = requests.post(
    f"{API_URL}/trigger",
    headers={
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    },
    json={
        "participant_name": "Test User",
        "participant_gender": "male",
        "ground_truth_fall": 0
    }
)

result = response.json()
print(f"Fall detected: {result['fall_detected']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL

```bash
curl -X POST "https://your-ngrok-url.ngrok.io/trigger" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"participant_name": "Test User"}'
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

const API_URL = 'https://your-ngrok-url.ngrok.io';
const API_KEY = 'your-api-key-here';

async function detectFall() {
  const response = await axios.post(
    `${API_URL}/trigger`,
    {
      participant_name: 'Test User',
      participant_gender: 'male',
      ground_truth_fall: 0
    },
    {
      headers: {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
      }
    }
  );

  console.log('Fall detected:', response.data.fall_detected);
  console.log('Confidence:', response.data.confidence);
}

detectFall();
```

---

## Important Notes

1. **Data Source**: The API uses real-time sensor data from InfluxDB. Ensure sensor data is being streamed before calling `/trigger`.

2. **Window Size**: The model analyzes 9 seconds of data (450 samples at 50Hz). Ensure sufficient data is available.

3. **Confidence Interpretation**:
   - `> 0.75`: High confidence fall
   - `0.60 - 0.75`: Moderate confidence fall
   - `0.50 - 0.60`: Low confidence fall (at threshold)
   - `< 0.50`: No fall detected

4. **Model Versions**: Different models are available (v0-v5). The server configuration determines which model is used.

---

## Support

For issues or questions, contact the Fall Detection team.
