## How It Works

### On the server laptop (run the API):

`git clone <repo>
pip install -r requirements.server.txt
cp .env.server.example .env
# edit .env: set MODEL_VERSION=v0
python server.py          # starts on 0.0.0.0:8000
ngrok http 8000           # exposes e.g. https://abc123.ngrok-free.app`

### On the client side - just 2 lines in `.env`:

`INFERENCE_MODE=remote
REMOTE_SERVER_URL=https://abc123.ngrok-free.app
REMOTE_API_KEY=           # if server has PUBLIC_ENDPOINT_ENABLED=true`

Then run `python main.py` as usual. The client:

1. Fetches sensor data from InfluxDB (as before)
2. Sends the raw sensor arrays via HTTP POST to the server's `/predict` endpoint
3. Server handles all preprocessing + inference
4. Client receives the result and continues normally (CSV export, notifications, etc.)

### What was changed/created:

| File | Change |
| --- | --- |
| remote_inference.py | **New** - HTTP client that sends sensor data to the server |
| settings.py | Added `INFERENCE_MODE`, `REMOTE_SERVER_URL`, `REMOTE_API_KEY` |
| .env.example | Added remote inference settings section |
| main.py | Initializes `RemoteInferenceClient` when mode=remote, passes to Flask app + monitoring |
| detection.py | Added remote branch: sends raw data to server instead of local inference |
| continuous_monitoring.py | Added remote branch for the background monitoring loop |
| requirements.txt | Added `requests` |

The key design: the `INFERENCE_MODE` toggle switches between local and remote inference without changing anything else. Everything else (InfluxDB queries, CSV export, UI, notifications) stays on the client.