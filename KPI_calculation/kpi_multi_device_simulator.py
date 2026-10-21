"""
Multi-device fall-detection KPI simulator.

Mirrors the mobile app's exact request/MQTT behaviour (6g-path-app), run
concurrently for N simulated patients/devices instead of pressing
"Simulate (real POST)" by hand N times.

KPI1 -- inference round trip (POST /predict):
    t0 = perf_counter() -> POST /predict -> t1 = perf_counter()
    kpi1_predict_ms = (t1 - t0) * 1000
Mirrors 6g-path-app/src/services/FallDetectionService.js:_performPredict
(same URL, X-API-Key header, request body shape, 15s timeout).

KPI2 -- MQTT alert round trip:
    publish fall/alert/<patient_id> (same JSON shape as
    FallConfirmationOverlay.jsx), subscribe to fall/ack/<patient_id>
    (mqttAckTopicTemplate), match the ack by observation_id,
    kpi2_alert_rtt_ms = ack_time - publish_time.
Mirrors 6g-path-app/src/services/MqttPublisher.js:trackAlertForAck /
_handleIncoming, and the caregiver-dashboard-side echo added in
_6G_integration_v3_k3s/fall_dashboard/mqtt_listener.py:_publish_ack.

Config values below mirror 6g-path-app/src/config/fallDetectionConfig.js --
keep them in sync if that file changes.

Usage:
    python kpi_multi_device_simulator.py --num-devices 8
    python kpi_multi_device_simulator.py --num-devices 8 --skip-alert   # KPI1 only
    python kpi_multi_device_simulator.py --help
"""

import argparse
import csv
import json
import logging
import ssl
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import paho.mqtt.client as mqtt

logger = logging.getLogger("kpi_simulator")

# --- Config (mirrors 6g-path-app/src/config/fallDetectionConfig.js) --------
INFERENCE_BASE_URL = "https://fall-api.smarko-health.de"
INFERENCE_API_KEY  = "J7gEYm2mOgaLqRRKhCAOZZpYOz_VnoHo-CwlFUJBMfc"
REQUEST_TIMEOUT_S   = 15   # requestTimeoutMs

MQTT_HOST                = "fall-mqtt.smarko-health.de"
MQTT_PORT                = 443
MQTT_PATH                = "/"
MQTT_USE_SSL             = True
MQTT_ACK_TOPIC_TEMPLATE  = "fall/ack/{patient_id}"   # mqttAckTopicTemplate '%s' -> '{patient_id}'
MQTT_ACK_TIMEOUT_S       = 15                         # mqttAckTimeoutMs

# --- Fake patient_id -> device_id (MAC) map -------------------------------
# Edit this by hand if you need specific patient IDs or MAC addresses --
# e.g. to match patients already registered in the caregiver dashboard's
# PATIENT_IDS/MAC_IDS env vars, or to add/remove simulated devices.
# --num-devices takes the first N entries in this dict, in order.
FAKE_DEVICES = {
    "sim-9001": "6c:1d:eb:04:a9:01",
    "sim-9002": "6c:1d:eb:04:a9:02",
    "sim-9003": "6c:1d:eb:04:a9:03",
    "sim-9004": "6c:1d:eb:04:a9:04",
    "sim-9005": "6c:1d:eb:04:a9:05",
    "sim-9006": "6c:1d:eb:04:a9:06",
    "sim-9007": "6c:1d:eb:04:a9:07",
    "sim-9008": "6c:1d:eb:04:a9:08",
}

HERE = Path(__file__).parent
DEFAULT_PAYLOADS = sorted(HERE.glob("*_predict_payload.json"))


def _iso_now_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class DeviceResult:
    def __init__(self, patient_id, device_id):
        self.patient_id = patient_id
        self.device_id = device_id
        self.observation_id = None
        self.fall_detected = None
        self.confidence = None
        self.model_version = None
        self.kpi1_predict_ms = None
        self.kpi2_alert_rtt_ms = None
        self.ack_received = False
        self.ack_received_at = None
        self.error = None

    def as_row(self) -> dict:
        return {
            "patient_id":        self.patient_id,
            "device_id":         self.device_id,
            "observation_id":    self.observation_id or "",
            "fall_detected":     self.fall_detected,
            "confidence":        f"{self.confidence:.4f}" if self.confidence is not None else "",
            "model_version":     self.model_version or "",
            "kpi1_predict_ms":   f"{self.kpi1_predict_ms:.1f}" if self.kpi1_predict_ms is not None else "",
            "kpi2_alert_rtt_ms": f"{self.kpi2_alert_rtt_ms:.1f}" if self.kpi2_alert_rtt_ms is not None else "",
            "ack_received":      self.ack_received,
            "error":             self.error or "",
        }


def run_device(patient_id: str, device_id: str, payload_path: Path, args, start_delay_s: float = 0.0) -> DeviceResult:
    if start_delay_s > 0:
        time.sleep(start_delay_s)

    result = DeviceResult(patient_id, device_id)
    logger.info(f"[start] {patient_id} ({device_id})  delay={start_delay_s:.1f}s  payload={payload_path.name}")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["patient_id"] = patient_id
    payload["device_id"] = device_id

    # ---- KPI1: POST /predict (FallDetectionService.js:_performPredict) ----
    headers = {"Content-Type": "application/json", "X-API-Key": args.api_key}
    url = f"{args.inference_url.rstrip('/')}/predict"

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S)
        t1 = time.perf_counter()
        resp.raise_for_status()
    except Exception as exc:
        result.error = f"predict failed: {exc}"
        return result

    result.kpi1_predict_ms = (t1 - t0) * 1000

    body = resp.json()
    result.observation_id = body.get("observation_id")
    inference = body.get("inference", {})
    result.fall_detected  = inference.get("fall_detected")
    result.confidence     = inference.get("confidence")
    result.model_version  = inference.get("model_version")

    if not args.skip_alert:
        _publish_alert_and_wait_ack(result, args)

    return result


def _publish_alert_and_wait_ack(result: DeviceResult, args) -> None:
    ack_topic   = MQTT_ACK_TOPIC_TEMPLATE.format(patient_id=result.patient_id)
    alert_topic = f"fall/alert/{result.patient_id}"

    ack_event  = threading.Event()
    ack_holder = {}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(ack_topic, qos=1)

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            obs = data.get("observation_id")
        except Exception:
            obs = msg.payload.decode("utf-8", errors="replace").strip()
            data = {"observation_id": obs}
        if obs == result.observation_id:
            ack_holder["recv_perf"] = time.perf_counter()
            ack_holder["data"] = data
            ack_event.set()

    client_id = f"sim-falldet-{result.patient_id}-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        transport="websockets",
    )
    client.ws_set_options(path=args.mqtt_path)
    if args.mqtt_use_ssl:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
        client.loop_start()

        # Give the subscribe a moment to land before publishing, mirroring
        # _ensureAckSubscription() running ahead of trackAlertForAck() in the app.
        time.sleep(0.3)

        alert_payload = {
            "observation_id":    result.observation_id,
            "patient_id":        result.patient_id,
            "device_id":         result.device_id,
            "timestamp":         _iso_now_ms(),
            "alert_time":        _iso_now_ms(),
            "fall_detected":     bool(result.fall_detected),
            "confidence":        result.confidence,
            "model_version":     result.model_version,
            "fhir_observation":  None,
            "patient_confirmed": args.patient_confirmed,
            "needs_help":        args.needs_help,
            "simulated":         True,
        }

        if args.confirmation_delay_s > 0:
            time.sleep(args.confirmation_delay_s)

        publish_perf = time.perf_counter()
        client.publish(alert_topic, json.dumps(alert_payload), qos=1)

        if ack_event.wait(timeout=MQTT_ACK_TIMEOUT_S):
            result.ack_received = True
            result.kpi2_alert_rtt_ms = (ack_holder["recv_perf"] - publish_perf) * 1000
            result.ack_received_at = ack_holder["data"].get("received_at")
        else:
            result.ack_received = False
    except Exception as exc:
        result.error = (result.error + "; " if result.error else "") + f"mqtt failed: {exc}"
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass


def compute_stats(values) -> dict:
    """min/max/mean/var/std/N over the non-None values in `values`.
    Sample variance/stdev (ddof=1); 0.0 when N==1, all-None when N==0."""
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "var": None, "std": None}
    return {
        "n":    n,
        "min":  min(clean),
        "max":  max(clean),
        "mean": statistics.mean(clean),
        "var":  statistics.variance(clean) if n > 1 else 0.0,
        "std":  statistics.stdev(clean) if n > 1 else 0.0,
    }


def _log_stats(label: str, stats: dict) -> None:
    if stats["n"] == 0:
        logger.info(f"{label}: no successful samples (N=0)")
        return
    logger.info(
        f"{label}: N={stats['n']}  min={stats['min']:.1f}  max={stats['max']:.1f}  "
        f"mean={stats['mean']:.1f}  var={stats['var']:.1f}  std={stats['std']:.1f}  (ms)"
    )


def build_devices(args):
    payload_files = [Path(p) for p in args.payloads] if args.payloads else DEFAULT_PAYLOADS
    if not payload_files:
        raise SystemExit(
            f"No *_predict_payload.json files found in {HERE} -- "
            "run extract_predict_payload.py first."
        )

    fake_devices = list(FAKE_DEVICES.items())
    if args.num_devices > len(fake_devices):
        raise SystemExit(
            f"--num-devices {args.num_devices} exceeds the {len(fake_devices)} entries "
            f"in FAKE_DEVICES -- add more patient_id/device_id pairs to that dict "
            f"at the top of {Path(__file__).name} first."
        )

    devices = []
    for i in range(args.num_devices):
        patient_id, device_id = fake_devices[i]
        payload_path = payload_files[i % len(payload_files)]
        devices.append((patient_id, device_id, payload_path))
    return devices


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-devices", type=int, default=8,
        help="How many devices to simulate -- takes the first N entries from "
             "the FAKE_DEVICES dict at the top of this file",
    )
    parser.add_argument(
        "--payloads", nargs="*",
        help="Specific *_predict_payload.json files to cycle through "
             "(default: all found in this folder)",
    )
    parser.add_argument("--inference-url", default=INFERENCE_BASE_URL)
    parser.add_argument("--api-key", default=INFERENCE_API_KEY)
    parser.add_argument("--mqtt-host", default=MQTT_HOST)
    parser.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    parser.add_argument("--mqtt-path", default=MQTT_PATH)
    parser.add_argument("--mqtt-use-ssl", action=argparse.BooleanOptionalAction, default=MQTT_USE_SSL)
    parser.add_argument("--patient-confirmed", default="yes", choices=["yes", "no", "not_answered"])
    parser.add_argument("--needs-help", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--confirmation-delay-s", type=float, default=0.0,
        help="Simulate the patient-confirmation popup wait before publishing "
             "the alert (real app: up to 20s). Default 0 = publish immediately, "
             "best for pure KPI2 (MQTT) measurement.",
    )
    parser.add_argument(
        "--stagger-s", type=float, default=0.0,
        help="Only used with --scenario concurrent. Delay (seconds) between each "
             "device's start time: device i starts at i * stagger-s. Default 0 = "
             "all devices fire at exactly the same time.",
    )
    parser.add_argument(
        "--scenario", choices=["sequential", "concurrent"], default="concurrent",
        help="scenario 1 'sequential': one device fully completes (predict + "
             "alert + ack wait) before the next one starts. "
             "scenario 2 'concurrent' (default): all devices run at once "
             "(optionally spread out with --stagger-s).",
    )
    parser.add_argument(
        "--skip-alert", action="store_true",
        help="Only run KPI1 (predict), skip MQTT alert/ack entirely",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV output path (default: kpi_results_<timestamp>.csv in this folder)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Directory for the run's log file (default: KPI_calculation/logs/)",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"kpi_run_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )

    devices = build_devices(args)
    out_path = Path(args.output) if args.output else HERE / f"kpi_results_{run_id}.csv"

    logger.info(f"Simulating {len(devices)} device(s) against "
                f"{args.inference_url} / {args.mqtt_host}:{args.mqtt_port}  "
                f"(scenario={args.scenario}"
                + (f", stagger={args.stagger_s}s)" if args.scenario == "concurrent" else ")"))
    for pid, did, ppath in devices:
        logger.info(f"  {pid}  ({did})  <- {ppath.name}")

    def _log_done(r: "DeviceResult") -> None:
        logger.info(
            f"[done]  {r.patient_id}  kpi1_predict_ms={r.kpi1_predict_ms}  "
            f"kpi2_alert_rtt_ms={r.kpi2_alert_rtt_ms}  ack={r.ack_received}  "
            f"fall_detected={r.fall_detected}  error={r.error}"
        )

    if args.scenario == "sequential":
        # scenario 1 -- one device fully completes before the next one starts.
        results = []
        for pid, did, ppath in devices:
            r = run_device(pid, did, ppath, args, start_delay_s=0.0)
            _log_done(r)
            results.append(r)
    else:
        # scenario 2 -- all devices run at once (optionally spread out via --stagger-s).
        results_by_pid = {}
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            future_to_pid = {
                pool.submit(run_device, pid, did, ppath, args, i * args.stagger_s): pid
                for i, (pid, did, ppath) in enumerate(devices)
            }
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                r = future.result()
                results_by_pid[pid] = r
                _log_done(r)
        results = [results_by_pid[pid] for pid, _, _ in devices]  # restore original order

    # ---- Write CSV ---------------------------------------------------
    fieldnames = [
        "patient_id", "device_id", "observation_id", "fall_detected", "confidence",
        "model_version", "kpi1_predict_ms", "kpi2_alert_rtt_ms", "ack_received", "error",
    ]
    kpi1_stats = compute_stats([r.kpi1_predict_ms for r in results])
    kpi2_stats = compute_stats([r.kpi2_alert_rtt_ms for r in results])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_row())

        stats_writer = csv.writer(f)
        stats_writer.writerow([])
        stats_writer.writerow(["metric", "n", "min", "max", "mean", "var", "std"])
        for label, s in (("kpi1_predict_ms", kpi1_stats), ("kpi2_alert_rtt_ms", kpi2_stats)):
            stats_writer.writerow([
                label, s["n"], s["min"], s["max"], s["mean"], s["var"], s["std"],
            ])

    # ---- Summary table (goes to both console and the log file) --------
    logger.info(f"{'patient_id':<14} {'kpi1_ms':>10} {'kpi2_rtt_ms':>12} {'ack':>5} {'fall':>7}  error")
    for r in results:
        kpi1 = f"{r.kpi1_predict_ms:.1f}" if r.kpi1_predict_ms is not None else "-"
        kpi2 = f"{r.kpi2_alert_rtt_ms:.1f}" if r.kpi2_alert_rtt_ms is not None else "-"
        logger.info(f"{r.patient_id:<14} {kpi1:>10} {kpi2:>12} "
                    f"{('yes' if r.ack_received else 'no'):>5} {str(r.fall_detected):>7}  {r.error}")

    logger.info("")
    _log_stats("KPI1 predict", kpi1_stats)
    _log_stats("KPI2 alert_rtt", kpi2_stats)

    logger.info(f"Wrote CSV: {out_path}")
    logger.info(f"Wrote log: {log_path}")


if __name__ == "__main__":
    main()
