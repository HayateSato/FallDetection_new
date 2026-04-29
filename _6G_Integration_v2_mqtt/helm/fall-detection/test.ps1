# In-cluster health probes for the fall-detection chart.
# Each test exec's into a pod and probes a service via K8s DNS — same path
# the running services use, so a green run here means real cross-service
# wiring is good (no port-forward involved).
#
# Run from _6G_Integration_v2_mqtt/ as cwd, after install.ps1 finishes.
#
# Notes for Windows PowerShell 5.1:
#   - python:3.11-slim images don't include curl OR wget. We use python3 urllib.
#   - Bash command arguments stay in single quotes so PowerShell doesn't
#     reinterpret them.

$ErrorActionPreference = "Continue"
$failed = 0

function Pass($msg) { Write-Host "  PASS: $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "  FAIL: $msg" -ForegroundColor Red; $script:failed++ }

# Probe one HTTP endpoint from inside a pod. Uses python3 urllib (works in
# python:3.11-slim images, which don't have curl).
function ProbeHttp($execPod, $targetUrl, $expectMatch, $label) {
    Write-Host ""
    Write-Host "[$label] $execPod -> $targetUrl"
    $py = "import urllib.request; print(urllib.request.urlopen('$targetUrl', timeout=5).read().decode())"
    $out = kubectl exec -n mcs-fall-detection $execPod -- python3 -c $py 2>&1
    if ($out -match $expectMatch) {
        Pass "match: $expectMatch"
    } else {
        Fail "did not match '$expectMatch' — got: $out"
    }
}

Write-Host ""
Write-Host "================================================"
Write-Host " fall-detection — in-cluster health probes"
Write-Host "================================================"

# --- Custom services ------------------------------------------------------
ProbeHttp "deploy/inference-server" "http://localhost:8001/health"      '"status":"ok"'         "1/8 inference-server self"
ProbeHttp "deploy/fall-dashboard"   "http://localhost:8002/api/patients" '"patients"'           "2/8 fall-dashboard self"
ProbeHttp "deploy/ml-dashboard"     "http://localhost:8004/api/status"  '"loaded_version"|"current"|"model"'  "3/8 ml-dashboard self"
ProbeHttp "deploy/server-health"    "http://localhost:8006/api/status"  '"overall"'             "4/8 server-health self"

# --- Cross-service via K8s DNS -------------------------------------------
# These prove the configmap URLs (INFERENCE_SERVER_URL, FALL_DASHBOARD_URL)
# resolve and the targets respond from inside the cluster.
ProbeHttp "deploy/server-health"  "http://inference-server:8001/health"  '"status":"ok"'  "5/8 server-health -> inference-server"
ProbeHttp "deploy/server-health"  "http://fall-dashboard:8002/api/patients" '"patients"'  "6/8 server-health -> fall-dashboard"
ProbeHttp "deploy/ml-dashboard"   "http://mlflow:5000/health"            'OK|ok'          "7/8 ml-dashboard -> mlflow (DNS rebinding allowed)"

# --- Grafana provisioning sanity ------------------------------------------
# Confirms gotcha #16 fix is in place — Grafana's API reports the 3 dashboards loaded.
Write-Host ""
Write-Host "[8/8] Grafana provisioning loaded dashboards"
$out = kubectl exec -n mcs-fall-detection deploy/grafana -- wget -qO- --user=admin --password=admin "http://localhost:3000/api/search?type=dash-db" 2>&1
if ($out -match "ml_server_overview" -and $out -match "model_performance" -and $out -match "fall_events_timeline") {
    Pass "all 3 dashboards present"
} else {
    Fail "expected 3 dashboards in /api/search response — got: $out"
}

# --- Summary --------------------------------------------------------------
Write-Host ""
Write-Host "================================================"
if ($failed -eq 0) {
    Write-Host " All probes passed." -ForegroundColor Green
    Write-Host "================================================"
    exit 0
} else {
    Write-Host " $failed probe(s) failed." -ForegroundColor Red
    Write-Host "================================================"
    exit 1
}
