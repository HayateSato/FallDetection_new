# Cross-namespace integration tests.
# Run after install.ps1 has finished and all pods are Running.
#
# IMPORTANT for Windows PowerShell 5.1 compatibility:
#   - Bash command arguments are wrapped in SINGLE quotes so PowerShell does
#     not interpret the contents (no expansion, no operator parsing)
#   - No && or || (PS 5.1 doesn't support pipeline-chain operators)
#   - Each kubectl exec on a single line (no backtick continuation, which can
#     interact badly with native-exe argument parsing)

$ErrorActionPreference = "Continue"
$failed = 0

function Pass($msg) { Write-Host "  PASS: $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "  FAIL: $msg" -ForegroundColor Red; $script:failed++ }

Write-Host ""
Write-Host "================================================"
Write-Host " 12.5.6 - Cross-namespace integration tests"
Write-Host "================================================"

# --- [1/4] DNS resolution test --------------------------------------------
# Validates that pods in mcs-fall-detection can resolve the FQDN of a service
# in the mock-focus namespace. getent is available in any glibc base image
# (python:3.11-slim is Debian-based).
Write-Host ""
Write-Host "[1/4] DNS: inference-server -> mock-fhir-server.mock-focus"
$out = kubectl exec -n mcs-fall-detection deploy/inference-server -- sh -c 'getent hosts mock-fhir-server.mock-focus.svc.cluster.local' 2>&1
if ($LASTEXITCODE -eq 0) {
    Pass "DNS resolved: $out"
} else {
    Fail "DNS lookup failed: $out"
}

# --- [2/4] Reverse direction HTTP -----------------------------------------
# Validates the path Isa's real Patient Dashboard will use: a pod in the
# FOCUS namespace fetches /api/patients from our fall-dashboard.
# Uses python3 urllib (wget not available in python:3.11-slim).
Write-Host ""
Write-Host "[2/4] HTTP: mock-patient-dashboard -> fall-dashboard /api/patients"
$out = kubectl exec -n mock-focus deploy/mock-patient-dashboard -- python3 -c "import urllib.request; print(urllib.request.urlopen('http://fall-dashboard.mcs-fall-detection.svc.cluster.local:8002/api/patients').read().decode())" 2>&1
if ($out -match '"patients"') {
    Pass "/api/patients reachable cross-namespace"
} else {
    Fail "GET /api/patients failed: $out"
}

# --- [3/4] FHIR push direction --------------------------------------------
# Validates the (optional) FHIR-push path that fires when FHIR_SERVER_URL is
# configured on the inference server.
# Uses python3 urllib (wget not available in python:3.11-slim).
Write-Host ""
Write-Host "[3/4] HTTP: inference-server -> mock-fhir-server /health"
$out = kubectl exec -n mcs-fall-detection deploy/inference-server -- python3 -c "import urllib.request; print(urllib.request.urlopen('http://mock-fhir-server.mock-focus.svc.cluster.local:8003/health').read().decode())" 2>&1
if ($out -match '"status":"ok"') {
    Pass "mock-fhir /health reachable cross-namespace"
} else {
    Fail "GET mock-fhir /health failed: $out"
}

# --- [4/4] InfluxDB reachable ---------------------------------------------
# In-namespace sanity check that the StatefulSet came up correctly.
Write-Host ""
Write-Host "[4/4] InfluxDB health (within mock-focus namespace)"
$out = kubectl exec -n mock-focus statefulset/mock-influxdb -- curl -s http://localhost:8086/health 2>&1
if ($out -match '"status":"pass"') {
    Pass "InfluxDB healthy"
} else {
    Fail "GET influxdb /health failed: $out"
}

# --- Summary --------------------------------------------------------------
Write-Host ""
Write-Host "================================================"
if ($failed -eq 0) {
    Write-Host " All cross-namespace tests passed." -ForegroundColor Green
    Write-Host " Open http://localhost:30090/ to exercise SSE manually."
    Write-Host "================================================"
    exit 0
} else {
    Write-Host " $failed test(s) failed." -ForegroundColor Red
    Write-Host "================================================"
    exit 1
}
