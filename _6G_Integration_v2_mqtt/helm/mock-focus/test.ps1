# Cross-namespace integration tests.
# Run after install.ps1 has finished and all pods are Running.

$ErrorActionPreference = "Continue"
$failed = 0

function Pass($msg) { Write-Host "  PASS: $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "  FAIL: $msg" -ForegroundColor Red; $script:failed++ }

Write-Host ""
Write-Host "================================================"
Write-Host " 12.5.6 — Cross-namespace integration tests"
Write-Host "================================================"

# --- DNS resolution test ---------------------------------------------------
Write-Host ""
Write-Host "[1/4] DNS: inference-server (mcs-fall-detection) -> mock-fhir-server (mock-focus)"
$out = kubectl exec -n mcs-fall-detection deploy/inference-server -- `
    sh -c "getent hosts mock-fhir-server.mock-focus.svc.cluster.local || nslookup mock-fhir-server.mock-focus.svc.cluster.local" 2>&1
if ($LASTEXITCODE -eq 0) { Pass "DNS resolved" } else { Fail "DNS lookup failed: $out" }

# --- Reverse direction HTTP -----------------------------------------------
Write-Host ""
Write-Host "[2/4] HTTP: mock-patient-dashboard (mock-focus) -> fall-dashboard (mcs-fall-detection)"
$out = kubectl exec -n mock-focus deploy/mock-patient-dashboard -- `
    sh -c "wget -qO- http://fall-dashboard.mcs-fall-detection.svc.cluster.local:8002/api/patients" 2>&1
if ($out -match '"patients"') { Pass "/api/patients reachable cross-namespace" } else { Fail "GET /api/patients failed: $out" }

# --- FHIR push direction --------------------------------------------------
Write-Host ""
Write-Host "[3/4] HTTP: inference-server (mcs-fall-detection) -> mock-fhir-server (mock-focus)"
$out = kubectl exec -n mcs-fall-detection deploy/inference-server -- `
    sh -c "wget -qO- http://mock-fhir-server.mock-focus.svc.cluster.local:8003/health" 2>&1
if ($out -match '"status":"ok"') { Pass "/health reachable cross-namespace" } else { Fail "GET mock-fhir/health failed: $out" }

# --- InfluxDB reachable ---------------------------------------------------
Write-Host ""
Write-Host "[4/4] HTTP: mock-influxdb health endpoint"
$out = kubectl exec -n mock-focus statefulset/mock-influxdb -- `
    sh -c "wget -qO- http://localhost:8086/health" 2>&1
if ($out -match '"status":"pass"') { Pass "InfluxDB healthy" } else { Fail "GET influxdb/health failed: $out" }

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
