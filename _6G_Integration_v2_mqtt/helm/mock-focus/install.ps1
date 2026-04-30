# Install the mock-focus chart into Docker Desktop K8s for the cross-namespace dry-run.
# This script ONLY installs mock-focus (mock-fhir, mock-influxdb, mock-patient-dashboard).
# fall-dashboard / inference-server / etc. live in mcs-fall-detection — install that
# chart separately via .\helm\fall-detection\install.ps1.
#
# Prereqs:
#   1. Docker Desktop > Settings > Kubernetes > Enable Kubernetes
#   2. kubectl context is "docker-desktop" (`kubectl config current-context`)
#   3. Run .\helm\mock-focus\build.ps1 first to build the mock images
#   4. (Recommended) Run .\helm\fall-detection\install.ps1 first so cross-NS
#      targets (fall-dashboard.mcs-fall-detection...) actually resolve. mock-focus
#      pods will start either way, but the dashboard at :30090 will show errors
#      until fall-dashboard is reachable.
#
# Run from _6G_Integration_v2_mqtt/ as cwd.

$ErrorActionPreference = "Stop"

Write-Host "==> Verifying kubectl context..."
$ctx = kubectl config current-context
if ($ctx -ne "docker-desktop") {
    Write-Warning "kubectl context is '$ctx', expected 'docker-desktop'"
    Write-Warning "Switch with: kubectl config use-context docker-desktop"
    exit 1
}

Write-Host "==> Installing mock-focus chart..."
helm upgrade --install mock-focus ./helm/mock-focus `
    --namespace mock-focus `
    --create-namespace `
    --wait --timeout 5m

Write-Host ""
Write-Host "==> Pod status (mock-focus):"
kubectl get pods -n mock-focus

Write-Host ""
Write-Host "==> Open the mock Patient Dashboard:"
Write-Host "    http://localhost:30090/"
Write-Host ""
Write-Host "==> Run cross-namespace tests (requires fall-detection chart installed too):"
Write-Host "    .\helm\mock-focus\test.ps1"
