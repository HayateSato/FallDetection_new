# Install (or upgrade) the fall-detection chart on Docker Desktop K8s.
# Prereqs:
#   1. Docker Desktop > Settings > Kubernetes > Enable Kubernetes
#   2. kubectl context = "docker-desktop" (`kubectl config current-context`)
#   3. Run helm/fall-detection/build.ps1 first to build all 4 custom images
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

Write-Host "==> Installing/upgrading fall-detection chart..."
helm upgrade --install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m
if ($LASTEXITCODE -ne 0) {
    Write-Error "helm upgrade failed. Run 'kubectl get pods -n mcs-fall-detection' to inspect."
    exit 1
}

Write-Host ""
Write-Host "==> Pod status:"
kubectl get pods -n mcs-fall-detection

Write-Host ""
Write-Host "All 10 services should be Running (migrate-job: Completed)."
Write-Host ""
Write-Host "Next:"
Write-Host "  .\helm\fall-detection\port-forward.ps1   # opens all UI tunnels in separate windows"
Write-Host "  .\helm\fall-detection\test.ps1           # in-cluster health probes"
