# Install both charts into Docker Desktop K8s for the cross-namespace dry-run.
# Prereqs:
#   1. Docker Desktop > Settings > Kubernetes > Enable Kubernetes
#   2. kubectl context is "docker-desktop" (`kubectl config current-context`)
#   3. Run helm/mock-focus/build.ps1 first to build local images
#   4. Run helm/fall-detection/build.ps1 (or equivalent) to build the real chart's images
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

Write-Host "==> Installing fall-detection chart..."
helm upgrade --install mcs-fall-detection ./helm/fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m

Write-Host ""
Write-Host "==> Pod status (mock-focus):"
kubectl get pods -n mock-focus

Write-Host ""
Write-Host "==> Pod status (mcs-fall-detection):"
kubectl get pods -n mcs-fall-detection

Write-Host ""
Write-Host "==> Open the mock Patient Dashboard:"
Write-Host "    http://localhost:30090/"
Write-Host ""
Write-Host "==> Run cross-namespace tests:"
Write-Host "    .\helm\mock-focus\test.ps1"
