# Tear down ONLY the mock-focus chart and delete its namespace.
# fall-detection is a separate chart — uninstall it via .\helm\fall-detection\teardown.ps1.
#
# Run from _6G_Integration_v2_mqtt/ as cwd.

$ErrorActionPreference = "Continue"

Write-Host "==> Uninstalling mock-focus chart..."
helm uninstall mock-focus -n mock-focus

Write-Host "==> Deleting namespace (also removes PVCs)..."
kubectl delete namespace mock-focus

Write-Host ""
Write-Host "Cleanup complete. Verify:"
Write-Host "  kubectl get namespaces | findstr mock-focus"
Write-Host "  (should print nothing)"
Write-Host ""
Write-Host "To also tear down fall-detection: .\helm\fall-detection\teardown.ps1"
