# Tear down the fall-detection chart and delete its namespace.
# Run from _6G_Integration_v2_mqtt/ as cwd.

$ErrorActionPreference = "Continue"

Write-Host "==> Uninstalling fall-detection chart..."
helm uninstall mcs-fall-detection -n mcs-fall-detection

Write-Host "==> Deleting namespace (also removes PVCs)..."
kubectl delete namespace mcs-fall-detection

Write-Host ""
Write-Host "Cleanup complete. Verify:"
Write-Host "  kubectl get namespaces | findstr mcs-fall-detection"
Write-Host "  (should print nothing)"
