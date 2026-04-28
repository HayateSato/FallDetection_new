# Tear down both charts and delete the namespaces.
# Run from _6G_Integration_v2_mqtt/ as cwd.

$ErrorActionPreference = "Continue"

Write-Host "==> Uninstalling charts..."
helm uninstall mcs-fall-detection -n mcs-fall-detection
helm uninstall mock-focus         -n mock-focus

Write-Host "==> Deleting namespaces..."
kubectl delete namespace mcs-fall-detection
kubectl delete namespace mock-focus

Write-Host ""
Write-Host "Cleanup complete. Verify:"
Write-Host "  kubectl get namespaces | findstr -E 'mcs-fall-detection|mock-focus'"
Write-Host "  (should print nothing)"
