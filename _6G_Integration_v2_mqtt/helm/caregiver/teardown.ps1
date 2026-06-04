# teardown.ps1 -- Uninstall the caregiver chart and delete the namespace.
# WARNING: this deletes all PVCs including the SQLite patient store.
# Run from _6G_Integration_v2_mqtt/ as working directory.

$Namespace   = "fall-dashboard"
$ReleaseName = "caregiver"

Write-Host "WARNING: This will delete all resources in namespace '$Namespace',"
Write-Host "         including the SQLite patient store PVC (patient data lost)."
$confirm = Read-Host "Type 'yes' to continue"
if ($confirm -ne "yes") { Write-Host "Aborted."; exit 0 }

Write-Host "Uninstalling Helm release '$ReleaseName' ..."
helm uninstall $ReleaseName --namespace $Namespace

Write-Host "Deleting namespace '$Namespace' ..."
kubectl delete namespace $Namespace

Write-Host "Done."
