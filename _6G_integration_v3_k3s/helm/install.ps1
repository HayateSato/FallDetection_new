# install.ps1 -- Install or upgrade the caregiver Helm chart into the FOCUS k3s cluster.
# Run from _6G_integration_v3_k3s/ as working directory.
#
# First-time setup -- run these steps in order:
#   1. Fill in all CHANGE_ME values in helm/values.yaml
#   2. Apply the Traefik TCP entrypoint (one-time cluster change):
#        kubectl apply -f helm/extras/traefik-mqtts-entrypoint.yaml
#   3. Create the registry pull secret in the namespace:
#        kubectl create namespace fall-dashboard
#        kubectl create secret docker-registry mcs-labs `
#          --docker-server=registry-smarko-health.de `
#          --docker-username=<user> `
#          --docker-password=<pass> `
#          --namespace fall-dashboard
#   4. Run this script.

$Namespace   = "fall-dashboard"
$ReleaseName = "caregiver"
$ChartPath   = "helm"

Write-Host "Installing caregiver chart (namespace: $Namespace) ..."

helm upgrade --install $ReleaseName $ChartPath `
    --namespace $Namespace `
    --create-namespace `
    --values helm/values.yaml `
    --wait `
    --timeout 120s

if (-not $?) { Write-Error "Helm install/upgrade failed"; exit 1 }

Write-Host ""
Write-Host "Pods:"
kubectl get pods -n $Namespace

Write-Host ""
Write-Host "Services:"
kubectl get svc -n $Namespace

Write-Host ""
Write-Host "IngressRoutes:"
kubectl get ingressroute,ingressroutetcp -n $Namespace
