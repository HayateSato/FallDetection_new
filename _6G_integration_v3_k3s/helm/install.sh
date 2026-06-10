#!/usr/bin/env bash
# install.sh -- Install or upgrade the caregiver Helm chart into the FOCUS k3s cluster.
# Run from _6G_integration_v3_k3s/ as working directory.
#
# First-time setup -- run these steps in order:
#   1. Fill in all CHANGE_ME values in helm/values_production.yaml
#   2. Create the registry pull secret in the namespace:
#        kubectl create namespace fall-dashboard
#        kubectl create secret docker-registry mcs-labs \
#          --docker-server=registry-smarko-health.de \
#          --docker-username=<user> \
#          --docker-password=<pass> \
#          --namespace fall-dashboard
#   3. Run this script.

set -euo pipefail

NAMESPACE="fall-dashboard"
RELEASE_NAME="caregiver"
CHART_PATH="helm"
VALUES_FILE="helm/values_production.yaml"

echo "Installing caregiver chart (namespace: ${NAMESPACE}) ..."

helm upgrade --install "${RELEASE_NAME}" "${CHART_PATH}" \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    --values "${VALUES_FILE}" \
    --wait \
    --timeout 120s

echo ""
echo "Pods:"
kubectl get pods -n "${NAMESPACE}"

echo ""
echo "Services:"
kubectl get svc -n "${NAMESPACE}"

echo ""
echo "IngressRoutes:"
kubectl get ingressroute -n "${NAMESPACE}" 2>/dev/null || true
