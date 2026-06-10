#!/usr/bin/env bash
# teardown.sh -- Uninstall the caregiver chart and delete the namespace.
# WARNING: this deletes all PVCs including the SQLite patient store (patient data lost).
# Run from _6G_integration_v3_k3s/ as working directory.

set -euo pipefail

NAMESPACE="fall-dashboard"
RELEASE_NAME="caregiver"

echo "WARNING: This will delete all resources in namespace '${NAMESPACE}',"
echo "         including the SQLite patient store PVC (patient data will be lost)."
read -r -p "Type 'yes' to continue: " CONFIRM
if [ "${CONFIRM}" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo "Uninstalling Helm release '${RELEASE_NAME}' ..."
helm uninstall "${RELEASE_NAME}" --namespace "${NAMESPACE}"

echo "Deleting namespace '${NAMESPACE}' ..."
kubectl delete namespace "${NAMESPACE}"

echo "Done."
