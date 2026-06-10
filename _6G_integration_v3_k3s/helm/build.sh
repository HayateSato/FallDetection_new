#!/usr/bin/env bash
# build.sh -- Build and push the fall-dashboard image to the registry.
# Run from _6G_integration_v3_k3s/ as working directory.
# mock-app is NOT built -- it is a dev-only simulation, not deployed to k3s.

set -euo pipefail

TAG="${1:-latest}"
REGISTRY="registry-smarko-health.de"
IMAGE_NAME="fall-detection/fall-dashboard"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building fall-dashboard image: ${FULL_IMAGE}"
docker build -t "${FULL_IMAGE}" -f fall_dashboard/Dockerfile .

echo "Pushing ${FULL_IMAGE} ..."
docker push "${FULL_IMAGE}"

echo ""
echo "Done. Image pushed: ${FULL_IMAGE}"
if [ "${TAG}" != "latest" ]; then
    echo "Update values_production.yaml fallDashboard.image if you used a non-latest tag."
fi
