# Build the four custom images for the fall-detection chart.
# Run from _6G_Integration_v2_mqtt/ as cwd.
#
# All four images are tagged with the registry prefix from values.yaml so
# Docker Desktop K8s (containerd, pullPolicy: Never) finds them by the exact
# name the deployment templates render. See project_k8s_local_testing.md
# gotcha #5 for why the prefix is required even locally.

$ErrorActionPreference = "Stop"
$REGISTRY = "registry.example.com"

function BuildImage($dockerfile, $imageName) {
    Write-Host "==> Building $imageName ..."
    docker build -f $dockerfile -t "$REGISTRY/${imageName}:latest" .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$imageName build failed"
        exit 1
    }
}

BuildImage  "inference_server/Dockerfile"  "inference-server"
BuildImage  "fall_dashboard/Dockerfile"    "fall-dashboard"
BuildImage  "ml_dashboard/Dockerfile"      "ml-dashboard"
BuildImage  "server_health/Dockerfile"     "server-health"

Write-Host ""
Write-Host "All four images built. Verify:"
Write-Host "  docker images | findstr `"$REGISTRY`""
Write-Host ""
Write-Host "Next: .\helm\fall-detection\install.ps1"
