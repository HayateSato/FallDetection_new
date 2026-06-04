# build.ps1 -- Build and push the fall-dashboard image to the FOCUS registry.
# Run from _6G_Integration_v2_mqtt/ as working directory.
# mock-app is NOT built -- it is a dev-only simulation, not deployed to k3s.

param(
    [string]$Tag = "latest"
)

$Registry  = "registry-smarko-health.de"
$ImageName = "fall-detection/fall-dashboard"
$FullImage = "${Registry}/${ImageName}:${Tag}"

Write-Host "Building fall-dashboard image: $FullImage"
docker build -t $FullImage -f fall_dashboard/Dockerfile .
if (-not $?) { Write-Error "Build failed"; exit 1 }

Write-Host "Pushing $FullImage ..."
docker push $FullImage
if (-not $?) { Write-Error "Push failed"; exit 1 }

Write-Host ""
Write-Host "Done. Image pushed: $FullImage"
Write-Host "Update values.yaml fallDashboard.image if you used a non-latest tag."
