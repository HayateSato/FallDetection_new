# Build the two custom images for the mock-focus chart.
# Run from _6G_Integration_v2_mqtt/ as the working directory.
#
# Local Docker Desktop K8s shares the docker daemon's image cache,
# so `imagePullPolicy: IfNotPresent` will find these images without a registry.

Write-Host "==> Building mock-fhir image..."
docker build `
  -f helm/mock-focus/dockerfiles/mock-fhir.Dockerfile `
  -t fd-mock-fhir:latest `
  .
if ($LASTEXITCODE -ne 0) { Write-Error "mock-fhir build failed"; exit 1 }

Write-Host "==> Building mock-patient-dashboard image..."
docker build `
  -f helm/mock-focus/dockerfiles/mock-patient-dashboard.Dockerfile `
  -t fd-mock-patient-dashboard:latest `
  .
if ($LASTEXITCODE -ne 0) { Write-Error "mock-patient-dashboard build failed"; exit 1 }

Write-Host ""
Write-Host "Both images built. Verify:"
Write-Host "  docker images | findstr fd-mock"
Write-Host ""
Write-Host "Next: helm install mock-focus ./helm/mock-focus --create-namespace"
