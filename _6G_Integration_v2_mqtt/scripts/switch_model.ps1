# Hot-swap the inference server model.
#
# Usage (from _6G_Integration_v2_mqtt/ as cwd):
#
#   # Load from MLflow registry (recommended — requires model promoted to a stage in UI)
#   .\scripts\switch_model.ps1 -Stage Production
#   .\scripts\switch_model.ps1 -Stage Staging
#
#   # Load from local .pkl file
#   .\scripts\switch_model.ps1 -Version v0_retrained
#   .\scripts\switch_model.ps1 -Version v0
#
#   # Point at a different server (default: localhost:8001)
#   .\scripts\switch_model.ps1 -Stage Production -ServerUrl http://192.168.1.50:8001

param(
    [string]$Stage      = "",
    [string]$Version    = "",
    [string]$ServerUrl  = "http://localhost:8001",
    [string]$ApiKey     = "626e6c481b78c77d52db774d0e54a06cefb0553cb245fa15d5c7050fc7424e7a"
)

if ($Stage -and $Version) {
    Write-Error "Provide -Stage OR -Version, not both."
    exit 1
}
if (-not $Stage -and -not $Version) {
    Write-Error "Provide -Stage (e.g. Production) or -Version (e.g. v0_retrained)."
    exit 1
}

$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key"    = $ApiKey
}

if ($Stage) {
    $body = @{ mlflow_stage = $Stage } | ConvertTo-Json
    Write-Host "Switching to MLflow stage: $Stage" -ForegroundColor Cyan
} else {
    $body = @{ version = $Version } | ConvertTo-Json
    Write-Host "Switching to file-based version: $Version" -ForegroundColor Cyan
}

try {
    $response = Invoke-RestMethod -Method POST -Uri "$ServerUrl/model/switch" `
        -Headers $headers -Body $body
    Write-Host "Success:" -ForegroundColor Green
    $response | ConvertTo-Json
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ErrorDetails.Message
}

# Verify
Write-Host "`nCurrent model info:" -ForegroundColor Green
Invoke-RestMethod -Uri "$ServerUrl/model/info" | ConvertTo-Json
