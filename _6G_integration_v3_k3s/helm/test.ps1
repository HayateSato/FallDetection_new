# test.ps1 -- Smoke-test the caregiver deployment after install.
# Run from _6G_integration_v3_k3s/ as working directory.
# Requires: kubectl, curl.exe

$Namespace = "fall-dashboard"
$Passed    = 0
$Failed    = 0

function Test-Probe {
    param([string]$Name, [scriptblock]$Check)
    try {
        $result = & $Check
        if ($result) {
            Write-Host "  PASS  $Name"
            $script:Passed++
        } else {
            Write-Host "  FAIL  $Name"
            $script:Failed++
        }
    } catch {
        Write-Host "  FAIL  $Name -- $_"
        $script:Failed++
    }
}

Write-Host "=== Caregiver smoke tests (namespace: $Namespace) ==="
Write-Host ""

# 1. Both pods running
Test-Probe "mosquitto pod Running" {
    $status = kubectl get pod -n $Namespace -l app=mosquitto -o jsonpath="{.items[0].status.phase}" 2>$null
    $status -eq "Running"
}

Test-Probe "fall-dashboard pod Running" {
    $status = kubectl get pod -n $Namespace -l app=fall-dashboard -o jsonpath="{.items[0].status.phase}" 2>$null
    $status -eq "Running"
}

# 2. Services exist
Test-Probe "mosquitto Service exists" {
    $svc = kubectl get svc mosquitto -n $Namespace 2>$null
    $svc -ne $null
}

Test-Probe "fall-dashboard Service exists" {
    $svc = kubectl get svc fall-dashboard -n $Namespace 2>$null
    $svc -ne $null
}

# 3. fall-dashboard /api/patients responds (via kubectl port-forward)
Write-Host ""
Write-Host "  Starting port-forward to fall-dashboard:8002 ..."
$pf = Start-Process -PassThru -WindowStyle Hidden -FilePath "kubectl" `
    -ArgumentList "port-forward -n $Namespace svc/fall-dashboard 18002:8002"
Start-Sleep -Seconds 4

Test-Probe "GET /api/patients returns 200" {
    try {
        $resp = curl.exe -s -o NUL -w "%{http_code}" http://localhost:18002/api/patients
        $resp -eq "200"
    } finally {
        Stop-Process -Id $pf.Id -ErrorAction SilentlyContinue
    }
}

# 4. IngressRoutes exist
Test-Probe "fall-dashboard IngressRoute exists" {
    $ir = kubectl get ingressroute fall-dashboard -n $Namespace 2>$null
    $ir -ne $null
}

Test-Probe "mosquitto WebSocket IngressRoute exists" {
    $ir = kubectl get ingressroute mosquitto-ws -n $Namespace 2>$null
    $ir -ne $null
}

Write-Host ""
Write-Host "=== Results: $Passed passed, $Failed failed ==="
if ($Failed -gt 0) { exit 1 }
