# Open all kubectl port-forwards in separate PowerShell windows so each one
# stays alive independently. Each window can be Ctrl+C'd individually to stop
# just that tunnel; close them all to end the session.
#
# Run from _6G_Integration_v2_mqtt/ as cwd. Prereqs: install.ps1 has finished
# and pods are Running.
#
# IMPORTANT for Windows: open URLs at 127.0.0.1:<port>, NOT localhost:<port>.
# port-forward only binds IPv4; Windows resolves localhost to ::1 (IPv6) first
# and may bypass the port-forward (or hit a stray Compose container). See
# project_k8s_local_testing.md gotcha #16.

$ErrorActionPreference = "Continue"

$tunnels = @(
    @{ name = "inference-server"; port = 8001 },
    @{ name = "fall-dashboard";   port = 8002 },
    @{ name = "ml-dashboard";     port = 8004 },
    @{ name = "server-health";    port = 8006 },
    @{ name = "mlflow";           port = 5000 },
    @{ name = "grafana";          port = 3000 },
    @{ name = "mqtt-broker";      port = 1883 }
)

Write-Host "==> Opening 7 port-forward windows..."
foreach ($t in $tunnels) {
    $title = "pf $($t.name) :$($t.port)"
    $cmd   = "`$Host.UI.RawUI.WindowTitle = '$title'; kubectl port-forward -n mcs-fall-detection svc/$($t.name) $($t.port):$($t.port)"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd | Out-Null
}

Write-Host ""
Write-Host "URLs (use 127.0.0.1, NOT localhost - Windows IPv4/IPv6 gotcha):"
Write-Host "  http://127.0.0.1:8002/   fall-dashboard (caregiver view)"
Write-Host "  http://127.0.0.1:8004/   ml-dashboard (admin: retrain + hot-swap)"
Write-Host "  http://127.0.0.1:8006/   server-health (admin: traffic-light status)"
Write-Host "  http://127.0.0.1:5000/   MLflow"
Write-Host "  http://127.0.0.1:3000/   Grafana (admin/admin)"
Write-Host ""
Write-Host "Inference + MQTT (no UI):"
Write-Host "  http://127.0.0.1:8001/health    inference-server health"
Write-Host "  tcp://127.0.0.1:1883            MQTT broker (use a client like mosquitto_sub)"
Write-Host ""
Write-Host "Close any window to drop just that tunnel; close all to end the session."
