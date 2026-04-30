# Open all 7 kubectl port-forwards in a SINGLE Windows Terminal tab as split panes.
# Sibling to port-forward.ps1 (which opens 7 separate windows).
# Requires Windows Terminal (wt.exe) -- default on Win11 / Win10 22H2.
# Falls back with an error message if wt.exe is not on PATH.
#
# Layout (4 top, 3 bottom):
#   +--------------+--------------+--------------+--------------+
#   | inference    | fall-dash    | ml-dash      | server-health|
#   | :8001        | :8002        | :8004        | :8006        |
#   +-----------------+----------------+------------------------+
#   | mlflow :5000    | grafana :3000  | mqtt-broker :1883      |
#   +-----------------+----------------+------------------------+
#
# Run from _6G_Integration_v2_mqtt/ as cwd. Prereqs: install.ps1 finished, pods Running.
#
# Tab close   -> drops ALL tunnels at once.
# Ctrl+Shift+W -> drops only the focused pane's tunnel.
# Alt+Arrow    -> move focus between panes.
#
# IMPORTANT for Windows: open URLs at 127.0.0.1:<port>, NOT localhost:<port>.
# port-forward only binds IPv4; Windows resolves localhost to ::1 first and may
# bypass the tunnel. See project_k8s_local_testing.md gotcha #16.

$ErrorActionPreference = "Stop"

if (-not (Get-Command wt.exe -ErrorAction SilentlyContinue)) {
    Write-Error "Windows Terminal (wt.exe) not found on PATH. Install from Microsoft Store, or use port-forward.ps1 (separate windows)."
    exit 1
}

$ns = "mcs-fall-detection"

# Build the pwsh command line that runs inside one pane:
#   - sets the pane title (so the tab shows what each tunnel is)
#   - runs kubectl port-forward
# -NoExit keeps the pane alive when port-forward dies, so you can read errors.
function PfCmd([string]$name, [int]$port) {
    "pwsh -NoExit -Command `"`$Host.UI.RawUI.WindowTitle='pf $name :$port'; kubectl port-forward -n $ns svc/$name ${port}:${port}`""
}

# Build the wt command line. ' ; ' (space-semicolon-space) is wt's command separator.
# Sequence reasoning:
#   1. new-tab with inference-server (becomes only pane = full window)
#   2. split horizontally below it -> mlflow takes bottom half (focus moves to mlflow)
#   3. move-focus up back to inference-server
#   4..6. split vertically 3x -> creates 4 columns in the top row
#         (sizes 0.75, 0.66, 0.5 give roughly 4 equal columns)
#   7. move-focus down to mlflow
#   8..9. split vertically 2x -> creates 3 columns in the bottom row
$wtArgs = (
    "new-tab --title k8s-tunnels $(PfCmd 'inference-server' 8001)",
    "split-pane -H -s 0.5 $(PfCmd 'mlflow' 5000)",
    "move-focus up",
    "split-pane -V -s 0.75 $(PfCmd 'fall-dashboard' 8002)",
    "split-pane -V -s 0.66 $(PfCmd 'ml-dashboard' 8004)",
    "split-pane -V -s 0.5 $(PfCmd 'server-health' 8006)",
    "move-focus down",
    "split-pane -V -s 0.66 $(PfCmd 'grafana' 3000)",
    "split-pane -V -s 0.5 $(PfCmd 'mqtt-broker' 1883)"
) -join ' ; '

Write-Host "==> Opening Windows Terminal with 7 port-forward panes..."
Start-Process wt.exe -ArgumentList $wtArgs

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
Write-Host "Close the tab to end the session; Ctrl+Shift+W to drop one pane; Alt+Arrow to switch focus."
