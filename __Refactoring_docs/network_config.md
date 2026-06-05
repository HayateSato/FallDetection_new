## Check existing Windows Firewall rules

### List all enabled inbound Allow rules (with port details)

```powershell
# Summary list — names only
Get-NetFirewallRule -Direction Inbound -Action Allow | Where-Object { $_.Enabled -eq "True" } | Select-Object DisplayName | Sort-Object DisplayName

# Full detail — name + port + protocol + profile
Get-NetFirewallRule -Direction Inbound -Action Allow | Where-Object { $_.Enabled -eq "True" } | ForEach-Object {
    $portFilter = $_ | Get-NetFirewallPortFilter
    [PSCustomObject]@{
        Name      = $_.DisplayName
        Profile   = $_.Profile
        Protocol  = $portFilter.Protocol
        LocalPort = $portFilter.LocalPort
        Enabled   = $_.Enabled
    }
} | Sort-Object Name | Format-Table -AutoSize
```

### Filter to Fall Detection rules only

```powershell
Get-NetFirewallRule -Direction Inbound -Action Allow | Where-Object { $_.DisplayName -match "Fall Detection|MQTT 1883" -and $_.Enabled -eq "True" } | ForEach-Object {
    $portFilter = $_ | Get-NetFirewallPortFilter
    [PSCustomObject]@{
        Name      = $_.DisplayName
        Profile   = $_.Profile
        Protocol  = $portFilter.Protocol
        LocalPort = $portFilter.LocalPort
    }
} | Format-Table -AutoSize
```

### Disable (not delete) an existing rule

Disabling keeps the rule for later re-enabling — safer than deleting.

```powershell
# Disable a single rule by exact display name
Set-NetFirewallRule -DisplayName "Fall Detection - Inference Server" -Enabled False

# Disable all Fall Detection rules at once
Get-NetFirewallRule | Where-Object { $_.DisplayName -match "Fall Detection" } | Set-NetFirewallRule -Enabled False

# Re-enable them again
Get-NetFirewallRule | Where-Object { $_.DisplayName -match "Fall Detection" } | Set-NetFirewallRule -Enabled True
```

### Delete a rule permanently

```powershell
# Delete a single rule by exact display name
Remove-NetFirewallRule -DisplayName "Fall Detection - Inference Server"

# Delete all Fall Detection rules at once
Get-NetFirewallRule | Where-Object { $_.DisplayName -match "Fall Detection" } | Remove-NetFirewallRule
```

> Run PowerShell as Administrator for Set-NetFirewallRule and Remove-NetFirewallRule.

### Current rules in place (as of 2026-06-05)

| Rule | Port | Profile | Notes |
|---|---|---|---|
| Fall Detection - Inference Server | 8001 | Any | Laptop 1 (MCS) |
| Fall Detection - MQTT | 1883 | Any | Laptop 2 (FOCUS) |
| Fall Detection - Fall Dashboard | 8002 | Any | Laptop 2 (FOCUS) |
| Fall Detection - Mock App | 8005 | Any | Laptop 2 (FOCUS) |
| MQTT 1883 | 1883 | Any | Older duplicate — safe to leave |

---

## Why the mobile app can't reach your laptops

### 1. Windows Firewall (most likely — fix this first)

Docker publishes ports to `0.0.0.0` (all interfaces) but **Windows Firewall still controls inbound access from other devices**. Docker Desktop usually adds a broad "allow" rule for itself, but it doesn't add per-port rules for the host-side bindings.

Run this on **both laptops** in PowerShell (as Administrator) to open the relevant ports:

`# Laptop 1 (MCS) — inference server
New-NetFirewallRule -DisplayName "Fall Detection - Inference Server" -Direction Inbound -Protocol TCP -LocalPort 8001 -Action Allow

# Laptop 2 (FOCUS) — MQTT + fall dashboard + mock app
New-NetFirewallRule -DisplayName "Fall Detection - MQTT" -Direction Inbound -Protocol TCP -LocalPort 1883 -Action Allow
New-NetFirewallRule -DisplayName "Fall Detection - Fall Dashboard" -Direction Inbound -Protocol TCP -LocalPort 8002 -Action Allow
New-NetFirewallRule -DisplayName "Fall Detection - Mock App" -Direction Inbound -Protocol TCP -LocalPort 8005 -Action Allow`

### 2. WiFi network profile set to "Public"

If the WiFi connection is classified as **Public** in Windows, the firewall blocks almost all inbound LAN traffic regardless of rules. Check:

- Settings → Network & Internet → WiFi → your network → **set to "Private"**

### 3. Docker Desktop + WSL2 port forwarding gap (Windows-specific)

Docker Desktop on Windows with the WSL2 backend routes traffic through a virtual adapter. Ports bound to `0.0.0.0` on the container are forwarded to Windows `localhost`, but occasionally the physical LAN adapter is not in that chain. You can verify with:

`# On Laptop 1 — from another machine, can you reach it?
# First check what IP the laptop has:
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq "Dhcp" } | Select-Object IPAddress, InterfaceAlias`

Then from the **mobile app or another machine**, try `curl http://<laptop1-ip>:8001/health`. If that fails even after opening the firewall, you need a port proxy:

`# On Laptop 1 — proxy WSL2 port to the physical interface
# (replace 172.x.x.x with your WSL2 IP from: wsl hostname -I)
netsh interface portproxy add v4tov4 listenport=8001 listenaddress=0.0.0.0 connectport=8001 connectaddress=<WSL2-IP>`

### 4. AP/client isolation on the WiFi router

Some routers (especially hotspot/guest networks) enable **AP isolation** which prevents devices on the same WiFi from reaching each other. The phone and laptop are both on WiFi → this would silently block all communication. Check your router settings and disable "AP isolation" / "client isolation".

---

## Quick diagnostic sequence

`# 1. Get Laptop 1's LAN IP
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq "Dhcp" }

# 2. From Laptop 2 (or any device on the same WiFi), test Laptop 1:
Invoke-WebRequest -Uri "http://<Laptop1-IP>:8001/health" -TimeoutSec 5

# 3. From Laptop 2, test MQTT port is reachable:
Test-NetConnection -ComputerName <Laptop2-IP> -Port 1883`

If step 2/3 fail even from **another laptop on the same WiFi**, the firewall or AP isolation is the blocker — not the mobile app. Fix those first, then the mobile app will work the same way.

---

**Summary:** Start with firewall rules (step 1) and network profile (step 2) — those fix 90% of cases. The WSL2 port proxy is only needed if ports work from `localhost` but not from a LAN IP.