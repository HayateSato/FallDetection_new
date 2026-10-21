# Helm Scripts Reference

All scripts live in `_6G_integration_v3_k3s/helm/` and must be run from
`_6G_integration_v3_k3s/` as the working directory.

Run with `bash helm/<script>.sh`

---

## build.sh

**Who runs it:** MCS (Mohammed) — not FOCUS DevOps.

Builds the `fall-dashboard` Docker image and pushes it to the MCS registry
(`registry-smarko-health.de/fall-detection/fall-dashboard:latest`).
FOCUS DevOps only needs to pull the image — this script is how MCS publishes it.

```bash
bash helm/build.sh           # builds and pushes with tag: latest
bash helm/build.sh v1.2.0    # optional: push with a specific tag
```

---

## install.sh

**Who runs it:** FOCUS DevOps — run once to install, and again after any config change.

Runs `helm upgrade --install` with `helm/values_production.yaml`. Creates the
`fall-dashboard` namespace if it does not exist yet. Waits up to 120 seconds for
both pods (mosquitto, fall-dashboard) to reach Running state before returning.

After it completes it prints the pod list, service list, and IngressRoutes so you
can confirm everything came up.

```bash
bash helm/install.sh
```

Re-run this same command for any future update — values change, new image pushed
by Mohammed, etc. `helm upgrade --install` is safe to run multiple times.

---

## test.sh

**Who runs it:** FOCUS DevOps — run immediately after install to verify the deployment.

Runs 7 probes against the live cluster:

| # | Probe |
|---|-------|
| 1 | mosquitto pod is Running |
| 2 | fall-dashboard pod is Running |
| 3 | mosquitto Service exists |
| 4 | fall-dashboard Service exists |
| 5 | `GET /api/patients` returns HTTP 200 (via temporary port-forward) |
| 6 | fall-dashboard IngressRoute exists |
| 7 | mosquitto WebSocket IngressRoute exists |

Exits with code 1 if any probe fails. Expected output: `7 passed, 0 failed`.

```bash
bash helm/test.sh
```

---

## teardown.sh

**Who runs it:** FOCUS DevOps — only when fully removing the deployment.

Prompts for confirmation (`yes`), then runs `helm uninstall` and
`kubectl delete namespace fall-dashboard`.

**Destructive:** deletes all K8s resources in the namespace including the
PVCs — the SQLite patient store and all MQTT broker persistence data are
permanently lost.

```bash
bash helm/teardown.sh
```
