---
name: FOCUS DevOps meeting decisions (2026-04-29)
description: Production-cluster constraints and architecture decisions confirmed with FOCUS DevOps. Drives values.yaml + chart shape + open work for Mohamed.
type: project
originSessionId: e8f08216-13fd-4c07-b839-eb1955ecf810
---
Meeting with FOCUS DevOps tech partner, ~2026-04-29. These are the answers that change the chart's production values and one architectural requirement from the clinical partner.

**Why:** these came from a synchronous meeting that won't be replayed; they're the only authoritative source for the production-cluster shape we have.

**How to apply:** reach for these whenever the question is about production deployment, k8s flavor, namespaces, registry, or "what's left for Mohamed".

## Production cluster facts

- **Distribution:** k3s (NOT vanilla K8s). Default StorageClass `local-path`, default ingress controller Traefik (already aligns with our chart). No NetworkPolicy enforcement (their cluster is "not production ready").
- **Hardware budget:** 8 CPU cores + 35 GB RAM total; ~20 GB already used by their existing services. **Our limit: ≤ 15 GB RAM.** Current chart sums to ~12 Gi memory limits / ~9.5 cores limits — over CPU budget on paper. Plan to trim mlflow, minio, prometheus, postgres limits.
- **Load balancer:** Traefik configured manually by FOCUS, NOT via our Ingress resource. We must specify endpoint + port + auth header so they configure Traefik routes themselves. Drop / gate `templates/fall-dashboard/ingress.yaml`.
- **One namespace only.** Our 10 services share the same namespace as FOCUS's existing FHIR/MySQL/InfluxDB/Flutter dashboard. `values.yaml namespaces.ours == namespaces.focus`. Drop `templates/namespace.yaml`; install with `--namespace <ns>` (NO `--create-namespace`).
- **Existing FOCUS services in the namespace:** FHIR server, MySQL, InfluxDB, Flutter-based dashboard portal. Patient Dashboard is built by Isa (location TBD; ask Isa).
- **Open questions still to ask:** namespace name, public hostname/FQDN, StorageClass name, available disk for PVCs, exact registry URL spelling, what `mcs-labs` covers, Traefik route handover process, ResourceQuotas.

## Registry

- URL: `registry-smarko-health.de` (verify spelling)
- Image-pull secret already in cluster: `mcs-labs` (from kompose annotation `kompose.image-pull-secret: mcs-labs`)
- Push credentials separate — must be obtained from the partner; `mcs-labs` is read-only/pull-only.
- Chart needs `imagePullSecrets: [name: mcs-labs]` added to all 5 custom-image deployments + migrate-job (templates currently don't reference imagePullSecrets at all).

## FHIR push: opted out

We are NOT pushing fall events to FOCUS's FHIR server. Decision is ours; user opted out for integration simplicity. Set `inferenceServer.fhirServerUrl: ""` (already the default; no code change). `influx_marker_writer.py` already deleted.

## Clinical requirement: 3-state dashboard ★ open work

Clinical partner wants the dashboard to display fall **detection** events too, not only patient-confirmed ones. Three states needed: idle (white) → detected/awaiting (yellow) → confirmed/emergency (red) or dismissed (back to white).

Recommended architecture: inference-server calls fall-dashboard via internal HTTP (`POST /internal/detected`) on every fall=True. This keeps inference-server MQTT-free (architectural decision) while giving fall-dashboard a single source of truth. Fall-dashboard then broadcasts SSE with a `state` field; phone's confirmation MQTT later updates state to confirmed/dismissed via `observation_id` join.

Files to touch when implementing: `inference_server/server.py`, `fall_dashboard/web.py`, `shared_db/migrations/`, `helm/mock-focus/dockerfiles/dashboard.html`, and coordinate with Isa for the real Flutter dashboard. Updates to data contract in `04_mobile_app_integration.md` will follow.

## Handover

Mohamed (MCS) is taking over the K8s integration work from Hayate. Detailed handover: [handover_docs_2/MOHAMED_focus_handover.md](../../../Documents/6G/FallDetection_new/handover_docs_2/MOHAMED_focus_handover.md). Scope is Step 9 (FOCUS-specific values + chart edits) + clinical-requirement architectural change above.
