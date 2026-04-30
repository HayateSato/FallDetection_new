# Mohamed — FOCUS K8s Integration Handover

**From:** Hayate (MCS)
**To:** Mohamed (MCS) — taking over the K8s integration work
**Date:** 2026-04-29
**Goal:** finish the K8s deployment so the fall-detection system runs inside the FOCUS production cluster.

You are not starting from scratch. The chart works end-to-end on Docker Desktop K8s (smoke test passed 2026-04-29) and is ready in principle for FOCUS. **What changed** between "ready in principle" and "ready for their cluster" came out of a meeting I had with FOCUS DevOps — that's what this doc is about.

Read in order:

1. Section 1 — what FOCUS told us at the meeting (the new constraints)
2. Section 2 — registry & image-pull secret (this is the most concrete unblock)
3. Section 3 — chart changes you need to make
4. Section 4 — open architectural work (the "fall detected" requirement is the biggest)
5. Section 5 — what to ask FOCUS next
6. Sections 6–8 — references, file map, contacts

---

## 1. New constraints from FOCUS DevOps (the meeting)

### 1.1 Hardware budget

FOCUS production is a Linux box with **8 CPU cores, ~35 GB RAM total**. Their existing services already use ~20 GB. **Our budget: ≤ 15 GB RAM and a fraction of 8 CPUs.** We can't fully reserve 8 cores — they're shared with FOCUS workloads.

Where we are today (sum of `values.yaml → resources.*.limits` for our chart):

| Resource | Sum of limits | Headroom |
|----------|---------------|----------|
| Memory   | ~12 GiB       | Tight but OK (3 GiB to spare) |
| CPU      | ~9.5 cores    | **Over budget on paper** — limits, not requests, but worth trimming |

Limits are ceilings, not reservations — actual use is usually lower. But if FOCUS enforces ResourceQuotas on the namespace, the chart will fail to install on the sum of limits. Plan to trim. **Most aggressive savings (without touching the data path):**

- **MLflow 2 Gi → 1.5 Gi** (was 1 Gi → OOM-killed; we can probably land at 1.5 Gi by reducing gunicorn workers via env var)
- **MinIO 2 Gi → 1 Gi** (artifact store; idle most of the time)
- **Prometheus 2 Gi → 1 Gi** (scrapes 2 endpoints — doesn't need 2 GiB)
- **Postgres 2 Gi → 1 Gi** (small dataset; fine)
- **Grafana already small at 0.5 Gi**

That gets us to ~8 Gi memory, ~7 cores limits. Verify under load before declaring victory.

### 1.2 Load balancer: Traefik (NOT a vanilla Ingress controller)

FOCUS uses Traefik as their load balancer. **Good news:** our chart already specifies `ingressClassName: traefik`. **Bad news:** we previously assumed FOCUS would route based on our Ingress resource. They've told us they configure Traefik **manually** and need us to specify the endpoint so they can add the route.

#### What you need to give them

| Item | Value (from current chart) |
|------|----------------------------|
| Service name | `inference-server` |
| Service port | `8001` |
| Endpoint | `POST /predict` |
| Auth header | `X-API-Key: <key>` (we generate the key, share separately) |
| Other endpoints they may want to route | `GET /health` (liveness), `GET /metrics` (Prometheus, internal only — DON'T expose) |

Also document the **fall-dashboard** endpoint (port 8002) — Isa's mobile app and the Patient Dashboard both consume from it:

| Item | Value |
|------|-------|
| Service name | `fall-dashboard` |
| Service port | `8002` |
| Endpoints | `GET /api/patients`, `GET /api/falls`, `GET /api/stream` (SSE — needs Traefik streaming config) |

> SSE on Traefik works without special config (unlike nginx) — but verify with FOCUS that they haven't set a low `responseTimeout` middleware globally. Our SSE stream stays open indefinitely.

### 1.3 No NetworkPolicy

Their cluster does not enforce NetworkPolicy ("not production ready" — their words). This **simplifies our work** — we don't need cross-namespace allow rules. But it also means the cluster has no isolation; anything in their cluster can talk to our pods. Not our problem to fix, but worth noting.

The `templates/fall-dashboard/networkpolicy.yaml` (if/when it gets added) is a no-op on their cluster and can be skipped for now.

### 1.4 Single namespace (not two)

FOCUS wants **everything in one namespace**, not two. Currently the chart assumes:

- `mcs-fall-detection` — our 10 services
- `focus-ns` — their FHIR + InfluxDB + Patient Dashboard

**New plan:** install our 10 services into FOCUS's existing namespace (whatever they call it). FOCUS's existing services (FHIR, MySQL, InfluxDB, Flutter dashboard) are already there. We add ours alongside.

#### What this changes in the chart

```yaml
# values.yaml — both fields collapse to the same value
namespaces:
  ours:  <focus-namespace-name>     # ← TBD; ask FOCUS
  focus: <focus-namespace-name>     # ← same value
```

Service DNS becomes simpler — no more `.<ns>.svc.cluster.local` cross-namespace lookups. The `templates/namespace.yaml` we ship needs to **NOT** create a new namespace if FOCUS's already exists. Either:

- Drop `templates/namespace.yaml` entirely and rely on `helm install --namespace <ns>` (no `--create-namespace`)
- OR add an `if .Values.createNamespace` flag in the template, default false for FOCUS

I'd go with the first — simpler.

### 1.5 They run k3s (not vanilla K8s)

[k3s](https://k3s.io) is a lightweight K8s distribution from Rancher. For our purposes, **the differences are mostly invisible**:

| Concern | Vanilla K8s | k3s | Impact on us |
|---------|-------------|-----|--------------|
| API surface | full | full (drops a few alpha APIs) | none — we use stable APIs |
| Container runtime | varies | containerd (built-in) | none — same image format |
| Default StorageClass | none | `local-path` (provisioned automatically) | **Good** — Postgres and MinIO PVCs will bind without us specifying a class |
| Default Ingress | none | Traefik (built-in) | aligns with what they already use |
| Default LoadBalancer | none | klipper-lb (servicelb) | not relevant — they have their own setup |
| etcd | external | embedded (or sqlite for single-node) | none |
| Networking | varies (Calico/Cilium) | flannel (built-in) | none for us; means NetworkPolicy not enforced unless they swap CNI (they haven't) |

**Practical takeaway:** our chart should run on k3s without changes once the namespace + registry adjustments are in. Set `postgres.storageClass: ""` and `minio.storageClass: ""` (empty string) — k3s's `local-path` is the cluster default and will be auto-selected.

> Beware: k3s `local-path` provisioner uses the host filesystem. If FOCUS has limited disk, our 30 Gi (10 Postgres + 20 MinIO) may not fit. **Action: ask FOCUS what disk is available.**

### 1.6 FHIR: not pushing

I asked: do we push fall events to their FHIR server? They said it's our call. **Decision: we are NOT pushing to FHIR.** Reasons:
- One less integration surface to debug
- Their FHIR server is their existing resource — touching it adds review friction
- Caregiver-facing path doesn't need it (we have our own fall_history in Postgres)

**What this means for the chart:** keep `inferenceServer.fhirServerUrl: ""` in values.yaml. The inference-server already treats empty string as "skip FHIR" — no code change.

### 1.7 Patient Dashboard: built by Isa, location unknown

The production Patient Dashboard is a Flutter app built by Isa. I do not have access to its repo / artifacts. The clinical partner wants the dashboard to display falls per patient — this is an Isa task, not yours, but you'll need to:

1. **Coordinate with Isa** to find out where the Flutter dashboard lives, what its build/deploy story is, and whether it can be containerised into our chart or whether FOCUS DevOps deploys it separately.
2. **Confirm the data source** Isa expects — does his dashboard call our `fall-dashboard /api/falls` and `/api/stream` directly? (If yes, just give him the URL once Traefik routes it.)
3. **Decide what we ship in the chart** in the meantime: I'd say **keep our `mock_patient_dashboard`** in the chart for now (renamed: drop "mock" if it's becoming the real thing) so something visualises falls until Isa's app is wired in.

### 1.8 Clinical request: show "fall detected" too, not just "fall confirmed"

> Currently the dashboard only flags a patient when the patient has confirmed the fall via the popup (or the 10s timeout passed without "no"). The clinical team wants to see **detected** falls too — so caregivers are alerted to pay attention even before the patient confirms.

This is the biggest architectural change in this handover. See section 4.1.

---

## 2. Registry & image-pull secret (the unblock)

### 2.1 What FOCUS told us

| Item | Value |
|------|-------|
| Container registry URL | `registry-smarko-health.de` (verify exact spelling with the partner — sounds right but I'm not 100% sure) |
| Image-pull secret name in their cluster | `mcs-labs` (already exists in their namespace; **don't try to create it**) |

The hint they gave us was a kompose annotation:

```yaml
kompose.image-pull-secret: mcs-labs
```

That's a [kompose](https://kompose.io/) annotation (kompose converts Docker Compose to K8s manifests). It maps to a K8s `imagePullSecrets:` field referencing a secret named `mcs-labs`. **We don't use kompose** — we use Helm — so what you actually do is patch our Helm templates to add `imagePullSecrets`. See section 3.2.

### 2.2 Two distinct credentials

Don't confuse these:

| Purpose | Where it's used | Who has it |
|---------|------------------|-----------|
| **Push** (our laptop / CI → registry) | `docker login registry-smarko-health.de` | You + me — ask SmarKo or the partner for these credentials |
| **Pull** (FOCUS cluster → registry) | The `mcs-labs` K8s secret | Already in FOCUS cluster — we just reference it by name |

**Action items for you:**

1. Get push credentials from the partner (or whoever owns `registry-smarko-health.de`). One person on our side needs them for the laptop or CI build pipeline.
2. Confirm with the partner that `mcs-labs` covers all the images we'll push (they may have configured it for a specific repo path).
3. Push the 5 custom images (or 4 if we drop the custom mlflow):
   ```
   registry-smarko-health.de/<path>/inference-server:<tag>
   registry-smarko-health.de/<path>/fall-dashboard:<tag>
   registry-smarko-health.de/<path>/ml-dashboard:<tag>
   registry-smarko-health.de/<path>/server-health:<tag>
   registry-smarko-health.de/<path>/mlflow:<tag>
   ```

The detailed push workflow (PowerShell snippet, docker login, retag commands) is already documented in [`_6G_Integration_v2_mqtt/helm/fall-detection/REGISTRY_SETUP.md`](../_6G_Integration_v2_mqtt/helm/fall-detection/REGISTRY_SETUP.md). It was originally written for GitLab Container Registry but the steps generalise — substitute the registry URL.

---

## 3. Chart changes you need to make

### 3.1 `values.yaml` — set FOCUS-specific values

```yaml
# helm/fall-detection/values.yaml

namespaces:
  ours:  <focus-namespace-name>     # CHANGED: same as focus
  focus: <focus-namespace-name>     # ASK FOCUS for the name

registry: registry-smarko-health.de/<path>     # CHANGED

images:
  pullPolicy: Always               # CHANGED from Never
  inferenceServer: { repository: inference-server, tag: <git-sha-or-version> }
  fallDashboard:   { repository: fall-dashboard,   tag: <git-sha-or-version> }
  mlDashboard:     { repository: ml-dashboard,     tag: <git-sha-or-version> }
  serverHealth:    { repository: server-health,    tag: <git-sha-or-version> }
  mlflow:          { repository: mlflow,           tag: <git-sha-or-version> }   # if custom mlflow image

imagePullSecrets:                  # NEW field
  - name: mcs-labs

inferenceServer:
  fhirServerUrl: ""                # CONFIRMED empty — no FHIR push
  apiKeys: <real-api-key>          # generate a strong key; share via secure channel

postgres:
  storageClass: ""                 # k3s will use 'local-path' default
  storageSize: 5Gi                 # CHANGED from 10Gi if disk is tight (ask FOCUS)

minio:
  storageClass: ""
  storageSize: 10Gi                # CHANGED from 20Gi if disk is tight

ingress:
  enabled: false                   # CHANGED — FOCUS configures Traefik manually, we don't ship an Ingress

resources:
  inferenceServer:
    limits: { cpu: "1.5", memory: 1.5Gi }   # CHANGED from 2/2Gi (still has headroom for inference)
  postgres:
    limits: { cpu: "0.5", memory: 1Gi }     # CHANGED from 1/2Gi
  mlflow:
    limits: { cpu: "0.5", memory: 1.5Gi }   # CHANGED from 1/2Gi
  minio:
    limits: { cpu: "0.5", memory: 1Gi }     # CHANGED from 1/2Gi
  prometheus:
    limits: { cpu: "0.5", memory: 1Gi }     # CHANGED from 1/2Gi
```

After all of those changes the resource sum is roughly **8 Gi RAM / 6 CPU limits** — comfortably inside the 15 Gi / 8 CPU budget.

### 3.2 Add `imagePullSecrets` to all Deployment + Job templates

The chart's templates currently don't reference `imagePullSecrets`. Add this snippet inside `spec.template.spec` of every Deployment template (and the migrate-job), at the same indent level as `containers:`:

```yaml
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 8 }}
{{- end }}
```

Files to touch:

```
templates/inference-server/deployment.yaml
templates/fall-dashboard/deployment.yaml
templates/ml-dashboard/deployment.yaml
templates/server-health/deployment.yaml
templates/mlflow/deployment.yaml
templates/migrate-job.yaml
```

Stock-image deployments (`postgres`, `mqtt-broker`, `minio`, `prometheus`, `grafana`) don't need this — they pull from public Docker Hub.

The `with` guard means the field is omitted when `imagePullSecrets` is empty — local Docker Desktop testing keeps working.

### 3.3 Drop `templates/namespace.yaml`

FOCUS's namespace already exists. If we ship `templates/namespace.yaml` AND someone runs `helm install --namespace <existing-ns>` we either (a) error because the namespace template tries to create something that exists, or (b) end up with weird ownership annotations. Cleanest is to delete the file and let `helm install --namespace <ns>` (no `--create-namespace`) handle placement.

### 3.4 Drop `templates/fall-dashboard/ingress.yaml` (or gate it)

FOCUS configures Traefik manually. They don't want us shipping an Ingress resource. Either:

- Delete the file
- Or wrap it: `{{- if .Values.ingress.enabled }}...{{- end }}` and set `ingress.enabled: false` in values.yaml

### 3.5 Verify helm install on Docker Desktop after changes

Same scripts as before. Don't ship changes to FOCUS that haven't been smoke-tested locally:

```powershell
.\helm\fall-detection\build.ps1
.\helm\fall-detection\install.ps1
.\helm\fall-detection\test.ps1
```

The local install uses `pullPolicy: Never` (via overlay) so `imagePullSecrets` is a no-op — your changes for FOCUS won't break local dev.

---

## 4. Open architectural work

### 4.1 Display "fall detected" (not just "fall confirmed") on the dashboard ★ biggest change

Today's flow:

```
phone → POST /predict → inference-server returns fall=True
phone shows confirmation popup (10s)
phone → MQTT PUBLISH fall/alert/<pid> { patient_confirmed, needs_help, observation_id }
  ↓
fall-dashboard subscribes, writes fall_history, broadcasts SSE
  ↓
Patient Dashboard renders the alert
```

The dashboard never sees the "detected, awaiting confirmation" interim state. Clinical wants three states:

| State | When | Card colour (suggestion) |
|-------|------|--------------------------|
| Idle | no recent fall | white |
| Detected — awaiting | inference-server returned fall=True; phone has not yet confirmed | yellow / amber |
| Confirmed — emergency | patient confirmed via popup OR 10s timeout passed | red |
| Detected — dismissed | patient said "no fall" | back to white (or grey for ~5 min) |

#### Architectural options

**Option A — inference-server publishes a "detected" event itself.**
On every fall=True response, it also publishes `fall/detected/<pid>` over MQTT (or directly invokes fall-dashboard via in-cluster HTTP). Phone is unaware of this; it still sends the confirmation later.
- ✓ Phone-app unchanged
- ✓ Single source of truth for detection
- ✗ Requires inference-server to gain an MQTT client (currently HTTP-only — see [project_6g_mqtt.md](../memory_links_only_TBD)). The architecture decision was deliberately to keep inference-server MQTT-free.
- Workaround: instead of MQTT, inference-server calls fall-dashboard via internal HTTP (`POST /internal/detected`). Both pods are in-namespace, simple call.

**Option B — phone publishes both events.**
Phone sends `fall/detected/<pid>` immediately on receiving fall=True from /predict, then publishes `fall/alert/<pid>` after the popup. Two MQTT publishes per fall.
- ✓ inference-server stays HTTP-only (architectural cleanliness)
- ✓ One subscriber in fall-dashboard handles both
- ✗ Phone has to publish twice, network reliability matters more
- ✗ If phone crashes between steps, dashboard sees "detected" forever — needs a TTL on the detected state

**Option C — fall-dashboard polls or tails inference-server's log.**
- ✗ Brittle; don't do this.

**My recommendation: Option A with internal HTTP.** Inference-server `POST /internal/detected` to fall-dashboard with `{patient_id, observation_id, timestamp}`. Fall-dashboard:
- Stores it in a new `detection_event` table
- Broadcasts SSE event with `state: detected`
- When the matching `fall/alert/<pid>` arrives later (with same `observation_id`), updates state to `confirmed` or `dismissed` and broadcasts again

That keeps the architecture clean (still no MQTT in inference-server), gives a clear single source of truth, and lets us add a "stale detected → auto-dismiss" sweeper later if we need to.

**Files to edit when you implement this:**
- [`inference_server/server.py`](../_6G_Integration_v2_mqtt/inference_server/server.py) — add internal HTTP call after fall=True
- [`fall_dashboard/web.py`](../_6G_Integration_v2_mqtt/fall_dashboard/web.py) — add `POST /internal/detected` endpoint, SSE state field, new DB table
- `shared_db/migrations/` — Alembic migration for the new table
- [`helm/mock-focus/dockerfiles/dashboard.html`](../_6G_Integration_v2_mqtt/helm/mock-focus/dockerfiles/dashboard.html) — render three states
- Coordinate with **Isa** for the real Flutter dashboard (section 1.7)
- Update [`04_mobile_app_integration.md`](04_mobile_app_integration.md) data contract — phone behaviour doesn't change but the dashboard SSE schema does

### 4.2 Decide what we do with existing FOCUS InfluxDB

FOCUS has an InfluxDB in their existing system. The mock-focus chart spins up a fake one to validate that "we don't read from it in production." Our inference path doesn't read InfluxDB. **But** the original integration plan included writing fall *markers* to InfluxDB so the FOCUS dashboard can overlay them on biosignal time series.

In my notes I see the marker writer was **deleted** (see project memory: "`influx_marker_writer.py` deleted (colleague handles InfluxDB)"). So we're punting that to the partner. Confirm with them — if they want fall markers in InfluxDB, they write that integration on their side reading our `fall_history` table or SSE stream.

### 4.3 mock_patient_dashboard: keep or drop?

I'd keep it (renamed to drop "mock") and ship it as our visual fallback until Isa's Flutter dashboard is integrated. The clinical 3-state requirement (4.1) needs to be implemented somewhere we control as the reference UI; doing it in `dashboard.html` first means Isa has a concrete spec to mirror.

If FOCUS objects to the extra service, gate it behind `mockPatientDashboard.enabled` in values.yaml (default `true` until Isa's app is live, then flip to `false`).

---

## 5. Things to ask FOCUS DevOps next

I missed some questions in the meeting. You should chase these:

| # | Question | Why it matters | Where the answer lands |
|---|----------|----------------|------------------------|
| 1 | Exact name of the namespace we share with FOCUS services | Required for `values.yaml → namespaces.ours` and `helm install --namespace` | values.yaml |
| 2 | Public hostname / FQDN for the system | Phones need a stable URL for `/predict` | Goes to Isa as well |
| 3 | StorageClass name (or confirm `local-path` default works) | k3s's default is `local-path`, but if they have an explicit storage tier (NVMe vs network), we should use it | `postgres.storageClass`, `minio.storageClass` |
| 4 | Available disk for our PVCs | Our defaults are 10 Gi (Postgres) + 20 Gi (MinIO) = 30 Gi | values.yaml `*.storageSize` |
| 5 | Confirm registry URL spelling (`registry-smarko-health.de`) and the path under it where we should push our 5 images | Push commands won't work otherwise | REGISTRY_SETUP.md |
| 6 | Confirm `mcs-labs` covers our image paths (it might be repo-scoped) | If not, FOCUS needs to update the secret OR create a new one | values.yaml `imagePullSecrets` |
| 7 | Confirm Traefik route mapping process — do they need YAML from us, or do they edit a Traefik config file directly? | Traefik can be configured via IngressRoute CRDs OR a static config file; their process matters for handoff | Traefik route doc |
| 8 | Confirm there are no global Traefik middlewares that break SSE (timeout, buffering) | Our `/api/stream` must stay open for hours | IT review |
| 9 | Their existing FOCUS namespace — any ResourceQuota that would limit us? | Drives our resource limit budget | values.yaml `resources.*` |
| 10 | Time-zone of the cluster — affects timestamps in fall_history | We use UTC server-side; if they expect local time, we may need to convert in UI | Doc-only |

---

## 6. What's done vs. what's left (todo.md status)

The full project todo lives at [`REFACTOR_DOCUS/todo.md`](../REFACTOR_DOCUS/todo.md). Status as of 2026-04-29:

| Step | Owner | Status | Notes |
|------|-------|--------|-------|
| 1–7 | Hayate | ✓ done | Inference server, mobile contract, alerting, metrics, Postgres, Grafana, hot-swap |
| 8 | Isa | partial | Two-role dashboard — Isa's Flutter dashboard outstanding (1.7) |
| 9 | Hayate + FOCUS DevOps | mostly done | Helm chart works locally; **FOCUS-specific values pending = your work** |
| 10 | Hayate + Isa | open | End-to-end integration test against FOCUS infra |
| 11 | Hayate | pipeline ✓, data pending | MLflow retraining works; needs Charite data |
| 11.5 | Hayate | partial | ml_dashboard + server_health UI — admin features deferred |
| 12, 12.5 | Hayate | ✓ done | Dockerfiles + local two-namespace dry run both passed |
| 13 | Hayate → Isa | doc-only | Isa integration handover — see [04_mobile_app_integration.md](04_mobile_app_integration.md) |
| 14 | Hayate + FOCUS DevOps | open | Pre-production checklist |

**Your immediate scope is Step 9 + the new architectural work (section 4.1).** Step 14 (pre-prod checklist) follows once 9 is finished.

---

## 7. File / repo orientation

Key paths under `_6G_Integration_v2_mqtt/`:

```
_6G_Integration_v2_mqtt/
├── inference_server/                       FastAPI :8001 — POST /predict (HTTP only)
├── fall_dashboard/                         FastAPI :8002 — MQTT subscriber + SSE fan-out
├── ml_dashboard/                           Admin UI :8004 — retrain + hot-swap
├── server_health/                          Admin UI :8006 — service status
├── ml_pipeline/                            Shared ML code (in inference + fall-dashboard images)
├── shared_db/                              Shared SQLAlchemy models + Alembic migrations
├── retrain/                                MLflow retraining pipeline (in mlflow image)
├── model/                                  XGBoost .pkl model files (in inference-server image)
├── infrastructure/
│   ├── grafana/                            Provisioning (datasources, dashboards)
│   ├── mosquitto/                          MQTT broker config
│   ├── postgres/                           DB init SQL
│   └── mlflow/                             Custom MLflow Dockerfile
├── helm/
│   ├── fall-detection/                     ★ The chart you ship to FOCUS — most of your edits go here
│   │   ├── values.yaml                     ★ THE single file to edit per environment
│   │   ├── REGISTRY_SETUP.md               How to push images + share the pull credential
│   │   └── templates/                      All deployment YAML
│   └── mock-focus/                         Local dry-run only — does NOT ship to FOCUS
└── local_dev/
    ├── mock_app/                           Reference impl of mobile app (Isa replicates this in RN)
    └── mock_focus/                         Local stand-ins for FOCUS services
```

Already-written docs in `handover_docs_2/`:

| Doc | Audience | Read when |
|-----|----------|-----------|
| [`01_k8s.md`](01_k8s.md) | FOCUS DevOps | They ask for the production deploy guide |
| [`02_fall_detection_algorithm.md`](02_fall_detection_algorithm.md) | anyone | You want to understand what the model does |
| [`03_fall_detection_system.md`](03_fall_detection_system.md) | anyone | Architecture overview + sequence diagram |
| [`04_mobile_app_integration.md`](04_mobile_app_integration.md) | Isa | Mobile-app data contract |
| [`05_web_app_integration.md`](05_web_app_integration.md) | Isa | Patient dashboard data contract |
| [`06–08_user_flow_*.md`](06_user_flow_patient.md) | UX / clinical | Per-actor user flows |
| [`ISA_local_setup_quickstart.md`](ISA_local_setup_quickstart.md) | Isa | Local K8s setup (recently added) |
| `Q&A.md` | anyone | One-off questions answered |

Also in repo:

- `REFACTOR_DOCUS/deployment_architecture.md` — full data-source breakdown across the two-namespace model (still mostly accurate, but section 1.4 of this doc supersedes the "two namespaces" assumption)
- `REFACTOR_DOCUS/helm_guide.md` — annotated walkthrough of every Helm template
- `REFACTOR_DOCUS/mqtt_architecture.md` — why we chose MQTT and the topic/payload conventions
- `_6G_Integration_v2_mqtt/helm/fall-detection/REGISTRY_SETUP.md` — image push + secret handover (originally GitLab-flavoured; substitute SmarKo registry URL)

---

## 8. Contacts

| For | Reach out to |
|-----|--------------|
| Anything in this codebase, model behaviour, why a decision was made | Hayate (until handover complete) |
| Patient dashboard (Flutter app), mobile app on RN | Isa (SmarKo) |
| FOCUS cluster credentials, namespace name, hostname, StorageClass, Traefik routing | FOCUS DevOps tech partner |
| Registry credentials for `registry-smarko-health.de` | Partner who provided `mcs-labs` annotation |
| Clinical requirement clarifications (the 3-state dashboard) | Clinical team via FOCUS partner |

---

## 9. Suggested order of operations

A reasonable two-week plan:

**Week 1 — unblock the deployment**

1. Day 1: get registry push credentials; do a `docker login registry-smarko-health.de` test
2. Day 1–2: confirm namespace name + storage answers (section 5 questions 1, 3, 4)
3. Day 2–3: implement chart edits (section 3.1–3.4); smoke-test locally
4. Day 3: push images
5. Day 4–5: first FOCUS install attempt; iterate on whatever breaks

**Week 2 — clinical requirement + dashboard**

1. Day 6–8: implement Option A from section 4.1 (inference-server → fall-dashboard internal HTTP)
2. Day 8–9: update mock_patient_dashboard HTML to render 3 states
3. Day 9–10: coordinate with Isa on his Flutter dashboard
4. Day 10: re-deploy to FOCUS; clinical team verifies the new states

If steps 1–2 of week 1 stall on partner response, that's the bottleneck — escalate fast, not a week in.

Good luck. Ping me on anything.
