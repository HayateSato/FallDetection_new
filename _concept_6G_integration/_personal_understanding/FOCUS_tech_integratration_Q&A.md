questions to ask 
1. Registry URL - can you share the full reference name is <registry>/<namespace>/<image>:<tag> where we should push our Docker Images
2. Namespace names 
  2.1. We are calling our namespace as "mcs-fall-detection". Do you have any policy or naming convention we should follow?
  2.2. What is the name of your namespace (namespace where your FHIR server, or InfluxDB is living)? we need this for cross-namespace URLs
3. aa
4. What StorageClass should we use for Postgres? For MinIO? Is there a default class, or do we need to specify one?
5. Do you enforce NetworkPolicy in this cluster? If yes — what labels/annotations should our pods carry so your policy allows the right traffic?
6. 


8. Here's our chart. Could you review values.yaml and tell us if everything you need is configurable, or if you'll need additional toggles?" Are there cluster-wide requirements (security context, labels, network policies) we should bake in by default?"

9. Does your registry require authentication for pulls? If yes, can you create an imagePullSecret in our namespace, or do we get the credentials and create it ourselves?





----

## 1. Registry URL — what it is and why we don't decide it

**What a "registry" is.** A container registry is a server that stores Docker images, the same way GitHub stores code. Public registry: Docker Hub (`docker.io`). Private registries: AWS ECR, GitHub Container Registry (`ghcr.io`), or self-hosted (Harbor, Nexus, GitLab registry).

**What "pushing an image" means.** When you run `docker build -t myimage:latest .` you create the image on your laptop. `docker push myimage:latest` uploads that image to a registry where other machines can pull it. The full reference name is `<registry>/<namespace>/<image>:<tag>` — e.g. `ghcr.io/anthropics/inference-server:v1.0.0`.

**Why we don't pick the URL ourselves.** Kubernetes pulls images by their full name. The K8s cluster runs in FOCUS's network. Their network typically:

- Has firewall rules that block outbound traffic except to whitelisted hosts
- May not have internet access at all (air-gapped clinical environment)
- Has a security policy requiring all images to be scanned by their security tool before deployment

So if I push to `ghcr.io/me/inference-server`, their cluster can't reach `ghcr.io` and the deployment fails with `ImagePullBackOff`. Our images **must** live at a registry their cluster can reach — usually one **they** operate inside their network.

**Two registries for two namespaces?** No, both namespaces' pods can pull from any registry the cluster can reach. The registry is a cluster-level resource, not namespace-scoped. One shared registry is normal.

**What to ask FOCUS:**

- "What's the registry URL for our images?" (e.g. `registry.charite.de/fall-detection/`)
- "Is it accessible to the cluster's nodes without VPN?"
- "Do you require image scanning / signing before we push?"
- "Can we get push credentials, or do you build/push from your CI?"

---

## 2. Namespace names

**What a K8s namespace is.** A logical partition inside a single K8s cluster. Like folders inside one filesystem. Resources in different namespaces are isolated from each other by default — they can't see each other's pods.

**Why we don't just pick the name.** Three reasons:

- **Their policies attach to specific names.** They probably have a NetworkPolicy that says "namespace `production-clinical` can talk to InfluxDB but `dev-experiments` cannot." We need to know which name their policies recognise.
- **Resource quotas.** They may have set "namespace X gets 8 CPU, 16 GiB RAM, 50 GiB disk." If we make up our own name, we get the cluster default (often 0 — the deployment will fail to schedule).
- **Naming conventions.** Most clinical organisations have rules like "namespace names are `<project>-<env>` lowercased, with no underscores." Picking arbitrary names violates their internal review.

You CAN ask "we want to call it `fall-detection`, is that okay?" — but you have to ask, because they might say "no, we already have one of those, use `fall-detection-mcs` instead."

**What to ask FOCUS:**

- "Can we use a namespace called `fall-detection` (or whatever you suggest)?"
- "What namespace will FOCUS resources we depend on (FHIR server, InfluxDB) live in? We need that name for the cross-namespace URLs."
- "Are there resource quotas we need to plan around (CPU, memory, storage)?"

---

## 3. FQDN — Fully Qualified Domain Name

A complete DNS path that identifies a service uniquely from anywhere in the network. Inside K8s, every Service has an FQDN of the form:

`<service>.<namespace>.svc.cluster.local`

Examples for our setup:

- `inference-server.fall-detection.svc.cluster.local` (port 8001)
- `postgres.fall-detection.svc.cluster.local` (port 5432)
- `mqtt-broker.fall-detection.svc.cluster.local` (port 1883)
- And from the FOCUS side: `fhir-server.<focus-namespace>.svc.cluster.local`

**Why it matters for our discussion:** the mobile app and FOCUS Patient Dashboard need to know *where* to send requests. Inside the cluster they use the FQDN above. From outside (e.g. mobile app on a phone), they go through an ingress URL (`fall-detection.charite.de`) — that's a different DNS layer.

When FOCUS DevOps tells you "your inference server FQDN is X" or "our FHIR server is at Y", they're giving you these in-cluster service addresses.

---

## 4. PVC class — PersistentVolumeClaim + StorageClass

Two related concepts, often referred to together.

**PersistentVolumeClaim (PVC):** the way a pod asks for storage. "I need 10 GiB of disk that survives my restart." K8s allocates it and binds it to the pod.

**StorageClass:** the *type* of storage. Different classes have different properties — SSD vs HDD, replicated vs single-node, with-snapshots vs without, fast vs cheap. Each cloud provider and bare-metal cluster has different StorageClasses. You don't pick the implementation; you pick the *quality tier*.

Example FOCUS may have:

- `fast-ssd` — for Postgres, MinIO data
- `standard` — for less critical things
- `backup` — slow, cheap, daily snapshots

In our `values.yaml` we currently have `storageClass: ""` (blank = "use cluster default"). That works on most clusters but can fail if FOCUS doesn't have a default class set, or if their default isn't appropriate (e.g. Postgres on slow HDD).

**What to ask FOCUS:**

- "What StorageClass should we use for Postgres? For MinIO?"
- "Is there a default class, or do we need to specify one?"
- "What's the maximum size you'll allocate to us?"

---

## 5. NetworkPolicy

A firewall rule defined inside K8s. By default, every pod can talk to every other pod across all namespaces — flat network. NetworkPolicies say "no, only allow these specific connections."

**Real example for our setup:** FOCUS might have a policy like:

> "Only pods in namespace `fall-detection` with the label `app=inference-server` may receive traffic on port 8001 from pods in namespace `focus-mobile-gateway`."
> 

That means:

- Our inference server gets traffic only from FOCUS's mobile gateway, not from random other pods
- Even if someone deploys a malicious pod elsewhere, it can't reach `/predict`

**Why this matters to us:**

- The mobile app and FOCUS Patient Dashboard need to be allowed by NetworkPolicy to reach our services. If the policy blocks them, calls fail with timeouts (not a clear error).
- Our cross-namespace calls (e.g. inference_server → FOCUS FHIR server) need to be explicitly allowed by their NetworkPolicy.

**What to ask FOCUS:**

- "Do you enforce NetworkPolicy in this cluster?"
- "If yes — what labels/annotations should our pods carry so your policy allows the right traffic?"
- "We need outbound HTTPS from our namespace to your FHIR server. Will that be allowed by default, or do we need a specific egress rule?"

---

## 6. SPA — Single Page Application

A web app that loads its HTML/JS/CSS once, then updates its content dynamically using JavaScript without reloading the page. React, Vue, Angular, Svelte, etc. produce SPAs.

**Contrast with MPA (Multi-Page App):** every click loads a fresh HTML page from the server. Old-school PHP / Django / Rails apps work this way.

**Why this comes up:** the FOCUS Patient Dashboard is most likely an SPA (it has a live biosignal panel). When Isa adds the fall panel, he's adding a component to that SPA. The fall panel uses our REST endpoints + SSE — no page reloads, just JS fetching new data.

This is mostly about Isa, not FOCUS DevOps. The DevOps side cares about it because:

- SPAs need their static assets served (HTML/JS/CSS) — usually from a CDN or nginx ingress
- SSE connections from SPAs are long-lived; some load balancers/proxies time them out at 30s and break the live updates. FOCUS's ingress (Traefik) has to be configured for long-lived connections.

---

## 7. Deployment vs StatefulSet

Both are K8s "kinds" (the `kind:` field in YAML). They both manage groups of pods, but with very different guarantees.

|  | Deployment | StatefulSet |
| --- | --- | --- |
| Pod identity | interchangeable; killed and replaced freely | each pod has a stable name (`postgres-0`, `postgres-1`) |
| Pod ordering | parallel; all start at once | sequential; pod-1 waits for pod-0 |
| Storage | usually no per-pod storage | each pod gets its own PVC that follows it |
| Use case | stateless apps (FastAPI, nginx, frontends) | databases, queues, anything with on-disk state per instance |

**Why Postgres needs StatefulSet, not Deployment.**

Imagine 3 Postgres replicas in a Deployment. K8s decides to roll out a new version: it kills `postgres-1`, the new pod comes up, but it has no idea which PVC to attach — they're shared / random. The new pod might attach to `postgres-2`'s old data, causing corruption. Worse: the pod has a different name on every restart, breaking master/replica configurations that pin to specific hostnames.

In a StatefulSet, `postgres-0` always has its own PVC, always boots first, always has the same hostname. That's the contract a database needs.

**When you'd use Deployment for stateful-looking services:** when storage is *external* (e.g. an inference server reading from S3 — the pod itself is stateless). Our `inference_server` is a Deployment, even though it deals with data, because the data lives in MinIO + Postgres, not on the pod's local disk.

**Is it always obvious?** No — there are edge cases. But the rule of thumb is: if killing a pod and bringing back a new one with a different name and a fresh disk would break things, it needs to be a StatefulSet.

---

## 8. values.yaml — what FOCUS DevOps actually edits

A Helm chart is a templated K8s deployment. `values.yaml` is the parameters file — values that get substituted into the templates. Think of it like environment-specific config.

Our chart exposes things like:

`images:
  inferenceServer: registry.placeholder/inference-server:v1.0.0
  pullPolicy: Always

ingress:
  host: fall-detection.placeholder.com

storageClass: ""

resources:
  postgres:
    requests: { cpu: 500m, memory: 1Gi }`

**Yes — FOCUS DevOps just fills in placeholders** in 90% of cases. If everything we need is already a `values.yaml` placeholder, they update those values and run `helm install` and it works.

**The 10% where it gets harder:**

- They may have requirements we didn't anticipate (e.g. all images must have `securityContext.runAsNonRoot: true`). If we didn't expose this as a value, they have to either patch the template directly (bad, makes upgrades hard) or ask us to add a values entry.
- They may use a private chart repository instead of installing from a folder.
- They may use ArgoCD / Flux (GitOps) — they don't run `helm install` manually, they commit values to a Git repo and a controller deploys it.

**What to say to FOCUS DevOps:**

- "Here's our chart. Could you review `values.yaml` and tell us if everything you need is configurable, or if you'll need additional toggles?"
- "What's your deployment workflow — manual `helm install`, GitOps, or something else?"
- "Are there cluster-wide requirements (security context, labels, network policies) we should bake in by default?"

---

## 9. "If your registry needs auth"

This is **registry authentication** — not application authentication. Two completely different things.

- **Application auth** (what your `X-API-Key` does): a user/client authenticates to your FastAPI app to make API calls. Browser ↔ app.
- **Registry auth**: when K8s tries to *download an image* from a private registry, the registry says "who are you, prove it." K8s presents credentials. If they're wrong, no image, no pod, no deployment.

In K8s terms, registry auth uses an `imagePullSecret` — a Kubernetes Secret containing a username/password (or token) for the registry. Pods reference it via:

`spec:
  imagePullSecrets:
    - name: focus-registry-creds`

If FOCUS uses a private registry, they create an imagePullSecret with the registry credentials in the `fall-detection` namespace, and our chart references it. Our `values.yaml` already has a placeholder for this (`imagePullSecretName: ""`).

**What to ask FOCUS:**

- "Does your registry require authentication for pulls?"
- "If yes, can you create an `imagePullSecret` in our namespace, or do we get the credentials and create it ourselves?"
- "What's the secret name we should reference in the Helm chart?"

---

## Summary checklist for the FOCUS DevOps meeting

Bring these questions, in roughly this order:

1. Registry URL + auth + image scanning policy
2. Our namespace name + their namespace names (for FHIR / InfluxDB FQDNs)
3. Resource quotas, ingress hostname, StorageClass for Postgres + MinIO
4. NetworkPolicy enforcement + required pod labels
5. Deployment workflow (manual helm vs GitOps) + values.yaml review
6. Cross-namespace allowed traffic (we → FHIR, mobile app → us)
7. SSO / JWT plan (touched in Step 11.5 — different conversation, same person likely)

Walk in with that list and you'll come across as someone who knows what they're asking for. Most of these are 5-minute answers from their side; the longest discussion will be NetworkPolicy because it requires them to map your traffic flows to specific rules.



----
### FOLLOW UP QUESTIONS
1. Registry URL --> so we cannot asnwer or know where their cluster can reach and where it cannot. hence we need their input?

2. for now i wll call it mcs-fall-detection and i will ask if this is accetable or not

3. what dose svc stand for in the DNS path?

4. PersistentVolumeClaim + StorageClass. i understood that it is not necessary to assign but if we wanted to assgin, do we need to assign PersistentVolumeClaim + StorageClass for each pod? (this is only relevant for the component that writes data in disk right? so postgres and minIO? what about prometheus, grafana and MQTT broker? you told me these also store in its own storage disk with shorter retention) (only those declared as StatefulSet in yaml file?)

5.  By default, every pod can talk to every other pod across all namespaces. you said this, so usually pod A in namespace a can talk to pod B in namespace b? where (in which file) can you set this kind of netoworking config/policy in k8s? or is it done outside k8s?

7. Imagine 3 Postgres replicas in a Deployment. K8s decides to roll out a new version: it kills postgres-1, the new pod comes up, but it has no idea which PVC to attach 
in this sentence i understood the following two words as below, is this correct? replica = instance? also PVC = running databse instance? 

9. where and how do you define imagePullSecret if they have a private registry? in values.yaml?

--- 

## 1. Registry URL — confirmation

Correct. We don't know:

- Whether their cluster has internet access at all
- Which external hosts (if any) are whitelisted in their egress firewall
- Whether they have an internal registry already running for other projects
- Their image scanning / signing policy

These are FOCUS internal-network concerns. Only their DevOps team can answer them. Even if we self-host a registry on, say, AWS ECR, their cluster can't pull from it unless their network is configured to allow it — which is their call.

---

## 2. mcs-fall-detection

Reasonable name. When asking, also tell them:

- "We use a single namespace for our 8 services" (sets expectations on resource quota)
- "We need to know your namespaces too, so our cross-namespace URLs (FHIR, InfluxDB) point at the right thing"

If they push back on the name, they'll usually suggest one that fits their convention (e.g. `mcs-falldetect-prod`).

---

## 3. `svc` = "service"

Literally short for "service". The full suffix is:

`<service-name>.<namespace>.svc.cluster.local`

Each piece:

- `<service-name>` — the K8s `Service` resource name (defined in YAML, `kind: Service`)
- `<namespace>` — the namespace the Service lives in
- `svc` — literally the string "svc" (means "this is a Service-type resource")
- `cluster.local` — the cluster's DNS root (configurable, but `cluster.local` is the standard default)

K8s also exposes Pods directly by similar names (`<pod>.<namespace>.pod.cluster.local`), which is where the `pod` segment shows up. The `svc` part disambiguates "I'm asking for a service, not a pod."

---

## 4. Which components need PVCs

Your intuition is roughly right but the list is broader than just Postgres + MinIO. The rule is: **anything that needs disk persistence beyond a single pod's lifetime needs a PVC.**

| Component | Needs PVC? | Stores |
| --- | --- | --- |
| Postgres | **yes** | The DB itself (`/var/lib/postgresql/data`) |
| MinIO | **yes** | All `.pkl` artifacts |
| Prometheus | **yes** | TSDB metric files (30-day retention) |
| Grafana | **yes** | Its own SQLite (dashboards + datasource configs + users) |
| MQTT broker (Mosquitto) | **maybe** | Only if you enable persistent sessions / retained messages. We don't, so optional. |
| MLflow tracking server | no | All state lives in Postgres + MinIO; MLflow itself is stateless |
| inference_server | no | Stateless (in-memory model loaded from MinIO at swap time) |
| fall_dashboard | no | Stateless (writes to Postgres via SSE, no local files) |
| ml_dashboard | no | Stateless |

**Does each pod get its own PVC?** It depends on `kind`:

- **StatefulSet** — yes, automatically. K8s gives each pod (`postgres-0`, `postgres-1`, `postgres-2`) its own dedicated PVC via `volumeClaimTemplates`. The PVC follows the pod by name across restarts. This is what databases need.
- **Deployment + PVC** — typically one shared PVC across all pods (or 1 replica). Used for things like Grafana where you want disk persistence but don't run multiple replicas. If you need multi-replica with shared writable storage, you use a special StorageClass that supports `ReadWriteMany` (cloud-managed file shares).

So your reading is mostly right — StatefulSet ↔ per-pod PVC. Grafana and Mosquitto can be Deployments with a single shared PVC because they're typically single-instance.

For our chart, the kinds shake out roughly:

- **StatefulSet:** Postgres, Prometheus, MinIO
- **Deployment + PVC:** Grafana, Mosquitto (if persistence)
- **Deployment, no PVC:** MLflow, inference_server, fall_dashboard, ml_dashboard

You don't need to declare a StorageClass for *each* PVC — you can set a default at the cluster level OR specify per-PVC. We currently do per-PVC with `storageClass: ""` (= "use cluster default") to keep `values.yaml` simple. FOCUS may want to override that for Postgres specifically (e.g. fast SSD).

---

## 5. NetworkPolicy default behaviour + where it's defined

**Default behaviour:** flat network. Every pod in any namespace can reach every other pod in any namespace, on any port. K8s out of the box does **not** firewall pod-to-pod traffic.

NetworkPolicies are *opt-in*. Once you define a NetworkPolicy targeting a namespace or pod, it switches to default-deny for the connections it covers — anything not explicitly allowed is blocked.

**Where it's defined:** in YAML, like any other K8s resource. The `kind:` is `NetworkPolicy`. It's a normal K8s API object, not something configured outside K8s.

Example — only allow the FOCUS mobile gateway namespace to reach our inference server on 8001:

`apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-mobile-to-inference
  namespace: mcs-fall-detection
spec:
  podSelector:
    matchLabels:
      app: inference-server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: focus-mobile-gateway
      ports:
        - protocol: TCP
          port: 8001`

This file lives in the chart (or applied separately by `kubectl apply -f`). It says: "for any pod in `mcs-fall-detection` labeled `app=inference-server`, only accept incoming TCP connections to port 8001 from pods in the namespace labeled `name=focus-mobile-gateway`. Everything else, deny."

**Important:** NetworkPolicies only do anything if the cluster's networking layer (CNI plugin: Calico, Cilium, etc.) supports them. Some clusters install K8s but skip the CNI piece, in which case the YAML is silently ignored. That's why you ask FOCUS "do you enforce NetworkPolicy?" — if they don't, our policy YAML is just decoration.

---

## 7. Replica vs instance vs PVC — terminology fix

You got one right and one wrong. Let me draw it cleanly.

| Term | What it is |
| --- | --- |
| **Pod** | A running container (or small set of containers). The actual process. *This is the running database instance.* |
| **Replica** | "How many copies of the same pod do we run?" `replicas: 3` means 3 pods, all running the same image, behind one Service. So yes — **replica ≈ instance ≈ pod**. |
| **PVC (PersistentVolumeClaim)** | A request for disk space. Just storage — empty by default until a process writes to it. **Not the database.** It's the *disk* the database stores its files on. |
| **PV (PersistentVolume)** | The actual disk volume that backs a PVC. K8s provisions one when a PVC is created. PVC is the "claim ticket"; PV is the actual disk. |

Mental model:

                      `Pod (the running process)
                       │
                       │ mounts at /var/lib/postgresql/data
                       ▼
                      PVC (a claim — "I want 10 GiB")
                       │
                       │ bound to
                       ▼
                      PV (the actual 10 GiB disk volume)
                       │
                       │ provisioned from
                       ▼
                      StorageClass (the type — fast SSD, etc.)`

**So when I said "the new pod has no idea which PVC to attach":** I meant the new database process (the new pod) doesn't know which old disk to mount to read its existing data. The pod is the "database process"; the PVC is the "disk where data lives". They're separate. A StatefulSet keeps them paired automatically.

**Why this matters in practice:**

- Pod dies → new pod starts → looks up its PVC by name → mounts the same disk → continues with all the existing data
- PVC dies → all data is gone, even if the pod is fine
- You can resize a PVC (grow the disk) without restarting the pod
- You can take a snapshot of a PVC (backup) without touching the pod

---

## 9. imagePullSecret — where and how

Two parts: the secret itself, and the reference to it.

**Part 1: create the Secret.** This is a K8s resource of `kind: Secret` containing the registry credentials. You normally create it via `kubectl` (you do NOT put it in values.yaml — that's a Git-tracked file, secrets shouldn't be in Git):

`kubectl create secret docker-registry focus-registry-creds `
  --docker-server=registry.charite.de `
  --docker-username=<user> `
  --docker-password=<password> `
  --docker-email=<email> `
  --namespace=mcs-fall-detection`

K8s wraps this into a Secret object. It exists in the cluster, not in the chart.

**Part 2: reference the secret name in the chart.** In `values.yaml`:

`imagePullSecretName: focus-registry-creds`

Then in each pod template:

`spec:
  imagePullSecrets:
    - name: {{ .Values.imagePullSecretName }}
  containers:
    - name: inference-server
      image: registry.charite.de/mcs-fall-detection/inference-server:v1.0.0`

K8s sees the `imagePullSecrets` list, looks up the named Secret in the same namespace, and uses those credentials when pulling the image.

**Who creates the Secret in practice?** Usually FOCUS DevOps creates it once as part of namespace setup, and you reference its name. They don't share the credentials with you because:

- Credentials should not be in Git
- They want to control rotation
- They might use service accounts / OIDC instead of static credentials, in which case there's no password at all

**What to ask FOCUS:**

- "Will you create the imagePullSecret in our namespace, or do we?"
- "What's the secret name we should reference?"
- "Is it a static username/password, or are you using a service-account token / OIDC?"

If they say "we'll create it, the name is `focus-registry-creds`" — great, you put `imagePullSecretName: focus-registry-creds` in values.yaml and you're done.




----



----

## 1. The strategic question — who changes what

**You're absolutely right, and the standard pattern is exactly what you instinctively proposed.**

The professional flow is:

`Us:              FOCUS DevOps:
─────────────    ────────────────────────────────────────────
Helm chart  ──→  values-overrides.yaml in their internal Git
                 (contains real registry URLs, hostnames,
                  secret names, NetworkPolicy labels)
                                    │
                                    ▼
                 helm install --values <our-values> --values <their-overrides>`

**Why this is the right pattern:**

1. **They never share sensitive info.** Registry credentials, internal hostnames, security policies — all stay on their side.
2. **You don't maintain their environment.** If they change their registry next quarter, you don't need to update your chart.
3. **It matches how Helm is designed to work.** `values.yaml` provides defaults; `-values overrides.yaml` (or `-set`) lets the operator customize per environment.
4. **Their security team prefers it.** They control the secret-creation step, not us.

What you ship them:

- The Helm chart (folder of templates + default `values.yaml`)
- A list of placeholder values they need to override (the `<placeholder>` strings in your default file)

What they do:

- Create their own `values-overrides.yaml` with real values, kept in *their* internal Git
- Create the imagePullSecret directly in the cluster via `kubectl`
- Run `helm install` with both files

**This means your email isn't really "tell us your values" — it's "here's the chart, here are the override points, please confirm you can fill them in and that the chart's defaults work for you."** Much less awkward, no sensitive info exchange.

---

## 2. Refined email — ready to send

Here's a clean version your colleague can paste into an email:

`Subject: Helm chart for fall-detection deployment — review request

Hi <DevOps team>,

We've prepared the Helm chart for the MCS fall-detection service and would
like your help confirming a few cluster-specific points before we proceed
with deployment. The chart lives at [link/folder]; values.yaml at [link].

We'd prefer not to receive any sensitive values directly — instead, we're
asking you to confirm where you'll plug them in on your side
(values-overrides.yaml / Secrets / cluster config).

------------------------------------------------------------
Section 1 — Container registry
------------------------------------------------------------

a) What's the full registry path our images should live at?
   Format: <registry>/<namespace>/<image>:<tag>
   (We'll push our 3 images: inference-server, fall-dashboard, mlflow-server)

b) Will you create the imagePullSecret in our namespace, or should we?
   If you create it: please share the secret name (we'll reference it
   in values.yaml).

c) Do you require image scanning / signing before deployment?
   If yes, what's the workflow?

------------------------------------------------------------
Section 2 — Namespaces
------------------------------------------------------------

a) We propose calling our namespace "mcs-fall-detection".
   Does this fit your naming convention? If not, what should we use?

b) What's the namespace name where the FHIR server and FOCUS-hosted
   InfluxDB live? We need it for cross-namespace service URLs.

c) Are there resource quotas attached to namespaces in this cluster?
   Our chart's defaults: ~8 vCPU / ~10 GiB RAM total across 8 services.

------------------------------------------------------------
Section 3 — Storage
------------------------------------------------------------

a) What StorageClass should we specify for Postgres and MinIO?
   (Our default is "" = cluster default. Confirm if that's appropriate.)

b) Maximum persistent volume size we can request?
   Our defaults: Postgres 20 Gi, MinIO 50 Gi, Prometheus 10 Gi.

------------------------------------------------------------
Section 4 — Networking
------------------------------------------------------------

a) Do you enforce NetworkPolicy in this cluster?
   (Our chart can include NetworkPolicy YAML if so.)

b) If yes, what labels/annotations should our pods carry so your
   policies allow:
     - Inbound to inference-server :8001 from the mobile gateway
     - Outbound from inference-server to FHIR server (HTTPS)
     - SSE traffic from fall-dashboard :8002 to caregiver browsers

c) What ingress hostname should we use? (We have a placeholder
   "fall-detection.example.com" — please replace with the real one.)

d) Is Traefik confirmed as the ingress controller? Our ingress.yaml
   uses `ingressClassName: traefik`.

------------------------------------------------------------
Section 5 — Deployment workflow
------------------------------------------------------------

a) What's your deployment workflow — manual `helm install`, GitOps
   (ArgoCD/Flux), or something else?

b) Could you review values.yaml and tell us if everything you need is
   configurable, or if you'll need additional toggles?

c) Are there cluster-wide requirements (security context like runAsNonRoot,
   labels for compliance, network policies) we should bake into the
   templates by default?

------------------------------------------------------------
Section 6 — Authentication (future)
------------------------------------------------------------

a) For the unified caregiver web app + admin dashboards (separate doc),
   we'll need to validate JWTs from your SSO. What's the issuer URL and
   the role claim format?
   (Not blocking for initial deployment — separate sync would help.)

------------------------------------------------------------

Would a 30-min call work for going through these together? Happy to
walk through the chart structure if useful.

Thanks,
<your name>`

A few things this draft does intentionally:

- Calls out up front that we don't want to receive sensitive values — sets the tone
- Numbers each question so they can answer inline
- Gives our defaults / proposed values where possible, so they can just say "confirm" or correct
- Includes a final "Section 6 — Authentication" as a future-touchpoint, not blocking, so they know it's coming
- Ends with a 30-min call offer — most DevOps teams prefer a synchronous run-through to async ping-pong

---

## 3. Follow-up questions

### "Change our namespace to mcs-fall-detection in all documentation"

**Files updated:**

- `REFACTOR_DOCUS/helm_guide.md` — all `-namespace`, `n`, `kubectl create namespace`, diagram label, `values.yaml` placeholder, FQDN
- `REFACTOR_DOCUS/todo.md` — all `-namespace`, `n`, `kubectl delete namespace`
- `REFACTOR_DOCUS/isa_integration_guide.md` — FQDN
- `handover_docs/hand_over_doc_isa.txt` — "OUR NAMESPACE" header line

**Files I deliberately did NOT change** (these use `fall-detection` as the project/chart/model name, not the namespace):

- Chart folder path (`helm/fall-detection/`), `helm install fall-detection ...` (release name), Chart.yaml `name: fall-detection`
- ConfigMap/Secret names (`fall-detection-config`, `fall-detection-secrets`)
- MLflow registered model name (`fall-detection-xgboost`)
- Project description prose ("our fall-detection backend")

If you want the chart name and the ConfigMap/Secret names also renamed for consistency, say so and I'll do another pass — but those are typically kept as the project's brand name regardless of the namespace they're deployed into.

---

### "PVC dies → all data is gone, even if the pod is fine — but you also said K8s provisions PV when PVC is created"

Good catch — I oversimplified. The accurate answer depends on the PV's **`reclaimPolicy`** (also called `persistentVolumeReclaimPolicy`):

`PVC                     PV (the actual disk)
 │  bound to            │  has a reclaimPolicy field
 ▼                      ▼
"I claim 10 GiB"        "Retain" / "Delete" / "Recycle"`

When you delete the PVC, here's what happens to the PV:

| reclaimPolicy | What happens when PVC is deleted | Data |
| --- | --- | --- |
| `Delete` | PV is also deleted, underlying disk wiped | **gone** |
| `Retain` | PV stays around in `Released` state, disk stays | **preserved** — you can manually create a new PVC and bind it to that PV |
| `Recycle` | DEPRECATED — wiped data and made the PV available again | (don't use) |

**The default for dynamic provisioning is `Delete`.** Most clusters dynamically provision PVs (the StorageClass creates a PV automatically when you create a PVC), and these are usually `Delete` by default — so accidental `kubectl delete pvc` typically does destroy the data.

**For production databases, the right configuration is:**

`apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data
spec:
  storageClassName: <class-with-Retain-policy>
  ...`

Or override the StorageClass to `reclaimPolicy: Retain`. With this, deleting the PVC doesn't lose data — the PV becomes "Released" and an admin can recover it. This is something to ask FOCUS DevOps about: "what's the reclaimPolicy of the StorageClass you'll allocate to us?"

So my earlier statement should have been: **"PVC dies → if reclaimPolicy is `Delete` (the default), the PV and data are gone; if `Retain`, they're preserved but require manual rebinding."**

---

### "Difference between chart and cluster"

Two completely separate concepts. Easy to confuse because the words sound similar but they're not related.

|  | Chart | Cluster |
| --- | --- | --- |
| **What it is** | A folder of YAML templates + values.yaml + Chart.yaml | A running Kubernetes installation — actual machines, actual processes |
| **Where it lives** | In Git, on your laptop, on a Helm repo | On servers (cloud or on-prem) |
| **What's in it** | Files describing what to deploy | Running pods, services, secrets, network rules — the *real things* |
| **Mutable?** | Edited by developers in Git | Mutated by `kubectl`/`helm install` commands |
| **Sensitive data?** | NO — chart is checked into Git, secrets must NEVER live here | YES — Secret objects in the cluster contain real credentials |

**The relationship:**

`Chart (in Git, version-controlled)
  │  helm install
  ▼
Cluster (running pods, secrets, services)`

The chart is the **blueprint**; the cluster is the **building**. You install the chart **into** the cluster.

**Where things live:**

| Thing | Lives in chart? | Lives in cluster? |
| --- | --- | --- |
| Pod template (YAML describing the pod) | yes | no — that's a template |
| The actual running pod | no | yes |
| Default values for config | yes (values.yaml) | no |
| The actual Secret with passwords | NO | yes |
| The Secret's *name* (so the pod template can reference it) | yes | no |

**Why this matters for your imagePullSecret question:** the Secret object itself (containing the real password) lives in the cluster. The chart just references it by name. So:

`# In chart's pod template (this goes to Git, fine):
spec:
  imagePullSecrets:
    - name: focus-registry-creds   # just the NAME, no password

# In cluster (kubectl create secret, NOT in Git):
apiVersion: v1
kind: Secret
metadata:
  name: focus-registry-creds
data:
  .dockerconfigjson: <base64-encoded-credentials>   # the actual password`

**Same separation applies to Postgres passwords, API keys, JWT signing keys, etc.** Chart references names. Cluster holds values. Never put real values in the chart.

This is also why FOCUS DevOps creating the Secret is the right pattern — they create it in the cluster, you reference it by name in the chart, no credential ever travels through email or chat.