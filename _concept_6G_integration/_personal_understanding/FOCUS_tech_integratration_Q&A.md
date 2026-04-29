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


----

## "Chart lives in Git" — what I meant

A Helm chart is just a folder of YAML files (templates + values + Chart.yaml). Like any code, it's stored in version control. So when I say "the chart lives in Git," I mean:

- Your chart files (`helm/fall-detection/templates/...`, `values.yaml`, etc.) are committed to a Git repository alongside your application code
- Each commit captures a specific version of the chart
- You can tag releases (`v1.0.0`), branch for changes, review via pull requests
- The chart is **source-controlled** — it's text, it's reviewable, it's diffable

In contrast, the cluster contains *running things* — pods, secrets, services. That state is in K8s's database (etcd), not in Git. The flow is "Git → helm install → cluster."

The reason this distinction matters for *secrets*: anything that goes into Git is visible to anyone with repo access. Secrets must never be in Git, so they never go into the chart — they go directly into the cluster via `kubectl create secret`.

## "helm chart = helm — same thing?"

No, two different concepts:

| Thing | What it is | Analogy |
| --- | --- | --- |
| **Helm** | The CLI tool. You run `helm install`, `helm upgrade`, `helm list`. | npm, apt, brew |
| **Helm chart** | A package format. A folder with a specific structure that Helm knows how to install. | a .deb file, a node module |

Helm is *what you use*. A Helm chart is *what you have*.

When someone says "send me the Helm chart," they mean "send me the folder of YAML files." When they say "use Helm to install it," they mean "run the `helm` command."

---

## Source code vs docker image vs helm chart — what we share

**Three separate things, three different audiences:**

| Artifact | Format | Goes where | Recipient |
| --- | --- | --- | --- |
| **Source code** | Python files in Git | our internal repo (private) | only us — we read and edit it |
| **Docker images** | Binary blobs (compiled images) | container registry | FOCUS cluster pulls them at deploy time |
| **Helm chart** | Folder of YAML | shared as folder/tarball/Helm repo | FOCUS DevOps reviews + installs |

**For the FOCUS handover specifically: we share the chart + image references. We do NOT share the Python source.**

Why this works:

- The Docker image already has the compiled Python inside — the cluster runs the bytecode, FOCUS DevOps doesn't need to read it
- The Helm chart has just enough info for them to deploy it (image name, ports, env vars, resource requests)
- Our IP / source code stays in our private repo

The chart will reference images like `registry.charite.de/mcs-fall-detection/inference-server:v1.0.0` — FOCUS DevOps see only the image *reference*, not what's inside it.

---

## Mock FOCUS namespace dry-run — yes, very good idea

This is exactly the kind of test that catches Helm/networking/NetworkPolicy issues before they hit FOCUS's cluster. I'd actually do this *before* sending the chart-handover email — it shows them you've validated cross-namespace traffic works on your side, makes the handover much smoother.

Let me add this as a new step in todo.md.

## Email template — handover (after the dry-run passes)

This goes to the same FOCUS DevOps person, after the questions email and after Step 12.5 verifies the chart works in two-namespace mode. Use it once their answers are integrated:

`Subject: Helm chart ready — taking over the deployment for fall-detection

Hi <DevOps team>,

Thanks for your answers to our setup questions. We've integrated everything
into the chart and validated it locally with a two-namespace simulation
(mock FOCUS namespace alongside ours). It now installs cleanly and all
cross-namespace traffic — FHIR push, SSE — works.

We're ready to hand it over for deployment in your cluster.

------------------------------------------------------------
What we're delivering
------------------------------------------------------------

1. The Helm chart at:
   <git URL or tarball link>
   tag: v1.0.0

2. The Docker images (already pushed to your registry per your earlier
   guidance):
   <registry>/mcs-fall-detection/inference-server:v1.0.0
   <registry>/mcs-fall-detection/fall-dashboard:v1.0.0
   <registry>/mcs-fall-detection/mlflow-server:v1.0.0

We're NOT sharing source code — the images contain the compiled artifact,
which is all the cluster needs. If your security team requires a source
audit, we can arrange a separate review under NDA.

------------------------------------------------------------
What you need to do (in your internal Git, never shared back to us)
------------------------------------------------------------

Create a `values-overrides.yaml` with the values you supplied:

  namespaces:
    ours:  mcs-fall-detection
    focus: <your-namespace-name>

  images:
    pullSecretName: <your-imagePullSecret-name>
    pullPolicy:     Always

  ingress:
    host: <your-ingress-hostname>

  storageClass: <your-storageclass>

  postgres:
    password: <generate-and-store-in-your-Secret>

  inferenceServer:
    apiKey: <generate-and-store-in-your-Secret>

  fhir:
    serverUrl: http://<your-fhir-host>/fhir

(The full list of overridable values is in `values.yaml` with comments.
Anything not in your override file will use our defaults.)

------------------------------------------------------------
Install
------------------------------------------------------------

helm install mcs-fall-detection ./helm/fall-detection \
  --namespace mcs-fall-detection \
  --create-namespace \
  --values values.yaml \
  --values <your-values-overrides.yaml>

Or via your GitOps pipeline if you use ArgoCD/Flux — same set of files.

------------------------------------------------------------
Smoke test after install
------------------------------------------------------------

1. kubectl get pods -n mcs-fall-detection
   All 8 pods should reach Running. The migrate-job and bucket-creation
   job should reach Completed.

2. kubectl exec -n mcs-fall-detection deploy/inference-server -- curl http://localhost:8001/health
   Expect: {"status":"ok","model_version":"v0",...}

3. From the FOCUS namespace, test cross-namespace reach:
   curl http://fall-dashboard.mcs-fall-detection.svc.cluster.local:8002/api/patients

If any of these fail, we're available for a debug session.

------------------------------------------------------------
What we still need from you (post-install)
------------------------------------------------------------

- Confirmation the ingress is reachable from outside the cluster
  (so the mobile app can POST to /predict)
- The real FHIR server URL (we currently have a placeholder)
- The SSO issuer URL + JWT role claim format
  (separate sync — for the admin/caregiver role split)

------------------------------------------------------------
Support
------------------------------------------------------------

We're happy to be on a call during the first install if useful.
Estimated install time: ~10 minutes once values-overrides.yaml is ready.

Thanks,
<your name>`

**Key tones:**

- Opens with "thanks for your answers" — reciprocates their effort
- Explicit about not sharing source code (preempts a likely question)
- Tells them the values they should override but doesn't list FOCUS-side specifics (they fill those in their own Git)
- Smoke test commands they can run themselves to validate
- Clear about what's still pending (ingress check, FHIR URL, SSO) so nothing falls through the cracks
- Offer to be on call — most DevOps appreciate it but won't always take it up



## 1. What "smoke test" means

A quick, low-effort test that checks the basic "does anything obviously break" health of a system after you've assembled it. The term comes from electrical engineering: you turn on a new circuit board and watch for smoke. If no smoke, you proceed to real testing.

For us, a smoke test of the chart looks like:

1. `helm install ...` succeeds (no errors)
2. `kubectl get pods` — all 8 pods reach `Running` or `Completed`
3. `wget /health` on inference-server returns 200
4. `wget /api/patients` on fall-dashboard returns JSON
5. Cross-namespace probes (the 4 in `helm/mock-focus/test.ps1`) all pass
6. Open the mock patient dashboard in the browser, manually trigger a fall via mock_app, see the live SSE flag flash red

That's it. ~5 minutes of work. The point is to catch silly errors (missing files, wrong env vars, broken DNS) before deploying to a real cluster — not to comprehensively test every feature. Comprehensive testing comes after the smoke test passes.

So when the README says "not yet smoke-tested end-to-end," it means: the chart was written, `helm lint` passes, but nobody has run `helm install` and watched it go green-pod-by-green-pod yet. That's exactly what Step 12.5 in `todo.md` is for.

---

## 2. Stop the mock-focus namespace

Use the `teardown.ps1` script that's already in the chart folder:

`.\helm\mock-focus\teardown.ps1`

Equivalently (if you want to be explicit about what's happening):

`helm uninstall mock-focus -n mock-focus
kubectl delete namespace mock-focus`

Verify it's gone:

`helm list -A
# should NOT show mock-focus

kubectl get namespaces | findstr mock-focus
# should print nothing`

You can re-install later with `.\helm\mock-focus\install.ps1` whenever you want to validate cross-namespace traffic again. Tear-down doesn't damage the mock-focus chart itself — only the running cluster state.

---

## 3. "Run each service first vs helm-and-everything-runs" — actually the same pattern

You're right that it *feels* different but it isn't. Both workflows have **two phases**: build images, then helm install. With mock-focus the script chained them together so you didn't notice. Let me draw it:

### What mock-focus actually did

`.\helm\mock-focus\build.ps1
   │
   ├── docker build mock-fhir.Dockerfile -t fd-mock-fhir
   └── docker build mock-patient-dashboard.Dockerfile -t fd-mock-patient-dashboard
                          ▲
                          │  these images now in your local Docker cache
                          │
.\helm\mock-focus\install.ps1
   │
   ├── helm install mock-focus ./helm/mock-focus
   │      │ K8s pulls 'influxdb:2.7' from Docker Hub
   │      │ K8s loads 'fd-mock-fhir' from local cache (pullPolicy: Never)
   │      │ K8s loads 'fd-mock-patient-dashboard' from local cache
   │      ▼
   │   pods start running
   │
   └── helm install mcs-fall-detection ./helm/fall-detection
       │  ... but if you didn't build inference-server / fall-dashboard images,
       │      this part fails with ImagePullBackOff
       ▼
       (fails silently here in your case)`

So the build step DID happen for mock-focus — the script just wrapped it. Same pattern applies to the real chart, but I documented it as two separate steps to make the dependency explicit.

### Why we need `docker build` at all (not just `helm install`)

There are **two kinds of images** in the chart:

| Image | Where it lives | Who downloads it |
| --- | --- | --- |
| `postgres:16-alpine`, `eclipse-mosquitto:2`, `minio/minio:latest`, `prom/prometheus:latest`, `grafana/grafana:10.4.0`, `ghcr.io/mlflow/mlflow:latest` | **Docker Hub / GHCR** (public) | K8s pulls them automatically when a pod starts |
| `inference-server:latest`, `fall-dashboard:latest` | **only your laptop's Docker cache** (we haven't pushed them anywhere) | Nobody downloads them — they must already be there |

The public ones K8s knows how to fetch. Our custom images aren't published anywhere yet, so K8s can only find them if they're sitting in the Docker daemon's image cache on the node it's running on. Docker Desktop K8s shares the daemon's cache, which is the only reason this works locally.

**`pullPolicy: Never`** in `values.yaml` is what enforces "don't try to pull from a registry, just use the local cache." If we set it to `Always` (the production default), K8s would try to fetch `inference-server:latest` from a registry and get a 404.

In FOCUS production:

- We push our images to FOCUS's registry: `<focus-registry>/inference-server:v1.0.0`
- `pullPolicy: Always` (or `IfNotPresent`)
- `docker build` happens once on a CI runner, not on every developer's laptop
- Then `helm install` is genuinely a one-step command

So your intuition is correct that **eventually** it should be one command. We're at the local-dev stage where you're standing in for the CI runner — hence the manual `docker build` first.

### The same logic applied to mock-focus

For mock-focus, the same two image categories exist:

| Image | Where it lives | Built by |
| --- | --- | --- |
| `influxdb:2.7` | Docker Hub | nobody — auto-pulled |
| `fd-mock-fhir:latest`, `fd-mock-patient-dashboard:latest` | local Docker cache only | `build.ps1` |

The `build.ps1` script handled the second category in advance. Then `install.ps1` ran `helm install` and everything came up.

---

## Bottom line

When you tear down mock-focus and focus on the real chart, the workflow you'd run is:

`# 1. Tear down mock-focus
.\helm\mock-focus\teardown.ps1

# 2. Build the two custom images (one-time, or after code changes)
docker build -f inference_server/Dockerfile -t inference-server:latest .
docker build -f fall_dashboard/Dockerfile  -t fall-dashboard:latest  .

# 3. Install the real chart
helm install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m

# 4. Watch pods come up
kubectl get pods -n mcs-fall-detection -w

# 5. Smoke-test
kubectl exec -n mcs-fall-detection deploy/inference-server -- wget -qO- http://localhost:8001/health
kubectl port-forward -n mcs-fall-detection svc/fall-dashboard 8002:8002
# then open http://localhost:8002/ in a browser`

If pods don't all reach `Running` within ~5 minutes, paste the output of `kubectl get pods -n mcs-fall-detection` and `kubectl describe pod <stuck-pod-name> -n mcs-fall-detection` — we'll diagnose from there. That's exactly the smoke-test loop the README mentions.

## 1. Yes — only the custom images need pushing to FOCUS's registry

The split is between **public** images (already on Docker Hub) and **our** images (only on your laptop). FOCUS DevOps doesn't need our copies of public images — their cluster pulls those from Docker Hub directly.

| Image | Push to FOCUS registry? | Why |
| --- | --- | --- |
| `inference-server` | **yes** | our code |
| `fall-dashboard` | **yes** | our code |
| `infrastructure/mlflow/` (the custom MLflow image with psycopg2/boto3) | **yes** | we customised the base image |
| `ml_dashboard`, `server_health` | **yes** (when added to chart) | our code |
| `postgres:16-alpine` | no | already on Docker Hub |
| `eclipse-mosquitto:2` | no | already on Docker Hub |
| `minio/minio:latest` | no | already on Docker Hub |
| `prom/prometheus:latest` | no | already on Docker Hub |
| `grafana/grafana:10.4.0` | no | already on Docker Hub |
| `fd-mock-fhir`, `fd-mock-patient-dashboard` (mock-focus) | **never** | dev-only, simulates FOCUS, must not ship |

So 3 images we push today (4–5 once ml_dashboard / server_health are added to the chart). The rest FOCUS's cluster pulls from public registries on its own.

---

## 2. `docker build` makes the image; `docker push` ships it — two separate steps

Your understanding is **almost** right. `docker build` creates the image in your local Docker daemon's cache. To get it into a registry, you need a second step that tags it with the registry's address and pushes:

`# Step 1 — build (already done by the chart README's Section 4)
docker build -f inference_server/Dockerfile -t inference-server:latest .

# Step 2 — tag for the destination registry (FOCUS-supplied URL)
docker tag inference-server:latest registry.charite.de/mcs-fall-detection/inference-server:v1.0.0

# Step 3 — push
docker push registry.charite.de/mcs-fall-detection/inference-server:v1.0.0`

Why the extra `docker tag`? Docker images are addressed by `<registry>/<namespace>/<name>:<tag>`. When you build with just `-t inference-server:latest`, the image is implicitly tagged for "no registry, just locally" — that's why `pullPolicy: Never` is the only way K8s finds it. To push, the image needs a name that includes the destination registry, hence the `docker tag` step.

In CI the build/tag/push are typically one script. For local dev they stay separate so you don't accidentally push every test build.

---

## 3. Two namespaces — yes, this is from the values.yaml drift

You're right. Earlier we updated all the **docs** to say `mcs-fall-detection`, but the chart's actual `values.yaml` still says `ours: fall-detection`. That's why both namespaces exist — let me verify and fix.

---

## 1. Yes — only the custom images need pushing to FOCUS's registry

The split is between **public** images (already on Docker Hub) and **our** images (only on your laptop). FOCUS DevOps doesn't need our copies of public images — their cluster pulls those from Docker Hub directly.

| Image | Push to FOCUS registry? | Why |
| --- | --- | --- |
| `inference-server` | **yes** | our code |
| `fall-dashboard` | **yes** | our code |
| `infrastructure/mlflow/` (the custom MLflow image with psycopg2/boto3) | **yes** | we customised the base image |
| `ml_dashboard`, `server_health` | **yes** (when added to chart) | our code |
| `postgres:16-alpine` | no | already on Docker Hub |
| `eclipse-mosquitto:2` | no | already on Docker Hub |
| `minio/minio:latest` | no | already on Docker Hub |
| `prom/prometheus:latest` | no | already on Docker Hub |
| `grafana/grafana:10.4.0` | no | already on Docker Hub |
| `fd-mock-fhir`, `fd-mock-patient-dashboard` (mock-focus) | **never** | dev-only, simulates FOCUS, must not ship |

So 3 images we push today (4–5 once ml_dashboard / server_health are added to the chart). The rest FOCUS's cluster pulls from public registries on its own.

---

## 2. `docker build` makes the image; `docker push` ships it — two separate steps

Your understanding is **almost** right. `docker build` creates the image in your local Docker daemon's cache. To get it into a registry, you need a second step that tags it with the registry's address and pushes:

`# Step 1 — build (already done by the chart README's Section 4)
docker build -f inference_server/Dockerfile -t inference-server:latest .

# Step 2 — tag for the destination registry (FOCUS-supplied URL)
docker tag inference-server:latest registry.charite.de/mcs-fall-detection/inference-server:v1.0.0

# Step 3 — push
docker push registry.charite.de/mcs-fall-detection/inference-server:v1.0.0`

Why the extra `docker tag`? Docker images are addressed by `<registry>/<namespace>/<name>:<tag>`. When you build with just `-t inference-server:latest`, the image is implicitly tagged for "no registry, just locally" — that's why `pullPolicy: Never` is the only way K8s finds it. To push, the image needs a name that includes the destination registry, hence the `docker tag` step.

In CI the build/tag/push are typically one script. For local dev they stay separate so you don't accidentally push every test build.

---

## 3. Two namespaces — yes, this is from the values.yaml drift

You're right. Earlier we updated all the **docs** to say `mcs-fall-detection`, but the chart's actual `values.yaml` still says `ours: fall-detection`. That's why both namespaces exist — let me verify and fix.

### Will the legacy pods stop on their own?

**No.** They keep running until you tell K8s to stop them. Pod state is independent of helm releases — even if helm forgets about them, K8s still owns the namespace and the resources inside it.

### How to clean up

First, see what's actually in the legacy namespace:

`kubectl get all -n fall-detection`

You'll likely see a mix of leftover resources from earlier attempts. To delete **everything** in that namespace in one shot, just delete the namespace itself — K8s cascades the deletion to every resource inside it:

`kubectl delete namespace fall-detection`

This is safe **as long as that namespace doesn't have data you care about**. PVCs in there are deleted too, so any Postgres data / MinIO objects in `fall-detection` will be gone. Since this is a leftover dev namespace from before we standardised the name, that's almost certainly fine — but worth eyeballing the output of the `get all` first.

Verify after:

`kubectl get namespaces
# fall-detection should be gone within a few seconds
# mcs-fall-detection should remain`

### What going forward looks like

Now that `values.yaml` is fixed:

`# (Optional) tear down the current install
helm uninstall mcs-fall-detection -n mcs-fall-detection
kubectl delete namespace mcs-fall-detection

# Re-install — both namespace.yaml template AND --namespace flag now agree on mcs-fall-detection
helm install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m`

Only one namespace gets created this time. The chart's own `namespace.yaml` template is technically redundant now (helm's `--create-namespace` flag handles it), but it's harmless — they refer to the same namespace name and Helm just no-ops the duplicate.

If you want, I can remove the `templates/namespace.yaml` file entirely since `--create-namespace` is the standard way to do this — that would prevent this kind of drift from ever happening again. Say the word.

----

## 1. Why `mcs-fall-detection` still exists after uninstall

This is a quirk of `helm install --create-namespace`:

| Resource | Created by | Removed by `helm uninstall`? |
| --- | --- | --- |
| `fall-detection` namespace | Chart's `templates/namespace.yaml` (a tracked chart resource) | **yes** — that's why it's in `Terminating` |
| `mcs-fall-detection` namespace | `--create-namespace` flag (outside the release lifecycle) | **no** |
| Pods, Services, ConfigMaps, etc. | Chart templates (tracked) | **yes** — already gone |
| PVCs from StatefulSets | Created by Kubernetes when StatefulSet pods come up | **no** — survives uninstall by design |

So the asymmetry: **the chart's namespace.yaml template was deleted; the helm-flag namespace stayed.** This is the same drift issue from before — the chart's template referenced `fall-detection` (per the old values.yaml) so that's what got cleaned up. The `mcs-fall-detection` namespace persists with the PVCs inside it.

To fully clean up `mcs-fall-detection`:

`kubectl delete namespace mcs-fall-detection`

That deletes the namespace and **everything inside it including PVCs** — which leads us to your second question.

---

## 2. Stop pods without deleting (keep PVCs)

Yes, several options depending on how aggressively you want to "stop":

### Option A — `helm uninstall` keeps PVCs already

You've actually already done this for the right thing. **`helm uninstall` does not delete StatefulSet PVCs by default.** That's a deliberate StatefulSet design: the data outlives the workload. Check what's still there:

`kubectl get pvc -n mcs-fall-detection
# you should see: data-postgres-0, data-minio-0, etc. — all "Bound"`

So your Postgres data and MinIO objects are safe right now. As long as you DON'T run `kubectl delete namespace mcs-fall-detection`, the PVCs survive. To bring everything back:

`helm install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --wait --timeout 5m`

The new pods will re-attach to the existing PVCs (because StatefulSet PVCs are bound by name: `data-postgres-0` always attaches to `postgres-0`). Postgres comes up with all the data from your last run.

### Option B — Scale to zero (keep release, just pause workloads)

If you don't want to uninstall but just stop the running pods:

`# Stops every Deployment / StatefulSet by setting replicas to 0
kubectl scale deployment --all --replicas=0 -n mcs-fall-detection
kubectl scale statefulset --all --replicas=0 -n mcs-fall-detection`

The Deployments / StatefulSets still exist; their pod count goes to zero. PVCs stay attached. To resume:

`kubectl scale deployment --all --replicas=1 -n mcs-fall-detection
kubectl scale statefulset --all --replicas=1 -n mcs-fall-detection`

This is faster than uninstall + reinstall and preserves the helm release record.

### Quick comparison

| Want | Command | What's kept |
| --- | --- | --- |
| Stop pods, keep everything else | `kubectl scale ... --replicas=0` | Deployments, Services, PVCs, helm release |
| Uninstall workloads, keep data | `helm uninstall` (you've done this) | PVCs, namespace |
| Full clean wipe | `helm uninstall && kubectl delete namespace` | nothing |

Right now you're at the second row — your data is safe. If you want the pods back without losing data, just `helm install` again.

---

## 3. Is the namespace mismatch fixed?

**Yes**, in the source files. I updated `values.yaml` in my last response:

`namespaces:
  ours: mcs-fall-detection    # confirmed 2026-04-29; FOCUS DevOps may override`

So from the **next** install onwards, the chart's `namespace.yaml` template will create `mcs-fall-detection` (matching what `--namespace` says), and you won't see two namespaces anymore.

The current cluster state still has the leftover `mcs-fall-detection` namespace from before the fix (because `--create-namespace` made it). To start completely fresh:

`# 1. Wipe the leftover namespace (also deletes its PVCs)
kubectl delete namespace mcs-fall-detection

# 2. Re-install — values.yaml + --namespace flag now agree
helm install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m

# 3. Confirm there's only one namespace, not two
kubectl get namespaces | findstr -i fall
# expected: only "mcs-fall-detection   Active   <new age>"`

If `kubectl get pvc -n mcs-fall-detection` shows old PVCs you'd rather keep (you have inference history / model registry data you want preserved), just skip step 1 and reinstall on top — the existing PVCs will be reused.

---

## Optional: prevent this from happening again

The chart contains a `templates/namespace.yaml` that creates the namespace. But `--create-namespace` does the same thing. Having both is what caused the drift to be visible — when values.yaml said `fall-detection` and the flag said `mcs-fall-detection`, two namespaces appeared.

The cleaner pattern is: **delete the chart's namespace.yaml template entirely** and rely on `--create-namespace` only. Then `values.yaml` → `namespaces.ours` becomes a setting that templates use to *target* the namespace (e.g., for NetworkPolicies that reference it), not to *create* it.

Want me to remove `helm/fall-detection/templates/namespace.yaml`? It's a 5-second edit and prevents this confusion forever. Alternatively we leave it alone — the values.yaml fix is enough now that the names match.

## The four system namespaces — DO NOT TOUCH

These are Kubernetes built-ins. Every cluster has them. They're not yours to manage:

| Namespace | Purpose |
| --- | --- |
| **`default`** | Where resources go if you don't specify `--namespace` and don't have a namespace set as your context default. Best practice: don't put your own resources here — make a real namespace. |
| **`kube-system`** | The K8s control plane: API server, scheduler, controller-manager, CoreDNS (the cluster's DNS resolver), kube-proxy. **Do not touch.** Breaking this breaks the cluster. |
| **`kube-public`** | World-readable namespace, even by unauthenticated users. Holds cluster-info ConfigMaps that bootstrapping nodes need. Almost empty. |
| **`kube-node-lease`** | Lightweight heartbeats. Each kubelet writes a `Lease` object here every few seconds so the control plane knows the node is alive. Internal mechanism. |

Curiosity check: `kubectl get pods -n kube-system` — you'll see ~10–15 pods, all part of K8s itself (`coredns-...`, `etcd-docker-desktop`, `kube-apiserver-docker-desktop`, etc.). That's how Kubernetes is built — it runs itself as pods inside its own first namespace.

So your `kubectl get namespaces` output is **completely clean and normal**: 4 system namespaces and nothing else. Once you do the build → install → check loop above, you'll see `mcs-fall-detection` appear as the 5th.

## Order of docker k8s operation: build images **first**, then helm install

Build comes first. Why:

`helm install
   ▼
K8s scheduler creates pods
   ▼
each pod tries to start its container
   ▼
container runtime looks for the image
   │
   ├─ pullPolicy: Always → pulls from registry
   └─ pullPolicy: Never → looks in local Docker cache only
                          │
                          └─ if not found → ImagePullBackOff (pod stays Pending forever)`

Since the chart uses `pullPolicy: Never`, the images **must already be in the local Docker cache** before `helm install` runs. Otherwise the pods just sit in `ImagePullBackOff` until you manually build the images (and then they self-recover within a minute).

So the right order:

`# 1. Build (puts images in local Docker cache)
docker build -f inference_server/Dockerfile -t inference-server:latest .
docker build -f fall_dashboard/Dockerfile  -t fall-dashboard:latest  .

# 2. Verify they're there
docker images | findstr -E "inference-server|fall-dashboard"
# expect 2 lines

# 3. Now install — pods can find their images immediately
helm install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m`

If you forget step 1, step 3 hangs at "waiting for pods" because the image-less pods never become Ready. The recovery is just `docker build`-ing then — K8s retries automatically every ~30s.

**Side note on Docker Compose Postgres:** you don't need to stop it. The K8s Postgres lives at `postgres.mcs-fall-detection.svc.cluster.local:5432` (in-cluster only), while Docker Compose's binds to host `localhost:5432`. They don't conflict. But if you accidentally point your Python venv (`.env` → `DATABASE_URL`) at one while running services that wrote to the other, you'll see "where's my data?" confusion. Pick one path at a time.

----

## requirement,txt installed or required?

Audit complete — the answer is good news: **only ml-dashboard had the gap**. Here's the summary:

| Service | Imports shared_db? | Imports ml_pipeline? | Spawns subprocess? | Status |
| --- | --- | --- | --- | --- |
| `inference-server` | yes (sqlalchemy) | yes (full pipeline → numpy/pandas/scipy/xgboost) | no | ✅ all deps in requirements |
| `fall-dashboard` | yes (sqlalchemy) | yes — but only `influx_client_manager` + `data_converter` (need numpy, pandas, influxdb-client; **scipy NOT needed**) | no | ✅ all deps in requirements |
| `ml-dashboard` | no | no | yes — `python -m retrain.retrain` (xgboost, sklearn, pandas, numpy, sqlalchemy, boto3) | ⚠️ fixed earlier this turn |
| `server-health` | no | no | no — only stdlib + httpx + sqlalchemy | ✅ all deps in requirements |

The reason fall-dashboard is OK despite copying the whole `ml_pipeline/` folder: it only touches `influx_client_manager` and `data_converter`, which need `numpy`, `pandas`, and `influxdb-client` — all already in `fall_dashboard/requirements.txt`. The heavier modules (`acc_resampler` needs scipy, `inference_engine` needs xgboost) are never imported by fall_dashboard, so Python never tries to load scipy/xgboost.

So no further action needed — your previous `helm upgrade` + ml-dashboard rebuild closes the only gap.



----

**Where the URL `http://mlflow:5000` is defined:**

templates/configmap.yaml:16 — our Helm chart sets:

`MLFLOW_TRACKING_URI: "http://mlflow:{{ .Values.mlflow.port }}"
# → http://mlflow:5000 after templating`

**Who wants to access it:**

The retrain subprocess inside the **ml-dashboard pod**. Chain:

1. User clicks "Retrain" in browser → `POST /api/retrain` to ml-dashboard
2. ml-dashboard's `web.py` spawns subprocess: `python -m retrain.retrain ...`
3. The subprocess inherits ml-dashboard's env, including `MLFLOW_TRACKING_URI=http://mlflow:5000` (loaded into the pod by `envFrom: configMapRef: fall-detection-config`)
4. retrain.py:175: `os.getenv("MLFLOW_TRACKING_URI", "./mlruns")` reads it
5. `MlflowClient(tracking_uri)` is constructed with that URL
6. Client calls `client.get_experiment_by_name(...)` → underneath it makes an HTTP request:
    
    `GET http://mlflow:5000/api/2.0/mlflow/experiments/get-by-name?...`
    

**Where `Host: mlflow:5000` comes from:**

You don't see it set anywhere because the **HTTP client library auto-populates it from the URL**. This is HTTP/1.1 spec — the `Host:` header carries the `host:port` part of the URL the client requested. In Python's `requests`/`urllib3`, when you do `requests.get("http://mlflow:5000/...")`, the library splits the URL and the resulting wire request looks like:

`GET /api/2.0/mlflow/experiments/get-by-name?... HTTP/1.1
Host: mlflow:5000
User-Agent: python-requests/...
...`

So:

- `http://mlflow:5000` (the URL) — set by US in the configmap, lives in the env var, used by the MLflow client to construct the URL.
- `Host: mlflow:5000` (the HTTP header) — set automatically by the HTTP client from the URL's authority component. We never write this directly.

**Who reads `Host: mlflow:5000`:**

The **MLflow tracking server** (gunicorn process inside the `mlflow` pod). MLflow 3.x has DNS-rebinding-protection middleware that compares `Host:` against an allowlist. The default allowlist is just localhost variants — so a Host of `mlflow:5000` (which is what every cross-pod call uses inside the cluster) gets rejected with 403. That's why we add `--allowed-hosts "mlflow:*,..."` to the server's launch flags.

**Summary diagram:**

`configmap.yaml
   MLFLOW_TRACKING_URI=http://mlflow:5000
            │
            ▼ envFrom
[ ml-dashboard pod ]
   $MLFLOW_TRACKING_URI = http://mlflow:5000
            │
            ▼ spawns subprocess (inherits env)
   retrain.py → MlflowClient("http://mlflow:5000")
            │
            ▼ HTTP/1.1 client splits URL
   GET /api/2.0/... HTTP/1.1
   Host: mlflow:5000          ← auto-populated by urllib3
            │
            ▼ K8s DNS resolves "mlflow" to a service ClusterIP
[ mlflow pod ]
   gunicorn → middleware checks Host header
              "mlflow:5000" not in allowlist → 403`