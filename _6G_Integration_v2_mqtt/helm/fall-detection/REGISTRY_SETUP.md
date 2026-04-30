# GitLab Container Registry — Setup & Secure Handover to FOCUS DevOps

**Audience:** image owner (Hayate / MCS) AND FOCUS DevOps team.
**Goal:** publish the five custom container images to a private GitLab Container Registry, then give FOCUS DevOps **read-only** pull access to that registry from their Kubernetes cluster — without ever sharing a long-lived personal credential.

Read this end-to-end once. Then act on the section that applies to you (Hayate vs. FOCUS DevOps).

---

## TL;DR — what each side does

| Step | Who | What |
|------|-----|------|
| 1    | MCS  | Create / pick a GitLab project and enable the Container Registry. |
| 2    | MCS  | Authenticate the local Docker daemon and push the five images. |
| 3    | MCS  | Generate a **Deploy Token** (scope: `read_registry` only). |
| 4    | MCS  | Send the deploy-token username + password to FOCUS DevOps via a secure channel (1Password share / one-time-secret link / GPG). Never email/Slack/paste-in-Teams plain. |
| 5    | DevOps | Create a `docker-registry` secret in the cluster from those credentials. |
| 6    | DevOps | Update `values.yaml` (`registry`, `images.pullPolicy`) and patch the chart so every Deployment references the secret. |
| 7    | Both | Verify with one `helm upgrade` — pods should pull cleanly. |
| 8    | MCS  | Rotate the deploy token every 6–12 months or on FOCUS staff turnover. |

---

## 1. Background: what we're publishing

Five custom images are built from this repo:

| Image | Source Dockerfile |
|-------|-------------------|
| `inference-server`   | [inference_server/Dockerfile](../../inference_server/Dockerfile) |
| `fall-dashboard`     | [fall_dashboard/Dockerfile](../../fall_dashboard/Dockerfile)   |
| `ml-dashboard`       | [ml_dashboard/Dockerfile](../../ml_dashboard/Dockerfile)       |
| `server-health`      | [server_health/Dockerfile](../../server_health/Dockerfile)     |
| `mlflow` (custom)    | [infrastructure/mlflow/Dockerfile](../../infrastructure/mlflow/Dockerfile) |

Everything else (`postgres`, `mqtt-broker`, `minio`, `prometheus`, `grafana`) is a stock public image — no registry work needed for those. The mock images under `helm/mock-focus/` are local-test-only and **should not be pushed to the production registry**.

---

## 2. (MCS) Set up the GitLab project & registry

### 2.1 Create or pick a project

GitLab Container Registry is enabled per-project. You can use:

- **GitLab.com SaaS** — free, registry URL is `registry.gitlab.com`
- **A self-hosted GitLab instance** (if MCS or FOCUS already runs one) — registry URL is whatever the admin configured (e.g., `registry.gitlab.mcs.com`)

For this doc the working assumption is GitLab.com SaaS under a group named `mcs` and a project named `fall-detection`. Replace `mcs/fall-detection` with whatever you actually use.

Steps in the GitLab UI:

1. New project → blank project → name **`fall-detection`** → group **`mcs`**.
2. Visibility: **Private** (this is the whole point — non-public registry).
3. After creation: **Settings → Packages and registries → Container Registry → Enabled**. (Default-on for new projects on GitLab.com; verify it's not disabled.)

Your image addresses will then be:

```
registry.gitlab.com/mcs/fall-detection/inference-server:latest
registry.gitlab.com/mcs/fall-detection/fall-dashboard:latest
registry.gitlab.com/mcs/fall-detection/ml-dashboard:latest
registry.gitlab.com/mcs/fall-detection/server-health:latest
registry.gitlab.com/mcs/fall-detection/mlflow:latest
```

### 2.2 Verify the registry endpoint

In the GitLab UI: **Project → Deploy → Container Registry**. The page header shows the exact base URL. Use that — don't guess.

---

## 3. (MCS) Push images from your laptop

### 3.1 Generate a Personal Access Token (one-time, for *your* push)

GitLab profile → **Edit profile → Access tokens → Add new token**.

- **Name:** `local-push-fall-detection`
- **Expires:** 30 days (this is for *you*, not FOCUS — short is fine)
- **Scopes:** `read_registry`, `write_registry` ← both needed to push
- Click **Create**, copy the token immediately (shown once).

This is **your** credential. Do not share it with FOCUS — they get a separate, narrower deploy token in section 4.

### 3.2 Log in (PowerShell)

```powershell
docker login registry.gitlab.com -u <your-gitlab-username> -p <pat-from-3.1>
```

Successful output: `Login Succeeded`. The credential is stored in `%USERPROFILE%\.docker\config.json` — protect that file.

### 3.3 Build, tag, push

A reusable PowerShell snippet — run from `_6G_Integration_v2_mqtt/`:

```powershell
$REGISTRY = "registry.gitlab.com/mcs/fall-detection"
$TAG      = "latest"   # or a semver / git-sha for production releases (recommended)

# Build (only if you haven't already)
docker build -f inference_server/Dockerfile -t inference-server:$TAG .
docker build -f fall_dashboard/Dockerfile   -t fall-dashboard:$TAG  .
docker build -f ml_dashboard/Dockerfile     -t ml-dashboard:$TAG    .
docker build -f server_health/Dockerfile    -t server-health:$TAG   .
docker build -f infrastructure/mlflow/Dockerfile -t mlflow:$TAG     infrastructure/mlflow

# Tag for the registry
docker tag inference-server:$TAG  "$REGISTRY/inference-server:$TAG"
docker tag fall-dashboard:$TAG    "$REGISTRY/fall-dashboard:$TAG"
docker tag ml-dashboard:$TAG      "$REGISTRY/ml-dashboard:$TAG"
docker tag server-health:$TAG     "$REGISTRY/server-health:$TAG"
docker tag mlflow:$TAG            "$REGISTRY/mlflow:$TAG"

# Push
docker push "$REGISTRY/inference-server:$TAG"
docker push "$REGISTRY/fall-dashboard:$TAG"
docker push "$REGISTRY/ml-dashboard:$TAG"
docker push "$REGISTRY/server-health:$TAG"
docker push "$REGISTRY/mlflow:$TAG"
```

> **Recommendation: stop using `latest` for production.** Tag with the git short-sha (`docker tag inference-server $REGISTRY/inference-server:$(git rev-parse --short HEAD)`). Then a deploy is reproducible and a rollback is `helm upgrade --set images.inferenceServer.tag=<old-sha>`. `latest` makes "what's actually running?" unanswerable.

### 3.4 Sanity-check after push

GitLab UI → **Deploy → Container Registry** → you should see all five repositories with the tag you just pushed. Click into one to confirm size and digest are sensible.

---

## 4. (MCS) Generate the credential FOCUS DevOps will use

This is the "secret I mentioned" — the credential the FOCUS K8s cluster needs to pull these private images. **Do not give FOCUS your personal access token from 3.1.** Issue a separate, scoped, revocable credential.

### 4.1 Pick the right credential type

GitLab offers three options. Use **Deploy Token**.

| Type | Pros | Cons | Verdict |
|------|------|------|---------|
| **Deploy Token** | Project-scoped, read-only by default, you can revoke without affecting users, dedicated username/password | Manual rotation | ✅ Use this |
| Project Access Token | Acts as a bot user, supports scopes | Requires GitLab Premium tier | Only if you already have Premium |
| Personal Access Token (yours) | Quickest | Tied to **you** — leaving the org / changing roles invalidates production. NEVER do this. | ❌ Never share |

### 4.2 Create the Deploy Token

In the GitLab project: **Settings → Repository → Deploy tokens → Add token**.

- **Name:** `focus-cluster-imagepull`
- **Username:** leave default (`gitlab+deploy-token-<n>`) or set something memorable like `focus-imagepull`
- **Expires at:** **set a date** — recommend 6 or 12 months from today. (Calendar-reminder this date now; expired tokens silently start failing pulls.)
- **Scopes:** ✅ `read_registry` only. (Do **not** check `read_repository`, `write_registry`, `write_repository`, `read_package_registry`, `write_package_registry`. FOCUS only needs to pull container images.)
- Click **Create deploy token**.

GitLab will **show the token password exactly once**. Copy both values:

```
Username: focus-imagepull         (or gitlab+deploy-token-<n>)
Password: <long random string>
```

If you close this page without copying, you must delete the token and create a new one — there is no "show again" button.

---

## 5. (MCS) Send the credential to FOCUS DevOps securely

The deploy token password is functionally a password to your private registry. Treat it like one.

### Acceptable channels (pick one)

| Method | Why it's good |
|--------|---------------|
| **1Password / Bitwarden / Keeper "shared item"** | End-to-end encrypted, audit log, easy to revoke |
| **One-time-secret link** (e.g., `password.link`, `onetimesecret.com`) | Self-destructs after first view; if the recipient sees it has already been viewed, you both know it was intercepted |
| **GPG-encrypted file** to FOCUS DevOps's published public key | Works without any third-party SaaS |
| **In-person / phone call** | Fine for the password if you hand over the username separately |

### Do NOT use

- ❌ Email body or attachment in plaintext
- ❌ Slack / Teams / WhatsApp DM (logged, often archived, replicated to vendor servers)
- ❌ Repository commits, even private ones (logged in CI, shows in `git log` forever)
- ❌ Shared cloud documents (Google Doc / SharePoint) without per-person ACL
- ❌ Screenshots in a ticket/Jira

### Defense-in-depth tip

Send the **username** by one channel (email is fine) and the **password** by a second channel (1Password share). An attacker would have to compromise both to use the credential.

### What the message to FOCUS should contain

```
Subject: Pull credentials for fall-detection container registry

Registry server : registry.gitlab.com
Username        : focus-imagepull
Password        : <use one-time-secret link below>
                  https://onetimesecret.com/secret/<id>

Project URL     : https://gitlab.com/mcs/fall-detection
Scope           : read_registry only (cannot push, cannot read source)
Expires         : 2026-10-29 (12 months) — we will rotate before this date

Images to pull  :
  registry.gitlab.com/mcs/fall-detection/inference-server:<tag>
  registry.gitlab.com/mcs/fall-detection/fall-dashboard:<tag>
  registry.gitlab.com/mcs/fall-detection/ml-dashboard:<tag>
  registry.gitlab.com/mcs/fall-detection/server-health:<tag>
  registry.gitlab.com/mcs/fall-detection/mlflow:<tag>

To use these in your cluster, see REGISTRY_SETUP.md section 6.
```

---

## 6. (FOCUS DevOps) Wire the credential into the cluster

### 6.1 Create the docker-registry secret

```bash
kubectl create secret docker-registry gitlab-registry-creds \
  --namespace mcs-fall-detection \
  --docker-server=registry.gitlab.com \
  --docker-username='focus-imagepull' \
  --docker-password='<paste-deploy-token-password>' \
  --docker-email='devops@focus.example.com'    # any address; not used by GitLab
```

> Run this **once per namespace**. K8s secrets are namespace-scoped; if you ever deploy this chart to a second namespace, repeat the command for that namespace.

Verify:

```bash
kubectl get secret gitlab-registry-creds -n mcs-fall-detection -o jsonpath='{.type}'
# expected: kubernetes.io/dockerconfigjson
```

### 6.2 Update `values.yaml`

```yaml
registry: registry.gitlab.com/mcs/fall-detection

images:
  pullPolicy: Always           # was Never (local-build mode); flip for real registry
  inferenceServer: { repository: inference-server, tag: <git-sha-or-version> }
  fallDashboard:   { repository: fall-dashboard,   tag: <git-sha-or-version> }
  mlDashboard:     { repository: ml-dashboard,     tag: <git-sha-or-version> }
  serverHealth:    { repository: server-health,    tag: <git-sha-or-version> }
  # add mlflow.repository / mlflow.tag if you also publish a custom mlflow image

imagePullSecrets:
  - name: gitlab-registry-creds   # NEW — must match secret name from 6.1
```

### 6.3 Patch the chart templates (one-time)

The chart in this repo today does **not** reference `imagePullSecrets` — the templates were written for the local Docker Desktop case where `pullPolicy: Never` makes auth irrelevant. Add the following snippet **once per Deployment template** (and the migrate-job template), inside the `spec.template.spec` block, at the same indent level as `containers:`:

```yaml
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 8 }}
{{- end }}
```

Files to edit:

```
templates/inference-server/deployment.yaml
templates/fall-dashboard/deployment.yaml
templates/ml-dashboard/deployment.yaml
templates/server-health/deployment.yaml
templates/mlflow/deployment.yaml
templates/migrate-job.yaml
```

(Skip stock-image deployments — `postgres`, `mqtt-broker`, `minio`, `prometheus`, `grafana`, `mock-*` — those pull from public Docker Hub and don't need the secret.)

The `with`-guard means it's a no-op when `imagePullSecrets` is empty — local-dev `pullPolicy: Never` still works after the patch.

### 6.4 Apply

```powershell
helm upgrade mcs-fall-detection .\helm\fall-detection -n mcs-fall-detection --wait --timeout 5m
kubectl get pods -n mcs-fall-detection -w
```

Expected: pods recreate, briefly show `ContainerCreating` while the image is pulled, then `Running`. If you see `ImagePullBackOff` jump to section 8.

---

## 7. (Optional) Auto-build & push from GitLab CI

If you want the registry to update on every push to a branch, add `.gitlab-ci.yml` at the repo root:

```yaml
build_and_push:
  image: docker:24
  services: [docker:24-dind]
  stage: build
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - |
      for svc in inference_server fall_dashboard ml_dashboard server_health; do
        name=$(echo $svc | tr '_' '-')
        docker build -f $svc/Dockerfile -t $CI_REGISTRY_IMAGE/$name:$CI_COMMIT_SHORT_SHA .
        docker push $CI_REGISTRY_IMAGE/$name:$CI_COMMIT_SHORT_SHA
      done
  only: [main, 6G_intergation_with_MQTT]
```

`CI_REGISTRY_USER`, `CI_REGISTRY_PASSWORD`, `CI_REGISTRY`, `CI_REGISTRY_IMAGE`, `CI_COMMIT_SHORT_SHA` are all **predefined** by GitLab CI — no secrets to configure. The job tags every image with the commit SHA, which is exactly what you want for reproducible deploys.

This doesn't change anything for FOCUS — they still pull via the deploy token from section 4.

---

## 8. Troubleshooting

### `ImagePullBackOff` / `ErrImagePull` after deploy

```bash
kubectl describe pod <pod-name> -n mcs-fall-detection | Select-String -Pattern "Failed|Error" -Context 0,3
```

Common causes:

| Symptom in `Events`                                     | Cause                                          | Fix |
|---------------------------------------------------------|------------------------------------------------|-----|
| `unauthorized: HTTP Basic: Access denied`               | Wrong token or token revoked                   | Recreate secret in 6.1 with correct password |
| `denied: requested access to the resource is denied`    | Token scope missing `read_registry`            | Recreate deploy token (section 4.2) |
| `manifest unknown` / `not found`                        | Image tag doesn't exist in registry            | Check the tag in GitLab UI; rebuild + push   |
| `no such host` for `registry.gitlab.com`                | Cluster has no outbound DNS / egress           | Check FOCUS cluster network policy           |
| `x509: certificate signed by unknown authority`         | Self-hosted GitLab without trusted CA in nodes | Install GitLab CA on each node, or use SaaS  |

### "It worked yesterday but pulls fail today"

Check **token expiry** in GitLab UI → Settings → Repository → Deploy tokens. Expired tokens silently 401. Recreate (section 4.2), update the K8s secret (`kubectl delete secret gitlab-registry-creds -n mcs-fall-detection && kubectl create secret docker-registry ...`), and `kubectl rollout restart deploy -n mcs-fall-detection` to force a re-pull.

### Verifying the secret is wired correctly

```bash
kubectl get pod <pod-name> -n mcs-fall-detection -o yaml | Select-String "imagePullSecrets" -Context 0,2
```

Should print the `name: gitlab-registry-creds` line. If it doesn't, the chart patch from 6.3 didn't reach this Deployment — re-run `helm upgrade` and re-check.

---

## 9. Rotation & revocation

### Routine rotation (every 6–12 months)

1. Create a new deploy token in GitLab (section 4.2). Different name, e.g. `focus-cluster-imagepull-2027`.
2. Send to FOCUS DevOps via secure channel (section 5).
3. FOCUS recreates the K8s secret with new credentials, then `kubectl rollout restart deploy -n mcs-fall-detection`.
4. Confirm pods running with new pull (events show recent `Successfully pulled image`).
5. Only then: revoke the old token in GitLab.

### Emergency revocation (suspected leak)

GitLab → Settings → Repository → Deploy tokens → **Revoke**. Effective immediately — any in-flight pull from that token starts 401-ing. New pulls will fail until FOCUS DevOps gets a replacement and updates the K8s secret. Plan to have the replacement ready before clicking revoke unless the leak is severe enough to justify the downtime.

### When FOCUS DevOps staff change

The deploy token is a service credential, not a user credential — it does **not** need to be reissued when their personnel change, because no individual person knows it (it lives only in 1Password and the K8s secret). Rotate only on a fixed schedule or on suspected leak.

---

## 10. Security checklist

Use this before declaring the registry "ready":

- [ ] GitLab project visibility = **Private**
- [ ] Container Registry enabled
- [ ] Pushed images visible only to logged-in members + deploy tokens (test in incognito → `docker pull` should 401)
- [ ] Deploy token created with `read_registry` only
- [ ] Deploy token expiry **set** (not "never")
- [ ] Calendar reminder for rotation set on Hayate's calendar
- [ ] Credentials sent to FOCUS via 1Password / one-time-secret / GPG (not email/Slack)
- [ ] K8s secret `gitlab-registry-creds` exists in `mcs-fall-detection` namespace
- [ ] All five custom-image Deployments + migrate-job reference `imagePullSecrets`
- [ ] `values.yaml` `images.pullPolicy: Always` for production (not `Never`)
- [ ] Image tags are immutable (git-sha or semver, not `latest`)
- [ ] First `helm upgrade` after switch produces all pods Running

---

## Appendix — quick reference

### Server URL
- SaaS: `registry.gitlab.com`
- Self-hosted: same hostname as GitLab UI, optionally with a port

### Image address pattern
```
<server>/<group>/<project>/<image>:<tag>
```

### Deploy token creation path (UI)
`Project → Settings → Repository → Deploy tokens → Add token`

### Deploy token scope for image pulls
`read_registry` (and only that)

### K8s secret type
`kubernetes.io/dockerconfigjson` (created via `kubectl create secret docker-registry`)

### Required helm values
```yaml
registry: <server>/<group>/<project>
images:
  pullPolicy: Always
imagePullSecrets:
  - name: gitlab-registry-creds
```
