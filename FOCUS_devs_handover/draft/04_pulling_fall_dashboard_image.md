# Pulling the Fall Dashboard Image

## Overview

The `fall-dashboard` service runs as a Docker image built and maintained by MCS.
FOCUS DevOps does not build this image — you only pull it from the MCS private registry.

| Item | Value |
|------|-------|
| Registry | `registry-smarko-health.de` |
| Image | `registry-smarko-health.de/fall-detection/fall-dashboard:latest` |
| Pull secret name | `mcs-labs` |
| Who pushes new versions | Mohammed (MCS) |

The `mosquitto` broker uses the public `eclipse-mosquitto:2` image from Docker Hub —
no pull secret or special setup needed for that one.

---

## Step 1 — Get registry credentials from Mohammed

Before you can pull, Mohammed must give you:

- Registry username
- Registry password

Do not proceed until you have both. Mohammed pushes the image to the registry first,
then shares the credentials with you.

---

## Step 2 — Create the pull secret in the cluster

Run once. Creates a Kubernetes Secret that allows pods in the `fall-dashboard`
namespace to pull from `registry-smarko-health.de`.

```bash
kubectl create secret docker-registry mcs-labs \
    --docker-server=registry-smarko-health.de \
    --docker-username=<username from Mohammed> \
    --docker-password=<password from Mohammed> \
    --namespace fall-dashboard
```

If the namespace does not exist yet, create it first:

```bash
kubectl create namespace fall-dashboard
```

Verify the secret was created:

```bash
kubectl get secret mcs-labs -n fall-dashboard
# Expected:
#   NAME       TYPE                             DATA
#   mcs-labs   kubernetes.io/dockerconfigjson   1
```

---

## Step 3 — Confirm values_production.yaml references the secret

Open `helm/values_production.yaml` and confirm these two lines are set:

```yaml
imagePullSecret: mcs-labs
fallDashboard.image: registry-smarko-health.de/fall-detection/fall-dashboard:latest
```

Both are already set to the correct values in the template. Do not change them.

---

## Step 4 — The pull happens automatically on helm install

When you run `bash helm/install.sh`, K3s pulls the image from the registry using
the `mcs-labs` secret. You do not need to run `docker pull` manually.

To confirm the image was pulled successfully after install:

```bash
kubectl describe pod -n fall-dashboard -l app=fall-dashboard | grep -A5 "Events:"
# Look for: Successfully pulled image
# Bad sign: ErrImagePull or ImagePullBackOff
```

Or check pod status directly:

```bash
kubectl get pods -n fall-dashboard
# fall-dashboard pod must show: READY 1/1, STATUS Running
```

---

## Troubleshooting pull failures

### ErrImagePull / ImagePullBackOff

The pod cannot pull the image. Most common causes:

**Wrong credentials:**
```bash
# Delete the existing secret and recreate with the correct credentials
kubectl delete secret mcs-labs -n fall-dashboard
kubectl create secret docker-registry mcs-labs \
    --docker-server=registry-smarko-health.de \
    --docker-username=<correct username> \
    --docker-password=<correct password> \
    --namespace fall-dashboard

# Restart the pod to retry the pull
kubectl rollout restart deployment/fall-dashboard -n fall-dashboard
```

**Image not pushed yet:**
Contact Mohammed to confirm the image exists on the registry before retrying.

**Registry unreachable from your cluster:**
```bash
# Test connectivity to the registry from inside the cluster
kubectl run reg-test --rm -it --restart=Never \
    --image=curlimages/curl -- \
    curl -I https://registry-smarko-health.de/v2/
# Expected: HTTP 200 or 401 (401 = registry reachable, auth required)
# If connection refused or timeout: network/firewall issue between cluster and registry
```

### Updating to a new image version

When Mohammed pushes an updated image, run:

```bash
bash helm/install.sh
# helm upgrade --install with imagePullPolicy: Always forces a fresh pull
```

No values change is needed — `imagePullPolicy: Always` is already set in
`values_production.yaml`, so every `helm upgrade` pulls the latest image.
