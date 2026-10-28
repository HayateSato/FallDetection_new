# Fall Dashboard Image Versioning and Rollback

Applies to the fall-dashboard on the Hetzner K3s test cluster (`5.75.255.114`).
Started 2026-10-21.

---

## 1. How versions are identified

| Term | Meaning |
|---|---|
| **Tag** (`:latest`, `:animation-2026-10-21`) | A movable label. Pushing a new image under the same tag moves the label; the old image is still in the registry. |
| **Digest** (`@sha256:...`) | The fixed fingerprint of one exact image. Never changes. The safest reference for rollback. |

The deployment runs `registry.smarko-health.de/fall-dashboard:latest` with `imagePullPolicy: Always`,
so every `kubectl rollout restart` pulls whatever `:latest` currently points to.

**Rule:** before pushing a new `:latest`, give the currently running image its own named tag
(section 4). Untagged images can be deleted if the registry is ever garbage-collected.

---

## 2. Version history

| Tag | Digest | Deployed | Contents |
|---|---|---|---|
| (untagged) | `sha256:04f3da3b28042f53f13f496ebc7338afcc058ce3e3f1ca4a68568ecccb038b34` | before 2026-10-21 | KPI ack logger (`fall/ack`); possible-fall notice stays until the card is clicked |
| `animation-2026-10-21` | `sha256:f6b205f942f2f14c02878b17c94592e2c2539e2e90b0e5c7f067208bb0d253a2` | 2026-10-21 | Card resolves when the patient answers: green "Patient reported no fall" / "Fall confirmed, patient is okay" (fades after 8 s); red "Patient did not respond" for no answer; possible-fall notice auto-clears after 60 s |
| `influx-write-2026-10-21` | `sha256:f05d4008b82a98bc02b3982db6cbe23a72bda1d6ca23461af5796da25c2ee007` | 2026-10-21 | Dashboard writes `fall_events` to its own InfluxDB on every `fall/alert` when `FALL_EVENTS_WRITE=true` (set on the Hetzner deployment); fall count dedupes by `observation_id`. Lets counts work while the mobile app writes to FOKUS's InfluxDB |

Add a row every time a new image is deployed.

### TODO: replace the `influx-write-2026-10-21` workaround with a proper fix

`FALL_EVENTS_WRITE=true` was a same-day workaround for the 2026-10-21 demo: the mobile app wrote
`fall_events` to FOKUS's InfluxDB and could not be changed, so the dashboard now writes its own
copy to the Hetzner InfluxDB. The same fall can end up in two databases. Before this goes further:

- [ ] **Decide who owns `fall_events` writes**: the mobile app (original design, `todos_isa.md`
      step 7) or the fall-dashboard (independent of the app's InfluxDB setting). Exactly one of
      them should write.
- [ ] If the dashboard owns it: remove the `fall_events` write from the mobile app, make
      `FALL_EVENTS_WRITE` the default (or drop the flag), and update `todos_isa.md` and the FOKUS
      handover docs.
- [ ] If the app owns it: set `FALL_EVENTS_WRITE=false` on Hetzner
      (`kubectl set env deployment/fall-dashboard -n fall-dashboard FALL_EVENTS_WRITE=false` and
      `writeFallEvents: false` in `values_hetzner_test.yaml`), and point the dashboard at the same
      InfluxDB the app writes to.
- [ ] Get a dedicated write token for the dashboard, scoped to the `fall_events` bucket only,
      instead of the shared token in `values_hetzner_test.yaml` (which is in git and has write
      access to all buckets). Move it to `--set` or a K8s secret and rotate the current one.
- [ ] Decide whether `simulated-fall` events should count toward the fall badges (they do now;
      the history list excludes them).
- [ ] Note: the dashboard only records a fall if the `fall/alert` reaches the broker. If the app
      fails before publishing, nothing is written.

---

## 3. Check what is running now

In the Portainer kubectl shell:

```bash
kubectl get pods -n fall-dashboard -l app=fall-dashboard -o jsonpath='{.items[*].status.containerStatuses[0].imageID}'
```

Compare the `sha256:` at the end with the table above. Only **one** pod should be listed. Two pods
both connect to MQTT as `fall-detection-caregiver` and kick each other off the broker.

---

## 4. Deploy a new version

On the laptop (PowerShell, Docker Desktop running), from `_6G_integration_v3_k3s/`:

```powershell
# 1. Preserve the current image under its own tag (rollback point)
docker tag registry.smarko-health.de/fall-dashboard:latest registry.smarko-health.de/fall-dashboard:<name-of-current-version>
docker push registry.smarko-health.de/fall-dashboard:<name-of-current-version>

# 2. Build and push the new version
docker build -f fall_dashboard/Dockerfile -t registry.smarko-health.de/fall-dashboard:latest .
docker push registry.smarko-health.de/fall-dashboard:latest

# 3. Optionally tag the new version too, so it has a name from day one
docker tag registry.smarko-health.de/fall-dashboard:latest registry.smarko-health.de/fall-dashboard:<name-of-new-version>
docker push registry.smarko-health.de/fall-dashboard:<name-of-new-version>
```

Step 1 only works if the local `:latest` is still the image that is running. Check with
`docker image inspect registry.smarko-health.de/fall-dashboard:latest --format '{{json .RepoDigests}}'`
and compare against section 3.

In the Portainer kubectl shell:

```bash
kubectl rollout restart deployment/fall-dashboard -n fall-dashboard
kubectl rollout status deployment/fall-dashboard -n fall-dashboard
```

Then run the check in section 3, reload the dashboard in the browser with **Ctrl+F5**, and add a
row to the table in section 2.

**Expected during every rollout:** for about 45 seconds the old and new pod both connect to MQTT as
`fall-detection-caregiver` and kick each other off, so the new pod logs `MQTT connected` every ~2 s.
It stops on its own once the old pod is gone. Do not deploy during a demo. To confirm it has stopped
(the timestamp should not move on repeated runs):

```bash
kubectl logs deployment/fall-dashboard -n fall-dashboard | grep -i "mqtt connected" | tail -1
```

If it keeps advancing, another client uses the same ID. Find it in the broker log:
`kubectl logs deployment/mosquitto -n fall-dashboard --since=2m | grep -E "fall-detection-caregiver|already connected"`.

---

## 5. Roll back

In the Portainer kubectl shell, using a tag:

```bash
kubectl set image deployment/fall-dashboard -n fall-dashboard \
  fall-dashboard=registry.smarko-health.de/fall-dashboard:animation-2026-10-21
kubectl rollout status deployment/fall-dashboard -n fall-dashboard
```

Or using a digest (works even for untagged versions, e.g. the pre-2026-10-21 image):

```bash
kubectl set image deployment/fall-dashboard -n fall-dashboard \
  fall-dashboard=registry.smarko-health.de/fall-dashboard@sha256:f6b205f942f2f14c02878b17c94592e2c2539e2e90b0e5c7f067208bb0d253a2
```

Takes about a minute. Then run the check in section 3 and reload the browser with **Ctrl+F5**.

### Rollback gotchas

- **Do not use `kubectl rollout undo`.** All revisions reference `:latest` with pull policy
  `Always`, so undo pulls `:latest` again, which is still the new image.
- **After a rollback the deployment is pinned** to the tag/digest you set. A later
  `kubectl rollout restart` keeps that version. To go back to tracking `:latest`, either
  `kubectl set image ... fall-dashboard=registry.smarko-health.de/fall-dashboard:latest` or run
  `helm upgrade` (which resets the image to the value in the values file).
- **Rollback restores code, not data.** Anything a version wrote to InfluxDB stays there.
