## Checking Grafana & Prometheus

Yes, both have browser UIs. With the stack running:

| Service | URL | Login |
| --- | --- | --- |
| **Grafana** | `http://localhost/grafana/` | `admin` / value of `GRAFANA_ADMIN_PASSWORD` in `.env` (default: `admin`) |
| **Prometheus** | `http://localhost:9090` | No login |

Prometheus port 9090 is internal-only (not published through nginx), so access it directly. Check **Status → Targets** — you should see `ml_server` as UP and data flowing in.

For Grafana, go to **Dashboards → Fall Detection** — the 3 dashboards should already be there. Make a few inference requests from `main.py` first so there's data to display, then refresh the dashboards.

---

## Port 8001 direct vs. through nginx — which is correct?

**For `main.py` running on your Windows host: direct to 8001 is correct and intentional.**

Here's why:
```bash
Scenario A — main.py (Windows host) → localhost:8001 (direct)  ✓ correct
Scenario B — browser/caregiver/emergency → localhost:80/api/ml/ → nginx → ml_server:8001  ✓ also correct
```
These are two different clients with different needs:

- `main.py` is a **trusted internal client** — it's the fall detection engine running on your laptop. It talks directly to the ML server over a published port. API key authentication still applies, so it's not unprotected.
- **Browsers and dashboards** go through nginx because nginx handles routing, SSL termination, and serves the static HTML files. It's the public-facing entry point.

**In production** you would typically remove the `"8001:8001"` port publishing and force everything through nginx — that way port 8001 is never exposed on the network at all. But for development, direct access is fine and actually simpler because you avoid nginx's auth headers and can hit `/docs` directly at `http://localhost:8001/docs`.