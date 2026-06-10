# Production Configuration Checklists

Separated into one file per party.

| Party | File |
|-------|------|
| **Mohammed** — MCS inference layer (Hetzner `.env`) | [`config_checklist_mohammed.md`](config_checklist_mohammed.md) |
| **FOCUS DevOps** — caregiver layer (k3s `values_production.yaml`) | [`config_checklist_focus_devops.md`](config_checklist_focus_devops.md) |

Both files include the cross-reference table (values that must match between the two deployments)
and the cross-party sharing table (who sends what to whom).
