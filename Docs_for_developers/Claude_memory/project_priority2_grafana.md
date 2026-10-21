---
name: Priority 2 — Model comparison (complete)
description: Model comparison is now a built-in operator dashboard sub-page, not Grafana. Status and key decisions recorded.
type: project
---

Priority 2 is **complete**. Model comparison moved from Grafana to a built-in operator dashboard sub-page.

**Decision:** Grafana model_comparison.json was abandoned. Model comparison is now at `/operator/model_comparison.html`, powered by Plotly.js charts fetching from `GET /model/comparison` on ml_server.

**Why:** Simpler architecture — no Grafana datasource UID complexity, charts live inside the operator's own UI, interactive zoom/pan/hover built into Plotly.

**What was implemented (2026-03-26):**
- `GET /model/comparison?since_days=N` endpoint in `system_operator/ml_server/server.py` — queries `inference_log WHERE inference_mode='replay'`, returns aggregated stats + raw timeseries
- `system_operator/operator_dashboard/model_comparison.html` — standalone sub-page
- `system_operator/operator_dashboard/model_comparison.js` — 5 Plotly charts: fall rate bar, latency grouped bar, confidence box plot (falls vs no-falls), confidence histogram, confidence scatter; plus percentile table + per-recording × model table + sessions audit table
- `index.html` updated: Grafana model-comparison card now links to `model_comparison.html`; "View Full Model Comparison →" button added after replay results

**Key implementation note:** `model_comparison.js` uses `const ML_SERVER = '/api/ml'` (same as `app.js` `ML_API`) — must always route through nginx proxy.

**Grafana still used for:** server health (ml_server_overview), model drift (model_performance), fall events timeline. Those 3 dashboards remain linked from the operator dashboard.

**End-to-end model comparison workflow:**
1. Upload CSV → MinIO (operator dashboard or http://localhost:9001)
2. Switch model → Run Replay → repeat for each model version
3. Click "View Full Model Comparison →" or open http://localhost/operator/model_comparison.html
