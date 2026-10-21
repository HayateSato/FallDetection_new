---
name: jcodemunch / jdocmunch indexes for this repo
description: Where the code and doc indexes live and how they're scoped — use search_symbols / search_sections before native Grep/Read in this repo
type: reference
originSessionId: a82d8b16-33ae-4bc4-b1d0-bed308f7c1b5
---
This repo has both jcodemunch (code) and jdocmunch (doc) MCP indexes. Prefer them over native Grep/Read for symbol lookup or section search — they are faster and pre-compute structure.

| Index | Repo identifier | Source root | Scope |
|-------|----------------|-------------|-------|
| Code | `local/_6G_Integration_v2_mqtt-0d8cd46f` | `_6G_Integration_v2_mqtt/` | Active 6G/Charite folder only — not the full repo |
| Docs | `local/fall-detection-work` | repo root (`FallDetection_new/`) | Whole repo: `handover_docs/`, `REFACTOR_DOCUS/`, READMEs, helm/grafana YAML+JSON configs |

**How to refresh after meaningful changes:**
- Code: `mcp__jcodemunch__index_folder` with `path=_6G_Integration_v2_mqtt`, `incremental=true`
- Docs: `mcp__jdocmunch__index_local` with `path=FallDetection_new`, `name="fall-detection-work"`, `incremental=false` (full re-index — picks up new files cleanly)

**Caveats:**
- Doc index is BM25-only — no embedding provider configured, so semantic search returns false from `index_local`. Keyword queries only.
- Other folders in the repo (`_EcoSystem_Integration/`, `_6G_Integration_v2_redis/` frozen, full-system branches) are NOT in the code index. If you need to search them, run `index_folder` against that path or fall back to native Grep.
- `.env` files and `helm/.../secrets.yaml` are auto-skipped as secrets — expected.
