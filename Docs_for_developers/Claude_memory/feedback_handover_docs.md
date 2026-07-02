---
name: Handover docs must be detailed and self-contained
description: When writing files in handover_docs/, write them so colleagues can read once and understand without coming back with questions
type: feedback
originSessionId: a82d8b16-33ae-4bc4-b1d0-bed308f7c1b5
---
Handover docs in `handover_docs/` are addressed to colleagues taking over after Hayate. They must be detailed, coherent, and self-contained — the reader should not need to come back to Hayate with follow-up questions.

**Why:** This is the artefact Hayate hands off when leaving the project. The cost of an unclear doc is a meeting Hayate has to take after handover; the cost of a long doc is just reading time.

**How to apply:**
- One file per audience. Current pattern: `ADMIN_ml_ops_related.md` (whoever runs MLflow / retraining), `ISA_mobile_app.md` (mobile app dev), `Tech_integrator.md` (FOCUS DevOps).
- Lead with audience + scope + repo/branch. State up front what the doc covers and what it doesn't (point at sibling docs for the rest).
- Include exact commands, exact JSON payloads, exact field rules. Don't just describe — show.
- Call out gotchas explicitly with a "things that will bite you" / "open questions" section. The reader inherits these.
- Cross-link siblings rather than duplicating content across docs.
- Don't be terse. Default Claude Code conciseness rules don't apply here — the user wants narrative depth.
