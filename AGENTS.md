# AGENTS.md

## Repository discipline
- Preserve explicit timing and overhead instrumentation across scripts and workflows.
- Prefer simple, auditable orchestration over hidden control flow.
- Keep this repository self-contained for GitHub Actions runs.
- Do not replace transparent metrics with opaque abstractions.
- Keep GitHub Actions reproducibility as a top priority (fixed runner labels and deterministic shard counts).
- Do not silently change model size; document model choice changes in metrics and docs.

## Phase execution guardrails
- Complete Phase 1 baseline first and keep it working before adding later phases.
- Keep student training on a single runner for Phase 1 and Phase 2.
- Do not introduce tightly coupled student-sharding prototypes in this repository during these phases.
