# AGENTS.md

## Repository operating principles
- Preserve explicit timing and overhead instrumentation in every phase.
- Prefer simple, auditable orchestration over clever abstractions.
- Keep the repository self-contained with tiny checked-in datasets.
- Do not replace transparent JSON metrics with opaque reporting layers.
- Keep GitHub Actions reproducibility as a top priority.
- Do not silently change model size; document model changes in metrics and docs.

## Workflow discipline
- Phase 1 must stay as a clean single-runner baseline.
- Phase 2 can shard teacher/eval across five runners, but keep student training on one runner.
- Timestamp major stages, print model IDs, and emit machine-readable metrics artifacts.
