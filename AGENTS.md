# Agent instructions for this repository

## Scope
These instructions apply to the entire repository.

## Instrumentation discipline
- Preserve explicit timing for setup, model load, teacher compute, merge, training, evaluation, transfer, waiting, and total wall time.
- Emit machine-readable JSON metrics for every benchmark phase and shard where applicable.
- Keep timestamped logs for every major phase transition (start/end markers).
- Always log the selected model ID and shard indices when sharding is used.

## Orchestration philosophy
- Prefer simple, auditable GitHub Actions workflows over highly abstract orchestration frameworks.
- Keep this repository self-contained (no dependency on external local repos).
- Do not replace transparent metrics fields with opaque abstractions.
- Keep GitHub Actions reproducibility as a top priority.

## Model policy
- Do not silently change model size or model family.
- If model selection changes, document it in both metrics output and README.
