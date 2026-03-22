# AGENTS.md

## Repository intent
This repository benchmarks distillation-pipeline **parallelization overhead vs throughput gain** on GitHub-hosted macOS Apple Silicon runners.

## Non-negotiable instrumentation discipline
- Keep explicit timing for setup, model load, teacher, merge, train, eval, and total wall time.
- Preserve machine-readable JSON metrics in `outputs/metrics/`.
- Timestamp major stage logs and include shard start/end and merge start/end markers.
- Record artifact sizes and transfer-related metrics whenever available.
- Emit a final bottleneck summary in each end-to-end path.

## Orchestration principles
- Prefer simple, auditable job graphs in GitHub Actions.
- Keep repository self-contained with checked-in tiny datasets.
- Do not replace transparent metrics with opaque abstractions.
- Keep reproducibility on standard GitHub-hosted macOS runners as a top priority.

## Model policy
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Allowed alternate model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Optional later stress model: `Qwen/Qwen2.5-3B-Instruct`.
- Do not silently change model size; document changes in both metrics and README.
- Do not use 7B+ models by default.
