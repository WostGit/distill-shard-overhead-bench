# distill-shard-overhead-bench

This repository benchmarks a tiny, CPU-only distillation-style pipeline on standard GitHub-hosted macOS Apple Silicon runners to measure the practical speedup from sharding teacher generation and evaluation across 5 runners, while explicitly quantifying setup, transfer, merge, waiting, and straggler overhead.

## Model choices
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`
- Second model (supported switch): `Qwen/Qwen2.5-1.5B-Instruct`
- Optional stress model for a later phase only: `Qwen/Qwen2.5-3B-Instruct`

The workflows and scripts accept `--model-id`, but default to `Qwen/Qwen2.5-0.5B-Instruct`.

## Why small Qwen variants
- They are realistic for CPU-only CI environments with limited RAM/disk.
- They keep benchmark turn-around practical on hosted macOS runners.
- They preserve apples-to-apples comparisons across baseline and sharded runs without relying on oversized models.

## What this benchmark measures
- **Phase 1 (single-runner baseline):** teacher generation, student training, evaluation, and end-to-end wall time on one runner.
- **Phase 2 (5-way sharded):** teacher generation and evaluation sharded across 5 runners, merged and aggregated into one overhead report.
- Machine-readable metrics are emitted as JSON artifacts from each stage.

## What this benchmark does not claim
- It does **not** claim five hosted runners behave like one larger machine.
- It does **not** shard student training in Phase 1 or Phase 2.
- It does **not** claim universal throughput numbers beyond this tiny, auditable CI setup.

## How to interpret speedup and overhead
- `speedup_vs_single = single-runner wall time / sharded wall time`
- `parallel_efficiency = single-runner (teacher+eval time) / (5 * sharded teacher+eval wall time)`
- `communication_overhead_fraction = (merge + waiting + transfer time) / total sharded wall time`

High speedup with low communication overhead means sharding is helping for this pipeline shape. Lower efficiency and higher overhead indicate bottlenecks from synchronization, artifact transfer, or stragglers.

## Workflows
- `.github/workflows/phase1-single-baseline-macos.yml`: single-runner baseline with Phase 1 metrics artifact.
- `.github/workflows/phase2-sharded-pipeline-macos.yml`: 5-way sharded teacher/eval workflow with aggregate Phase 2 report.

## Local quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_single_baseline.py
```

Outputs are written under `outputs/` and metrics JSON under `outputs/metrics/`.
