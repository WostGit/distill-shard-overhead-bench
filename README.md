# Distill shard overhead benchmark (GitHub Actions, macOS Apple Silicon)

This repository benchmarks a tiny, CPU-only distillation-style pipeline on GitHub-hosted macOS Apple Silicon runners to measure the practical speedup from sharding teacher/eval work across 5 runners versus the overhead paid for setup, artifact transfer, merge, waiting, and straggler effects.

## Model choices
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`
- Optional second model: `Qwen/Qwen2.5-1.5B-Instruct`
- Optional stress model for later prompts: `Qwen/Qwen2.5-3B-Instruct`

All scripts enforce an allowlist in `scripts/model_config.py`. By default, workflows run with `Qwen/Qwen2.5-0.5B-Instruct` for reproducible, budget-aware CPU benchmarking.

## Why small Qwen variants
- Standard GitHub-hosted macOS runners have constrained CPU, memory, and disk.
- The benchmark goal is pipeline parallelism and overhead accounting, not maximum model quality.
- Smaller models keep CI runtime practical while still exercising model load, generation, and eval paths.
- 7B+ defaults are intentionally avoided to reduce out-of-memory and excessive download risk on hosted runners.

## What this benchmark measures
- Phase 1 (single runner): end-to-end baseline timing for setup, model load, teacher generation, student training, evaluation, total wall time, and artifact bytes.
- Phase 2 (5-way sharded): teacher and eval sharding overhead/benefit across 5 runners, plus merge and transfer costs.
- Machine-readable JSON metrics from every phase and shard for transparent comparisons.

## What this benchmark does not claim
- It does **not** claim 5 hosted runners behave like one large machine.
- It does **not** shard student training in Phase 1 or Phase 2.
- It does **not** optimize for peak model quality; this is throughput/overhead instrumentation.
- It does **not** hide orchestration details behind opaque tooling.

## How to interpret speedup and overhead
- `speedup_vs_single = single_runner_wall_time_sec / sharded_wall_time_sec`
- `parallel_efficiency = single_runner_teacher+eval_time / (5 * sharded_teacher+eval_wall_time)`
- `communication_overhead_fraction = (merge + waiting + transfer) / total_sharded_wall_time`

Interpretation tips:
- High speedup with low overhead fraction means sharding helped.
- Lower efficiency can still be acceptable if wall time dropped enough for CI targets.
- High overhead fraction indicates orchestration or transfer dominates and should be reduced.

## Repository layout
- `data/` tiny checked-in JSONL prompt/eval files.
- `scripts/` explicit, timestamped benchmark scripts and metric aggregation.
- `.github/workflows/phase1-single-baseline-macos.yml` single-runner baseline workflow.
- `.github/workflows/phase2-sharded-pipeline-macos.yml` 5-way sharded teacher/eval workflow.
- `outputs/metrics/` machine-readable metrics artifacts.

## Running locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_single_baseline.py
```

## Output contract status
- ✅ Phase 1 implemented: single-runner workflow, JSON metrics artifact, clear timestamped logs.
- ✅ Phase 2 implemented: 5-way teacher/eval sharding, per-job JSON metrics, aggregate JSON report.
