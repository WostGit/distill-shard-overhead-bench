# distill-shard-overhead-bench

This repository provides a self-contained, GitHub Actions-first benchmark for measuring the tradeoff between useful parallel work and orchestration overhead when a tiny distillation-style pipeline is run on standard GitHub-hosted macOS Apple Silicon runners.

## Model choices
- Default: `Qwen/Qwen2.5-0.5B-Instruct`
- Second option: `Qwen/Qwen2.5-1.5B-Instruct`
- Optional later stress model: `Qwen/Qwen2.5-3B-Instruct`

The active model is controlled with `MODEL_ID` and is validated in `scripts/model_config.py`.

## Why small Qwen variants
Small Qwen variants are intentionally used because this benchmark targets CPU-only hosted runners with modest RAM and storage budgets; the goal is to measure pipeline overhead and practical speedup, not maximize raw model quality with very large checkpoints.

## What this benchmark measures
- **Phase 1 (single baseline)**: setup, model load, teacher generation, synthetic student-train stage, eval stage, total wall time, and output artifact bytes.
- **Phase 2 (5-way sharded)**: end-to-end sharded wall time, teacher/eval shard timing, merge timing, transfer-related metrics, waiting/idle components, and derived speedup/efficiency/overhead metrics.

## What this benchmark does not claim
- It does **not** claim that 5 hosted runners behave like one large tightly coupled machine.
- It does **not** claim state-of-the-art distillation quality.
- It does **not** include student-training sharding in Phase 1 or Phase 2.

## How to interpret speedup and overhead
- `speedup_vs_single = single_runner_wall_time_sec / sharded_wall_time_sec`
- `parallel_efficiency = single_runner_teacher+eval_time / (5 * sharded_teacher+eval_wall_time)`
- `communication_overhead_fraction = (merge + waiting + transfer) / total_sharded_wall_time`

Higher speedup is better, but efficiency and overhead should be read together to understand the coordination cost and straggler sensitivity.

## Workflow overview
- `.github/workflows/phase1-single-baseline-macos.yml` runs the single-runner baseline and uploads one JSON metrics artifact.
- `.github/workflows/phase2-sharded-pipeline-macos.yml` runs baseline reference + 5-way teacher sharding + merge + single-runner train + 5-way eval sharding + aggregate metrics.

## Local smoke run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_single_baseline.py --metrics outputs/metrics/phase1_single_baseline_metrics.json
```
