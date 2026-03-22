# distill-shard-overhead-bench

This repository is a self-contained GitHub Actions benchmark that measures single-runner versus 5-way sharded execution overhead in a tiny CPU-only distillation-style pipeline on GitHub-hosted macOS Apple Silicon runners.

## Model choices
- Default model (teacher + student by default): `Qwen/Qwen2.5-0.5B-Instruct`
- Secondary configured model: `Qwen/Qwen2.5-1.5B-Instruct`
- Optional stress model for later phases only: `Qwen/Qwen2.5-3B-Instruct`

The scripts read `TEACHER_MODEL_ID` and `STUDENT_MODEL_ID` environment variables. If unset, they default to the 0.5B model to keep CPU-only CI practical.

## Why small Qwen variants
Small Qwen variants are used so benchmarks remain runnable on standard hosted macOS runners with limited CPU, RAM, and disk, while still preserving realistic model load, generation, and evaluation costs and enabling reproducible 5-way sharding experiments.

## What this benchmark measures
- Phase 1 (`phase1-single-baseline-macos.yml`): single-runner end-to-end baseline with teacher generation, student training, evaluation, and one JSON metrics artifact.
- Phase 2 (`phase2-sharded-pipeline-macos.yml`): 5-way sharded teacher + 5-way sharded eval, with merge + single-runner train and an aggregate JSON report computing speedup and overhead.
- Per-stage timing, artifact size accounting, and explicit overhead-oriented metrics (merge, waiting, transfer).

## What this benchmark does not claim
- It does **not** claim five hosted runners behave like one large workstation.
- It does **not** implement tightly coupled student-training sharding in Phase 1 or Phase 2.
- It does **not** claim production-quality model training quality from this tiny CI dataset.

## How to interpret speedup and overhead
- `speedup_vs_single = single_runner_wall_time_sec / sharded_wall_time_sec`
- `parallel_efficiency = single-runner teacher+eval time / (5 * sharded teacher+eval wall time)`
- `communication_overhead_fraction = (merge + waiting + transfer) / total sharded wall time`

Interpret speedup together with communication overhead and bottleneck logs. High speedup with high overhead can still indicate fragility; lower speedup with stable overhead may be more reproducible.

## Repository layout
- `data/`: tiny prompt and eval JSONL files checked in.
- `scripts/`: all benchmark scripts, logging and metrics utilities, and model config.
- `outputs/metrics/.gitkeep`: artifact output folder placeholder.
- `.github/workflows/`: Phase 1 and Phase 2 CI workflows.

## Running locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_single_baseline.py
```

## GitHub Actions workflows
- **Phase 1**: runs `scripts/run_single_baseline.py`, then uploads `outputs/metrics/phase1_single_baseline_metrics.json`.
- **Phase 2**: runs baseline, 5 teacher shards, merge + train, 5 eval shards, and aggregate report upload.
