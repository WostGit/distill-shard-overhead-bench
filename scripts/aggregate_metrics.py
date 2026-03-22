from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_utils import log
from metrics_utils import read_json, write_json
from model_config import resolve_model_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--single-metrics", type=Path, required=True)
    p.add_argument("--teacher-metrics-dir", type=Path, required=True)
    p.add_argument("--eval-metrics-dir", type=Path, required=True)
    p.add_argument("--merge-metrics", type=Path, required=True)
    p.add_argument("--transfer-bytes", type=int, default=0)
    p.add_argument("--transfer-time-sec", type=float, default=0.0)
    p.add_argument("--idle-or-wait-time-sec", type=float, default=0.0)
    p.add_argument("--failure-count", type=int, default=0)
    p.add_argument("--retry-count", type=int, default=0)
    p.add_argument("--sharded-wall-time-sec", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_many_json(pattern: str, d: Path) -> list[dict]:
    return [read_json(p) for p in sorted(d.glob(pattern))]


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id()

    single = read_json(args.single_metrics)
    teacher_metrics = load_many_json("teacher_metrics_shard_*.json", args.teacher_metrics_dir)
    eval_metrics = load_many_json("eval_metrics_shard_*.json", args.eval_metrics_dir)
    merge = read_json(args.merge_metrics)

    teacher_compute = max((m.get("teacher_compute_time_sec", 0.0) for m in teacher_metrics), default=0.0)
    eval_compute = max((m.get("eval_time_sec", 0.0) for m in eval_metrics), default=0.0)
    train_time = float(single.get("train_time_sec", 0.0))
    setup_time = float(single.get("setup_time_sec", 0.0))
    single_wall = float(single.get("total_wall_time_sec", 0.0))
    sharded_wall = float(args.sharded_wall_time_sec)
    merge_time = float(merge.get("merge_time_sec", 0.0))

    speedup = (single_wall / sharded_wall) if sharded_wall > 0 else 0.0
    single_teacher_eval = float(single.get("teacher_compute_time_sec", 0.0)) + float(single.get("eval_time_sec", 0.0))
    shard_teacher_eval_wall = teacher_compute + eval_compute
    parallel_efficiency = (single_teacher_eval / (5 * shard_teacher_eval_wall)) if shard_teacher_eval_wall > 0 else 0.0
    overhead_fraction = (
        (merge_time + args.idle_or_wait_time_sec + args.transfer_time_sec) / sharded_wall
        if sharded_wall > 0
        else 0.0
    )

    result = {
        "model_id": model_id,
        "single_runner_wall_time_sec": round(single_wall, 4),
        "sharded_wall_time_sec": round(sharded_wall, 4),
        "teacher_compute_time_sec": round(teacher_compute, 4),
        "merge_time_sec": round(merge_time, 4),
        "train_time_sec": round(train_time, 4),
        "eval_time_sec": round(eval_compute, 4),
        "setup_time_sec": round(setup_time, 4),
        "idle_or_wait_time_sec": round(args.idle_or_wait_time_sec, 4),
        "artifact_transfer_bytes": int(args.transfer_bytes),
        "artifact_transfer_time_sec": round(args.transfer_time_sec, 4),
        "failure_count": int(args.failure_count),
        "retry_count": int(args.retry_count),
        "speedup_vs_single": round(speedup, 4),
        "parallel_efficiency": round(parallel_efficiency, 4),
        "communication_overhead_fraction": round(overhead_fraction, 4),
    }
    write_json(args.output, result)
    log(
        "Final bottleneck summary: "
        f"teacher_eval_wall={shard_teacher_eval_wall:.3f}s merge={merge_time:.3f}s transfer={args.transfer_time_sec:.3f}s wait={args.idle_or_wait_time_sec:.3f}s"
    )


if __name__ == "__main__":
    main()
