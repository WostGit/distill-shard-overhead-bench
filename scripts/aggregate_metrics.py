from __future__ import annotations

import argparse
import glob
from pathlib import Path

from logging_utils import log_event
from metrics_utils import read_json, write_json
from model_config import resolve_model_id


def aggregate(args: argparse.Namespace) -> dict:
    log_event("aggregate metrics start")

    baseline = read_json(args.single_baseline_metrics)
    merge_metrics = read_json(args.merge_metrics)
    train_metrics = read_json(args.train_metrics)

    teacher_shard_metrics = [read_json(p) for p in sorted(glob.glob(args.teacher_metrics_glob))]
    eval_shard_metrics = [read_json(p) for p in sorted(glob.glob(args.eval_metrics_glob))]

    teacher_wall = max((m["teacher_compute_time_sec"] for m in teacher_shard_metrics), default=0.0)
    teacher_sum = sum(m["teacher_compute_time_sec"] for m in teacher_shard_metrics)
    eval_wall = max((m["eval_time_sec"] for m in eval_shard_metrics), default=0.0)
    setup_time_sec = baseline.get("setup_time_sec", 0.0)
    merge_time_sec = merge_metrics.get("merge_time_sec", 0.0)

    artifact_bytes = 0
    for pattern in [args.teacher_outputs_glob, args.eval_outputs_glob]:
        for file_path in glob.glob(pattern):
            artifact_bytes += Path(file_path).stat().st_size

    artifact_transfer_time_sec = float(args.artifact_transfer_time_sec)
    idle_or_wait_time_sec = max(0.0, (teacher_sum - teacher_wall) + (sum(m["eval_time_sec"] for m in eval_shard_metrics) - eval_wall))

    sharded_wall = (
        setup_time_sec
        + teacher_wall
        + merge_time_sec
        + train_metrics.get("train_time_sec", 0.0)
        + eval_wall
        + artifact_transfer_time_sec
    )
    single_wall = baseline.get("total_wall_time_sec", 0.0)

    speedup = (single_wall / sharded_wall) if sharded_wall > 0 else 0.0
    single_teacher_eval = baseline.get("teacher_compute_time_sec", 0.0) + baseline.get("eval_time_sec", 0.0)
    parallel_efficiency = (single_teacher_eval / (5 * (teacher_wall + eval_wall))) if (teacher_wall + eval_wall) > 0 else 0.0
    communication_overhead_fraction = (
        (merge_time_sec + idle_or_wait_time_sec + artifact_transfer_time_sec) / sharded_wall
        if sharded_wall > 0
        else 0.0
    )

    aggregate_payload = {
        "phase": "phase2_sharded_pipeline",
        "model_id": resolve_model_id(args.model_id),
        "single_runner_wall_time_sec": single_wall,
        "sharded_wall_time_sec": sharded_wall,
        "teacher_compute_time_sec": teacher_wall,
        "merge_time_sec": merge_time_sec,
        "train_time_sec": train_metrics.get("train_time_sec", 0.0),
        "eval_time_sec": eval_wall,
        "setup_time_sec": setup_time_sec,
        "idle_or_wait_time_sec": idle_or_wait_time_sec,
        "artifact_transfer_bytes": artifact_bytes,
        "artifact_transfer_time_sec": artifact_transfer_time_sec,
        "failure_count": int(args.failure_count),
        "retry_count": int(args.retry_count),
        "speedup_vs_single": speedup,
        "parallel_efficiency": parallel_efficiency,
        "communication_overhead_fraction": communication_overhead_fraction,
        "teacher_shard_count": len(teacher_shard_metrics),
        "eval_shard_count": len(eval_shard_metrics),
    }

    write_json(args.output_path, aggregate_payload)
    log_event(
        "final bottleneck summary: "
        f"teacher_wall={teacher_wall:.3f}s merge={merge_time_sec:.3f}s train={train_metrics.get('train_time_sec', 0.0):.3f}s "
        f"eval_wall={eval_wall:.3f}s transfer={artifact_transfer_time_sec:.3f}s"
    )
    return aggregate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-baseline-metrics", required=True)
    parser.add_argument("--teacher-metrics-glob", required=True)
    parser.add_argument("--merge-metrics", required=True)
    parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--eval-metrics-glob", required=True)
    parser.add_argument("--teacher-outputs-glob", required=True)
    parser.add_argument("--eval-outputs-glob", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--artifact-transfer-time-sec", default="0")
    parser.add_argument("--failure-count", default="0")
    parser.add_argument("--retry-count", default="0")
    parser.add_argument("--model-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())
