from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from logging_utils import log_event
from metrics_utils import bytes_on_disk, write_json


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-metrics", required=True)
    parser.add_argument("--teacher-metrics", nargs="+", required=True)
    parser.add_argument("--merge-metrics", required=True)
    parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--eval-metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_event("aggregate start")
    start = time.perf_counter()

    single = read_json(args.single_metrics)
    teacher_metrics = [read_json(p) for p in args.teacher_metrics]
    merge = read_json(args.merge_metrics)
    train = read_json(args.train_metrics)
    eval_metrics = [read_json(p) for p in args.eval_metrics]

    sharded_teacher_wall = max(m["total_wall_time_sec"] for m in teacher_metrics)
    sharded_eval_wall = max(m["eval_time_sec"] for m in eval_metrics)
    teacher_compute_sum = sum(m["teacher_compute_time_sec"] for m in teacher_metrics)
    eval_compute_sum = sum(m["eval_time_sec"] for m in eval_metrics)
    setup_time_sec = sum(m["setup_time_sec"] for m in teacher_metrics)

    artifact_transfer_bytes = bytes_on_disk(
        args.teacher_metrics + args.eval_metrics + [args.merge_metrics, args.train_metrics]
    )
    artifact_transfer_time_sec = 0.0

    sharded_wall_time_sec = (
        sharded_teacher_wall + merge["merge_time_sec"] + train["train_time_sec"] + sharded_eval_wall
    )
    idle_or_wait_time_sec = max(0.0, sharded_wall_time_sec - teacher_compute_sum - eval_compute_sum)

    speedup = single["total_wall_time_sec"] / sharded_wall_time_sec if sharded_wall_time_sec else 0.0
    denom = 5 * (sharded_teacher_wall + sharded_eval_wall)
    parallel_eff = (
        (single["teacher_compute_time_sec"] + single["eval_time_sec"]) / denom if denom else 0.0
    )
    comm_frac = (
        (merge["merge_time_sec"] + idle_or_wait_time_sec + artifact_transfer_time_sec)
        / sharded_wall_time_sec
        if sharded_wall_time_sec
        else 0.0
    )

    report = {
        "single_runner_wall_time_sec": single["total_wall_time_sec"],
        "sharded_wall_time_sec": sharded_wall_time_sec,
        "teacher_compute_time_sec": teacher_compute_sum,
        "merge_time_sec": merge["merge_time_sec"],
        "train_time_sec": train["train_time_sec"],
        "eval_time_sec": eval_compute_sum,
        "setup_time_sec": setup_time_sec,
        "idle_or_wait_time_sec": idle_or_wait_time_sec,
        "artifact_transfer_bytes": artifact_transfer_bytes,
        "artifact_transfer_time_sec": artifact_transfer_time_sec,
        "failure_count": 0,
        "retry_count": 0,
        "speedup_vs_single": speedup,
        "parallel_efficiency": parallel_eff,
        "communication_overhead_fraction": comm_frac,
        "model_id": single["model_id"],
        "bottleneck_summary": {
            "longest_teacher_shard_sec": sharded_teacher_wall,
            "longest_eval_shard_sec": sharded_eval_wall,
            "aggregate_elapsed_sec": time.perf_counter() - start,
        },
    }

    write_json(args.output, report)
    log_event(
        "aggregate end: "
        f"speedup={speedup:.4f} parallel_eff={parallel_eff:.4f} overhead_frac={comm_frac:.4f}"
    )


if __name__ == "__main__":
    main()
