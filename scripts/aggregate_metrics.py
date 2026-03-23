from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from logging_utils import log
from metrics_utils import write_json
from model_config import get_model_id, validate_model_id


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sum_key(records, key):
    return float(sum(float(r.get(key, 0.0)) for r in records))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--single", required=True)
    p.add_argument("--teacher-metrics-glob", required=True)
    p.add_argument("--merge-metrics", required=True)
    p.add_argument("--train-metrics", required=True)
    p.add_argument("--eval-metrics-glob", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--artifact-transfer-bytes", type=float, default=0.0)
    p.add_argument("--artifact-transfer-time-sec", type=float, default=0.0)
    p.add_argument("--idle-or-wait-time-sec", type=float, default=0.0)
    p.add_argument("--setup-time-sec", type=float, default=0.0)
    p.add_argument("--failure-count", type=int, default=0)
    p.add_argument("--retry-count", type=int, default=0)
    args = p.parse_args()

    model_id = get_model_id()
    validate_model_id(model_id)

    single = read_json(args.single)
    teacher = [read_json(pth) for pth in sorted(glob.glob(args.teacher_metrics_glob))]
    merge = read_json(args.merge_metrics)
    train = read_json(args.train_metrics)
    evals = [read_json(pth) for pth in sorted(glob.glob(args.eval_metrics_glob))]

    teacher_compute = sum_key(teacher, "teacher_compute_time_sec")
    teacher_wall = max((float(x.get("teacher_compute_time_sec", 0.0)) for x in teacher), default=0.0)
    eval_compute = sum_key(evals, "eval_time_sec")
    eval_wall = max((float(x.get("eval_time_sec", 0.0)) for x in evals), default=0.0)

    merge_time = float(merge.get("merge_time_sec", 0.0))
    train_time = float(train.get("train_time_sec", 0.0))
    setup_time = float(args.setup_time_sec)
    idle_wait = float(args.idle_or_wait_time_sec)
    transfer_time = float(args.artifact_transfer_time_sec)

    sharded_wall = setup_time + teacher_wall + merge_time + train_time + eval_wall + idle_wait + transfer_time
    single_wall = float(single.get("total_wall_time_sec", 0.0))

    speedup = (single_wall / sharded_wall) if sharded_wall > 0 else 0.0
    single_teacher_eval = float(single.get("teacher_compute_time_sec", 0.0)) + float(single.get("eval_time_sec", 0.0))
    sharded_teacher_eval_wall = teacher_wall + eval_wall
    parallel_eff = (
        single_teacher_eval / (5.0 * sharded_teacher_eval_wall)
        if sharded_teacher_eval_wall > 0
        else 0.0
    )

    comm_overhead_fraction = (
        (merge_time + idle_wait + transfer_time) / sharded_wall if sharded_wall > 0 else 0.0
    )

    payload = {
        "model_id": model_id,
        "single_runner_wall_time_sec": single_wall,
        "sharded_wall_time_sec": sharded_wall,
        "teacher_compute_time_sec": teacher_compute,
        "merge_time_sec": merge_time,
        "train_time_sec": train_time,
        "eval_time_sec": eval_compute,
        "setup_time_sec": setup_time,
        "idle_or_wait_time_sec": idle_wait,
        "artifact_transfer_bytes": float(args.artifact_transfer_bytes),
        "artifact_transfer_time_sec": transfer_time,
        "failure_count": int(args.failure_count),
        "retry_count": int(args.retry_count),
        "speedup_vs_single": speedup,
        "parallel_efficiency": parallel_eff,
        "communication_overhead_fraction": comm_overhead_fraction,
    }
    write_json(args.output, payload)
    log("aggregate complete")


if __name__ == "__main__":
    main()
