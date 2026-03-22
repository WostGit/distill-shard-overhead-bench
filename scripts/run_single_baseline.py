from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from logging_utils import log_event
from metrics_utils import bytes_on_disk, write_json
from model_config import teacher_model_id


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/phase1")
    parser.add_argument("--metrics-output", default="outputs/metrics/phase1_single_baseline_metrics.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_start = time.perf_counter()
    setup_start = time.perf_counter()
    log_event(f"phase1 baseline start: model={teacher_model_id()}")

    teacher_out = output_dir / "teacher_outputs.jsonl"
    teacher_metrics = output_dir / "teacher_metrics.json"
    subprocess.run(
        [
            "python",
            "scripts/run_teacher_shard.py",
            "--output",
            str(teacher_out),
            "--metrics-output",
            str(teacher_metrics),
            "--num-shards",
            "1",
            "--shard-id",
            "0",
        ],
        check=True,
    )

    train_metrics = output_dir / "train_metrics.json"
    subprocess.run(
        [
            "python",
            "scripts/run_student_train.py",
            "--teacher-input",
            str(teacher_out),
            "--metrics-output",
            str(train_metrics),
            "--max-steps",
            "1",
        ],
        check=True,
    )

    eval_metrics = output_dir / "eval_metrics.json"
    subprocess.run(
        [
            "python",
            "scripts/run_eval_shard.py",
            "--metrics-output",
            str(eval_metrics),
            "--num-shards",
            "1",
            "--shard-id",
            "0",
        ],
        check=True,
    )

    total_wall = time.perf_counter() - phase_start
    teacher = read_json(teacher_metrics)
    train = read_json(train_metrics)
    evaluation = read_json(eval_metrics)
    output_bytes = bytes_on_disk([output_dir])

    phase1_metrics = {
        "model_id": teacher["model_id"],
        "setup_time_sec": time.perf_counter() - setup_start - teacher["teacher_compute_time_sec"] - train["train_time_sec"] - evaluation["eval_time_sec"],
        "model_load_time_sec": teacher["model_load_time_sec"],
        "teacher_compute_time_sec": teacher["teacher_compute_time_sec"],
        "train_time_sec": train["train_time_sec"],
        "eval_time_sec": evaluation["eval_time_sec"],
        "total_wall_time_sec": total_wall,
        "output_artifact_bytes": output_bytes,
    }
    write_json(args.metrics_output, phase1_metrics)

    log_event(
        "phase1 baseline end: "
        f"total={total_wall:.3f}s teacher={teacher['teacher_compute_time_sec']:.3f}s "
        f"train={train['train_time_sec']:.3f}s eval={evaluation['eval_time_sec']:.3f}s bytes={output_bytes}"
    )


if __name__ == "__main__":
    main()
