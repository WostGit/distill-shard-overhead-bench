from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from logging_utils import log_event
from merge_teacher_outputs import merge
from metrics_utils import read_json, timed, write_json
from model_config import resolve_model_id
from run_eval_shard import evaluate
from run_student_train import train
from run_teacher_shard import generate


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    started_epoch = time.time()
    model_id = resolve_model_id(args.model_id)
    log_event(f"single baseline start model_id={model_id}")

    with timed() as setup_timer:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

    teacher_output = os.path.join(args.output_dir, "teacher_single.jsonl")
    teacher_metrics_path = os.path.join(args.output_dir, "teacher_single_metrics.json")
    teacher_metrics = generate(
        argparse.Namespace(
            prompts_path=args.teacher_prompts,
            output_path=teacher_output,
            metrics_path=teacher_metrics_path,
            model_id=model_id,
            max_new_tokens=args.max_new_tokens,
            shard_index=0,
            shard_count=1,
        )
    )

    merged_output = os.path.join(args.output_dir, "teacher_merged.jsonl")
    merge(
        argparse.Namespace(
            input_glob=teacher_output,
            output_path=merged_output,
            metrics_path=os.path.join(args.output_dir, "merge_single_metrics.json"),
        )
    )

    student_model = os.path.join(args.output_dir, "student_model.json")
    train(
        argparse.Namespace(
            teacher_outputs_path=merged_output,
            output_model_path=student_model,
            metrics_path=os.path.join(args.output_dir, "student_train_metrics.json"),
            top_k=128,
        )
    )
    train_metrics = read_json(os.path.join(args.output_dir, "student_train_metrics.json"))

    eval_output = os.path.join(args.output_dir, "eval_single.jsonl")
    evaluate(
        argparse.Namespace(
            eval_path=args.eval_path,
            student_model_path=student_model,
            output_path=eval_output,
            metrics_path=os.path.join(args.output_dir, "eval_single_metrics.json"),
            shard_index=0,
            shard_count=1,
        )
    )
    eval_metrics = read_json(os.path.join(args.output_dir, "eval_single_metrics.json"))

    total_wall = time.perf_counter() - started
    output_bytes = sum(Path(path).stat().st_size for path in [teacher_output, merged_output, student_model, eval_output])

    result = {
        "phase": "phase1_single_runner_baseline",
        "model_id": model_id,
        "setup_time_sec": setup_timer["elapsed"],
        "model_load_time_sec": teacher_metrics["model_load_time_sec"],
        "teacher_compute_time_sec": teacher_metrics["teacher_compute_time_sec"],
        "train_time_sec": train_metrics["train_time_sec"],
        "eval_time_sec": eval_metrics["eval_time_sec"],
        "total_wall_time_sec": total_wall,
        "output_artifact_bytes": output_bytes,
        "started_at_epoch_sec": started_epoch,
        "finished_at_epoch_sec": time.time(),
    }

    write_json(args.metrics_path, result)
    log_event(f"single baseline end total_wall_time_sec={total_wall:.3f}, output_artifact_bytes={output_bytes}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-prompts", default="data/tiny_teacher_prompts.jsonl")
    parser.add_argument("--eval-path", default="data/tiny_eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/phase1")
    parser.add_argument("--metrics-path", default="outputs/metrics/phase1_single_metrics.json")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
