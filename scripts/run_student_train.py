from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from logging_utils import log_event
from metrics_utils import timed, write_json


def tokenize(text: str) -> list[str]:
    return [tok.strip(".,:;!?()[]{}\"'`).-").lower() for tok in text.split() if tok.strip()]


def train(args: argparse.Namespace) -> dict:
    log_event("student train start")
    counter: Counter[str] = Counter()
    samples = 0

    with timed() as timer:
        with open(args.teacher_outputs_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                counter.update(tokenize(row.get("teacher_text", "")))
                samples += 1

        model_payload = {
            "student_type": "token_frequency_proxy",
            "top_tokens": counter.most_common(args.top_k),
            "num_training_samples": samples,
        }
        os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
        with open(args.output_model_path, "w", encoding="utf-8") as handle:
            json.dump(model_payload, handle, indent=2)
            handle.write("\n")

    output_bytes = Path(args.output_model_path).stat().st_size
    metrics = {
        "event": "student_train",
        "train_time_sec": timer["elapsed"],
        "num_training_samples": samples,
        "output_artifact_bytes": output_bytes,
    }
    if args.metrics_path:
        write_json(args.metrics_path, metrics)
    log_event(f"student train end: samples={samples}, train_sec={timer['elapsed']:.3f}, bytes={output_bytes}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-outputs-path", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--metrics-path")
    parser.add_argument("--top-k", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
