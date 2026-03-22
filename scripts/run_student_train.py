from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_utils import log
from metrics_utils import timed, write_json
from model_config import get_model_id, validate_model_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-input", required=True)
    p.add_argument("--student-output", required=True)
    p.add_argument("--metrics", required=True)
    args = p.parse_args()

    model_id = get_model_id()
    validate_model_id(model_id)
    log(f"train start model_id={model_id}")

    teacher_path = Path(args.teacher_input)
    lines = [json.loads(line) for line in teacher_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    with timed() as train_t:
        # Lightweight stand-in for a tiny distillation pass: summarize teacher outputs.
        total_chars = sum(len(x.get("teacher", "")) for x in lines)
        avg_chars = (total_chars / len(lines)) if lines else 0.0
        student = {
            "model_id": model_id,
            "num_examples": len(lines),
            "avg_teacher_chars": avg_chars,
            "note": "Synthetic lightweight student artifact for pipeline overhead benchmarking.",
        }

    out = Path(args.student_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(student, indent=2), encoding="utf-8")

    payload = {
        "model_id": model_id,
        "train_time_sec": train_t.seconds,
        "student_artifact_bytes": out.stat().st_size if out.exists() else 0,
    }
    write_json(args.metrics, payload)
    log("train end")


if __name__ == "__main__":
    main()
