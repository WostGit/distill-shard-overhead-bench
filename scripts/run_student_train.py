from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from logging_utils import log
from metrics_utils import Timer, write_json
from model_config import resolve_model_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-input", type=Path, required=True)
    p.add_argument("--student-output", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id()

    log(f"Student training start: model={model_id}")
    with Timer() as train_t:
        token_counts: Counter[str] = Counter()
        total_chars = 0
        with args.teacher_input.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row["teacher_text"]
                total_chars += len(text)
                token_counts.update(text.lower().split())

        snapshot = {
            "model_id": model_id,
            "total_teacher_chars": total_chars,
            "top_tokens": token_counts.most_common(100),
        }
        args.student_output.parent.mkdir(parents=True, exist_ok=True)
        with args.student_output.open("w", encoding="utf-8") as out:
            json.dump(snapshot, out, indent=2)

    write_json(
        args.metrics,
        {
            "model_id": model_id,
            "train_time_sec": round(train_t.seconds, 4),
            "vocab_size": len(token_counts),
        },
    )
    log(f"Student training end: vocab_size={len(token_counts)}")


if __name__ == "__main__":
    main()
