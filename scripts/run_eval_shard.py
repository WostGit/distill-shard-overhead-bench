from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from logging_utils import log_event
from metrics_utils import timed, write_json


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shard_rows(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    return [row for idx, row in enumerate(rows) if idx % shard_count == shard_index]


def evaluate(args: argparse.Namespace) -> dict:
    log_event(f"eval shard start: shard={args.shard_index}/{args.shard_count}")
    eval_rows = shard_rows(load_jsonl(args.eval_path), args.shard_index, args.shard_count)

    with open(args.student_model_path, "r", encoding="utf-8") as handle:
        student = json.load(handle)
    known_tokens = {tok for tok, _count in student.get("top_tokens", [])}

    results: list[dict] = []
    with timed() as timer:
        for row in eval_rows:
            targets = set(tok.lower() for tok in row.get("target_keywords", []))
            hit_count = len(targets.intersection(known_tokens))
            denom = max(1, len(targets))
            score = hit_count / denom
            results.append({"id": row["id"], "score": score, "hit_count": hit_count, "target_count": len(targets)})

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")

    output_bytes = Path(args.output_path).stat().st_size
    mean_score = sum(r["score"] for r in results) / max(1, len(results))

    metrics = {
        "event": "eval_shard",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "eval_time_sec": timer["elapsed"],
        "records_in_shard": len(results),
        "mean_score": mean_score,
        "output_artifact_bytes": output_bytes,
    }
    if args.metrics_path:
        write_json(args.metrics_path, metrics)
    log_event(
        f"eval shard end: shard={args.shard_index}/{args.shard_count}, rows={len(results)}, "
        f"eval_sec={timer['elapsed']:.3f}, mean_score={mean_score:.3f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metrics-path")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
