from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from logging_utils import log_event
from metrics_utils import timed, write_json


def merge(args: argparse.Namespace) -> dict:
    log_event("merge teacher outputs start")
    paths = sorted(glob.glob(args.input_glob))
    merged: list[dict] = []

    with timed() as timer:
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        merged.append(json.loads(line))

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as out:
            for row in sorted(merged, key=lambda x: x["id"]):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_bytes = Path(args.output_path).stat().st_size
    metrics = {
        "event": "merge_teacher_outputs",
        "input_shards": len(paths),
        "merged_records": len(merged),
        "merge_time_sec": timer["elapsed"],
        "output_artifact_bytes": output_bytes,
    }
    if args.metrics_path:
        write_json(args.metrics_path, metrics)
    log_event(f"merge teacher outputs end: shards={len(paths)}, records={len(merged)}, bytes={output_bytes}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metrics-path")
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
