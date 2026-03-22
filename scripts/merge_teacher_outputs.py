from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_utils import log
from metrics_utils import Timer, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard_files = sorted(args.input_dir.glob("teacher_shard_*.jsonl"))
    log(f"Merge start: found {len(shard_files)} teacher shard files")
    merged: list[dict] = []
    with Timer() as merge_t:
        for sf in shard_files:
            with sf.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))
        merged.sort(key=lambda x: x["id"])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as out:
            for item in merged:
                out.write(json.dumps(item) + "\n")

    write_json(
        args.metrics,
        {
            "merge_time_sec": round(merge_t.seconds, 4),
            "merged_examples": len(merged),
            "input_files": len(shard_files),
        },
    )
    log(f"Merge end: merged_examples={len(merged)}")


if __name__ == "__main__":
    main()
