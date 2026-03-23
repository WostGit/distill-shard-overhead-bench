from __future__ import annotations

import argparse
from pathlib import Path

from logging_utils import log
from metrics_utils import timed, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", default="teacher_shard_*.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", required=True)
    args = p.parse_args()

    log("merge start")
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with timed() as merge_t:
        with out.open("w", encoding="utf-8") as wf:
            for f in files:
                wf.write(f.read_text(encoding="utf-8"))

    payload = {
        "merge_time_sec": merge_t.seconds,
        "input_files": len(files),
        "merged_artifact_bytes": out.stat().st_size if out.exists() else 0,
    }
    write_json(args.metrics, payload)
    log(f"merge end files={len(files)}")


if __name__ == "__main__":
    main()
