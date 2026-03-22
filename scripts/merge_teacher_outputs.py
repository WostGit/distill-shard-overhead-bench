from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from logging_utils import log_event
from metrics_utils import bytes_on_disk, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    log_event("merge start")
    start = time.perf_counter()
    rows: list[dict] = []
    for in_path in args.inputs:
        with Path(in_path).open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    rows.sort(key=lambda r: r["id"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    elapsed = time.perf_counter() - start
    artifact_bytes = bytes_on_disk(list(args.inputs) + [output_path])
    write_json(
        args.metrics_output,
        {
            "merge_time_sec": elapsed,
            "num_rows": len(rows),
            "artifact_transfer_bytes": artifact_bytes,
        },
    )
    log_event(f"merge end: rows={len(rows)} wall={elapsed:.3f}s bytes={artifact_bytes}")


if __name__ == "__main__":
    main()
