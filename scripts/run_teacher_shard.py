from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import log_event
from metrics_utils import bytes_on_disk, write_json
from model_config import teacher_model_id


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tiny_teacher_prompts.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    model_id = teacher_model_id()
    log_event(f"teacher shard start: shard={args.shard_id}/{args.num_shards} model={model_id}")
    prompts = load_jsonl(args.input)
    shard_rows = [r for i, r in enumerate(prompts) if i % args.num_shards == args.shard_id]
    setup_start = time.perf_counter()

    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model_load_time_sec = time.perf_counter() - model_load_start

    compute_start = time.perf_counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in shard_rows:
            prompt = row["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            out.write(json.dumps({"id": row["id"], "prompt": prompt, "teacher_text": text}) + "\n")

    teacher_compute_time_sec = time.perf_counter() - compute_start
    total_wall = time.perf_counter() - setup_start
    artifact_bytes = bytes_on_disk([output_path])

    metrics = {
        "model_id": model_id,
        "num_rows": len(shard_rows),
        "output_artifact_bytes": artifact_bytes,
        "setup_time_sec": total_wall - teacher_compute_time_sec,
        "model_load_time_sec": model_load_time_sec,
        "teacher_compute_time_sec": teacher_compute_time_sec,
        "total_wall_time_sec": total_wall,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    write_json(args.metrics_output, metrics)
    log_event(
        "teacher shard end: "
        f"shard={args.shard_id} rows={len(shard_rows)} wall={total_wall:.3f}s bytes={artifact_bytes}"
    )


if __name__ == "__main__":
    main()
