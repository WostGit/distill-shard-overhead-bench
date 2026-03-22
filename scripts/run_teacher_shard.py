from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import log_event
from metrics_utils import timed, write_json
from model_config import resolve_model_id


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shard_rows(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    return [row for idx, row in enumerate(rows) if idx % shard_count == shard_index]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(args: argparse.Namespace) -> dict:
    model_id = resolve_model_id(args.model_id)
    log_event(f"teacher shard start: shard={args.shard_index}/{args.shard_count}, model_id={model_id}")

    with timed() as load_timer:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
    model_load_time_sec = load_timer["elapsed"]

    all_rows = load_jsonl(args.prompts_path)
    rows = shard_rows(all_rows, args.shard_index, args.shard_count)

    outputs: list[dict] = []
    with timed() as compute_timer:
        for row in rows:
            prompt = row["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            full_text = tokenizer.decode(out[0], skip_special_tokens=True)
            completion = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text
            outputs.append(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "teacher_text": completion,
                    "model_id": model_id,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                }
            )
    teacher_compute_time_sec = compute_timer["elapsed"]

    write_jsonl(args.output_path, outputs)
    output_bytes = Path(args.output_path).stat().st_size

    metrics = {
        "event": "teacher_shard",
        "model_id": model_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "records_in_shard": len(rows),
        "model_load_time_sec": model_load_time_sec,
        "teacher_compute_time_sec": teacher_compute_time_sec,
        "output_artifact_bytes": output_bytes,
        "started_at_epoch_sec": time.time() - (model_load_time_sec + teacher_compute_time_sec),
        "finished_at_epoch_sec": time.time(),
    }

    if args.metrics_path:
        write_json(args.metrics_path, metrics)

    log_event(
        f"teacher shard end: shard={args.shard_index}/{args.shard_count}, rows={len(rows)}, "
        f"compute_sec={teacher_compute_time_sec:.3f}, bytes={output_bytes}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metrics-path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
