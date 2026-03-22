from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import log
from metrics_utils import Timer, write_json
from model_config import resolve_model_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=48)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id()
    rows = load_jsonl(args.input)
    shard = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]

    log(f"Teacher shard start: shard={args.shard_index}/{args.num_shards} model={model_id}")
    with Timer() as load_t:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with Timer() as gen_t:
        outputs: list[dict] = []
        for row in shard:
            prompt = row["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            outputs.append({"id": row["id"], "prompt": prompt, "teacher_text": text})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")

    metrics = {
        "model_id": model_id,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model_load_time_sec": round(load_t.seconds, 4),
        "teacher_compute_time_sec": round(gen_t.seconds, 4),
        "num_examples": len(outputs),
    }
    write_json(args.metrics, metrics)
    log(f"Teacher shard end: shard={args.shard_index} examples={len(outputs)}")


if __name__ == "__main__":
    main()
