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
    p.add_argument("--eval-input", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
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
    rows = load_jsonl(args.eval_input)
    shard = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]

    log(f"Eval shard start: shard={args.shard_index}/{args.num_shards} model={model_id}")
    with Timer() as load_t:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    preds: list[dict] = []
    with Timer() as eval_t:
        for row in shard:
            inputs = tokenizer(row["prompt"], return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            is_correct = any(word in text.lower() for word in row["reference"].lower().split()[:4])
            correct += int(is_correct)
            preds.append({"id": row["id"], "prediction": text, "reference": row["reference"], "correct": is_correct})

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    write_json(
        args.metrics,
        {
            "model_id": model_id,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "model_load_time_sec": round(load_t.seconds, 4),
            "eval_time_sec": round(eval_t.seconds, 4),
            "eval_examples": len(preds),
            "eval_accuracy": (correct / len(preds)) if preds else 0.0,
        },
    )
    log(f"Eval shard end: shard={args.shard_index} examples={len(preds)}")


if __name__ == "__main__":
    main()
