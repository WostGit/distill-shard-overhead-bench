from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import log_event
from metrics_utils import write_json
from model_config import student_model_id


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tiny_eval.jsonl")
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    model_id = student_model_id()
    log_event(f"eval shard start: shard={args.shard_id}/{args.num_shards} model={model_id}")
    all_rows = load_jsonl(args.input)
    rows = [r for i, r in enumerate(all_rows) if i % args.num_shards == args.shard_id]

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    nlls: list[float] = []
    with torch.no_grad():
        for row in rows:
            batch = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=256)
            out = model(**batch, labels=batch["input_ids"])
            nlls.append(float(out.loss.detach().cpu().item()))

    eval_time_sec = time.perf_counter() - start
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    write_json(
        args.metrics_output,
        {
            "model_id": model_id,
            "eval_rows": len(rows),
            "eval_loss_mean": sum(nlls) / len(nlls) if nlls else float("nan"),
            "eval_perplexity": ppl,
            "eval_time_sec": eval_time_sec,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        },
    )
    log_event(f"eval shard end: shard={args.shard_id} rows={len(rows)} wall={eval_time_sec:.3f}s")


if __name__ == "__main__":
    main()
