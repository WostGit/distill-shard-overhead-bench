from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import log
from metrics_utils import timed, write_json
from model_config import get_model_id, validate_model_id


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def shard_rows(rows, shard_index: int, shard_count: int):
    n = len(rows)
    chunk = math.ceil(n / shard_count)
    start = shard_index * chunk
    end = min(n, start + chunk)
    return rows[start:end]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/tiny_teacher_prompts.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    model_id = get_model_id()
    validate_model_id(model_id)
    log(f"teacher shard start model_id={model_id} shard={args.shard_index}/{args.shard_count}")

    rows = list(load_jsonl(Path(args.input)))
    subset = shard_rows(rows, args.shard_index, args.shard_count)

    with timed() as model_load_t:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with timed() as compute_t:
        with output_path.open("w", encoding="utf-8") as f:
            for row in subset:
                answer = generate(model, tokenizer, row["prompt"])
                f.write(json.dumps({"id": row["id"], "prompt": row["prompt"], "teacher": answer}) + "\n")

    payload = {
        "model_id": model_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "examples": len(subset),
        "model_load_time_sec": model_load_t.seconds,
        "teacher_compute_time_sec": compute_t.seconds,
        "output_artifact_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }
    write_json(args.metrics, payload)
    log(f"teacher shard end shard={args.shard_index} examples={len(subset)}")


if __name__ == "__main__":
    main()
