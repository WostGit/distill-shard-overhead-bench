from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_utils import log
from metrics_utils import Timer, file_size_bytes, write_json
from model_config import resolve_model_id
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-prompts", type=Path, default=Path("data/tiny_teacher_prompts.jsonl"))
    p.add_argument("--eval-prompts", type=Path, default=Path("data/tiny_eval.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--metrics-output", type=Path, default=Path("outputs/metrics/phase1_single_metrics.json"))
    p.add_argument("--max-new-tokens", type=int, default=48)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with Timer() as total_t:
        with Timer() as setup_t:
            teacher_rows = load_jsonl(args.teacher_prompts)
            eval_rows = load_jsonl(args.eval_prompts)
            teacher_out = args.output_dir / "teacher_outputs_single.jsonl"
            student_snapshot = args.output_dir / "student_snapshot.json"
            eval_out = args.output_dir / "eval_outputs_single.jsonl"

        log(f"Phase1 baseline start model_id={model_id}")
        with Timer() as load_t:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        with Timer() as teacher_t:
            with teacher_out.open("w", encoding="utf-8") as out:
                for row in teacher_rows:
                    inp = tokenizer(row["prompt"], return_tensors="pt")
                    with torch.no_grad():
                        gen = model.generate(
                            **inp,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    text = tokenizer.decode(gen[0], skip_special_tokens=True)
                    out.write(json.dumps({"id": row["id"], "prompt": row["prompt"], "teacher_text": text}) + "\n")

        with Timer() as train_t:
            top_words: dict[str, int] = {}
            with teacher_out.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        for token in json.loads(line)["teacher_text"].lower().split():
                            top_words[token] = top_words.get(token, 0) + 1
            with student_snapshot.open("w", encoding="utf-8") as out:
                json.dump(
                    {
                        "model_id": model_id,
                        "token_histogram_size": len(top_words),
                        "top_tokens": sorted(top_words.items(), key=lambda kv: kv[1], reverse=True)[:100],
                    },
                    out,
                    indent=2,
                )

        with Timer() as eval_t:
            with eval_out.open("w", encoding="utf-8") as out:
                for row in eval_rows:
                    inp = tokenizer(row["prompt"], return_tensors="pt")
                    with torch.no_grad():
                        gen = model.generate(
                            **inp,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    text = tokenizer.decode(gen[0], skip_special_tokens=True)
                    out.write(json.dumps({"id": row["id"], "prediction": text, "reference": row["reference"]}) + "\n")

    artifact_bytes = file_size_bytes(teacher_out) + file_size_bytes(student_snapshot) + file_size_bytes(eval_out)
    metrics = {
        "model_id": model_id,
        "setup_time_sec": round(setup_t.seconds, 4),
        "model_load_time_sec": round(load_t.seconds, 4),
        "teacher_compute_time_sec": round(teacher_t.seconds, 4),
        "train_time_sec": round(train_t.seconds, 4),
        "eval_time_sec": round(eval_t.seconds, 4),
        "total_wall_time_sec": round(total_t.seconds, 4),
        "output_artifact_bytes": artifact_bytes,
    }
    write_json(args.metrics_output, metrics)
    log(f"Phase1 baseline done metrics={args.metrics_output} bytes={artifact_bytes}")


if __name__ == "__main__":
    main()
