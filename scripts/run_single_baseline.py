from __future__ import annotations

import argparse
import json
import time
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def token_overlap(a: str, b: str) -> float:
    aset = set(a.lower().split())
    bset = set(b.lower().split())
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(1, len(bset))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-input", default="data/tiny_teacher_prompts.jsonl")
    p.add_argument("--eval-input", default="data/tiny_eval.jsonl")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--metrics", default="outputs/metrics/phase1_single_baseline_metrics.json")
    args = p.parse_args()

    total_start = time.perf_counter()
    setup_start = time.perf_counter()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = get_model_id()
    validate_model_id(model_id)
    log(f"phase1 baseline start model_id={model_id}")

    with timed() as setup_t:
        teacher_rows = list(load_jsonl(Path(args.teacher_input)))
        eval_rows = list(load_jsonl(Path(args.eval_input)))
        _ = setup_start

    with timed() as model_load_t:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()

    teacher_out = out_dir / "teacher_outputs.jsonl"
    with timed() as teacher_t:
        with teacher_out.open("w", encoding="utf-8") as f:
            for row in teacher_rows:
                ans = generate(model, tokenizer, row["prompt"])
                f.write(json.dumps({"id": row["id"], "prompt": row["prompt"], "teacher": ans}) + "\n")

    student_out = out_dir / "student_artifact.json"
    with timed() as train_t:
        entries = [json.loads(line) for line in teacher_out.read_text(encoding="utf-8").splitlines() if line.strip()]
        total_chars = sum(len(x.get("teacher", "")) for x in entries)
        student_out.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "num_examples": len(entries),
                    "avg_teacher_chars": (total_chars / len(entries)) if entries else 0.0,
                    "note": "Synthetic lightweight student artifact for overhead benchmark.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    eval_out = out_dir / "eval_outputs.jsonl"
    with timed() as eval_t:
        with eval_out.open("w", encoding="utf-8") as f:
            for row in eval_rows:
                pred = generate(model, tokenizer, row["prompt"])
                score = token_overlap(pred, row["reference"])
                f.write(json.dumps({"id": row["id"], "score": score, "prediction": pred}) + "\n")

    artifact_bytes = sum(
        pth.stat().st_size for pth in [teacher_out, student_out, eval_out] if pth.exists()
    )

    total_wall = time.perf_counter() - total_start
    metrics = {
        "model_id": model_id,
        "setup_time_sec": setup_t.seconds,
        "model_load_time_sec": model_load_t.seconds,
        "teacher_compute_time_sec": teacher_t.seconds,
        "train_time_sec": train_t.seconds,
        "eval_time_sec": eval_t.seconds,
        "total_wall_time_sec": total_wall,
        "output_artifact_bytes": artifact_bytes,
    }
    write_json(args.metrics, metrics)

    log(
        "phase1 bottleneck summary "
        f"teacher={teacher_t.seconds:.2f}s eval={eval_t.seconds:.2f}s load={model_load_t.seconds:.2f}s"
    )
    log("phase1 baseline end")


if __name__ == "__main__":
    main()
