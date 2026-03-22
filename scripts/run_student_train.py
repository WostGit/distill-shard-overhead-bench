from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import log_event
from metrics_utils import write_json
from model_config import student_model_id


def load_teacher_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-input", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    model_id = student_model_id()
    log_event(f"student train start: model={model_id}")
    rows = load_teacher_rows(args.teacher_input)

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    losses: list[float] = []
    steps = min(args.max_steps, max(1, len(rows)))
    for i in range(steps):
        row = rows[i % len(rows)]
        text = f"{row['prompt']}\n{row['teacher_text']}"
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item()))

    train_time_sec = time.perf_counter() - start
    write_json(
        args.metrics_output,
        {
            "model_id": model_id,
            "train_steps": steps,
            "train_loss_mean": sum(losses) / len(losses),
            "train_time_sec": train_time_sec,
        },
    )
    log_event(f"student train end: steps={steps} wall={train_time_sec:.3f}s")


if __name__ == "__main__":
    main()
