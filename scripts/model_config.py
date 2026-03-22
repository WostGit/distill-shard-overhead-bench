from __future__ import annotations

import os

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SECOND_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OPTIONAL_STRESS_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def teacher_model_id() -> str:
    return os.getenv("TEACHER_MODEL_ID", DEFAULT_MODEL_ID)


def student_model_id() -> str:
    # Keep student small by default for CPU-only GitHub Actions.
    return os.getenv("STUDENT_MODEL_ID", DEFAULT_MODEL_ID)
