from __future__ import annotations

import os

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SECOND_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OPTIONAL_STRESS_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

ALLOWED_MODELS = {
    DEFAULT_MODEL_ID,
    SECOND_MODEL_ID,
    OPTIONAL_STRESS_MODEL_ID,
}


def resolve_model_id() -> str:
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID).strip()
    if model_id not in ALLOWED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_id}'. Allowed models: {sorted(ALLOWED_MODELS)}"
        )
    return model_id
