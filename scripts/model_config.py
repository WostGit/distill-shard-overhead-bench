from __future__ import annotations

import os

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SECOND_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OPTIONAL_STRESS_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def get_model_id() -> str:
    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    return model_id


def validate_model_id(model_id: str) -> None:
    allowed = {DEFAULT_MODEL_ID, SECOND_MODEL_ID, OPTIONAL_STRESS_MODEL_ID}
    if model_id not in allowed:
        raise ValueError(
            f"Unsupported MODEL_ID='{model_id}'. Allowed models: {sorted(allowed)}"
        )
