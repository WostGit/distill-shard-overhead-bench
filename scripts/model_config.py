from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoices:
    default_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    second_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    stress_model: str = "Qwen/Qwen2.5-3B-Instruct"


CHOICES = ModelChoices()


def resolve_model_id(requested_model: str | None) -> str:
    return requested_model or CHOICES.default_model
