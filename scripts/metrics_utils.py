from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    marker = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield marker
    finally:
        marker["elapsed"] = time.perf_counter() - marker["start"]


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
