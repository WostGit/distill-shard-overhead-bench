from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator


@dataclass
class Timer:
    start: float = 0.0
    end: float = 0.0

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


@contextmanager
def timed() -> Iterator[Timer]:
    t = Timer(start=time.perf_counter())
    try:
        yield t
    finally:
        t.end = time.perf_counter()


def write_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def jsonl_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return os.path.getsize(path)
