from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class TimedBlock:
    name: str
    started_at: float
    ended_at: float

    @property
    def elapsed_sec(self) -> float:
        return self.ended_at - self.started_at


@contextmanager
def timed_block(name: str) -> Iterator[TimedBlock]:
    block = TimedBlock(name=name, started_at=time.perf_counter(), ended_at=0.0)
    yield block
    block.ended_at = time.perf_counter()


def write_json(path: str | Path, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bytes_on_disk(paths: list[str | Path]) -> int:
    total = 0
    for p in paths:
        path = Path(p)
        if path.exists():
            if path.is_file():
                total += path.stat().st_size
            else:
                total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total
