from __future__ import annotations

from datetime import datetime, timezone


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(message: str) -> None:
    print(f"[{utc_ts()}] {message}", flush=True)
