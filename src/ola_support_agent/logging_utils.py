from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .constants import LOG_DIR


class JsonlLogger:
    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or LOG_DIR / "requests.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    def log(self, **payload: object) -> None:
        enriched = {
            "timestamp_ms": int(time.time() * 1000),
            **payload,
        }

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched) + "\n")
