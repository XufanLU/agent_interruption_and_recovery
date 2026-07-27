from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEBUG_FLOW_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESS_LOG_PATH = DEBUG_FLOW_DIR / "test_logs" / "agent_process_logs.jsonl"


def resolve_process_log_path(path: str | Path | None = None) -> Path:
    configured = path or os.getenv("DEBUG_FLOW_PROCESS_LOG_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_PROCESS_LOG_PATH


def display_process_log_path(path: str | Path) -> str:
    path_obj = Path(path).expanduser()
    try:
        return str(path_obj.resolve().relative_to(DEBUG_FLOW_DIR))
    except ValueError:
        return str(path_obj)


class ProcessLogger:
    def __init__(self, path: str | Path | None = None, *, experiment_id: str | None = None):
        self.path = resolve_process_log_path(path)
        self.experiment_id = experiment_id or new_experiment_id()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "experiment_id": self.experiment_id,
            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def new_experiment_id(prefix: str = "debug_flow") -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"
