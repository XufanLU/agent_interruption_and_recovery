from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEBUG_FLOW_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_LOG_PATH = DEBUG_FLOW_DIR / "debug_ab_results.jsonl"


def resolve_results_log_path(path: str | Path | None = None) -> Path:
    configured = path or os.getenv("DEBUG_FLOW_AB_LOG_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_RESULTS_LOG_PATH


def append_result(summary: dict[str, Any], path: str | Path | None = None) -> Path:
    destination = resolve_results_log_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(summary, sort_keys=True) + "\n")
    return destination


def display_log_path(path: str | Path) -> str:
    path_obj = Path(path).expanduser()
    try:
        return str(path_obj.resolve().relative_to(DEBUG_FLOW_DIR))
    except ValueError:
        return str(path_obj)
