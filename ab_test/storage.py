from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from data_paths import activity_history_dir, backend_root, matching_dir, processed_xas_dir, viz_dir


def _normalize_key(key: str) -> str:
    candidate = str(key or "").strip().replace("\\", "/").lstrip("/")
    parts = [part for part in PurePosixPath(candidate).parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid storage key: {key}")
    return "/".join(parts)


def storage_path_for_key(key: str, *, root_dir: str | Path | None = None) -> Path:
    normalized = _normalize_key(key)
    root = backend_root(root_dir)
    mapping = {
        "processed_xas": processed_xas_dir(root),
        "viz": viz_dir(root),
        "matching": matching_dir(root),
        "logs": activity_history_dir(root),
    }
    prefix, _, remainder = normalized.partition("/")
    base_dir = mapping.get(prefix)
    return (base_dir / remainder) if base_dir and remainder else root / normalized


def upload_file(file_path: str, bucket: str = "local-storage", object_name: str | None = None) -> bool:
    if object_name is None:
        return Path(file_path).exists()
    destination = storage_path_for_key(object_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file_path, destination)
    return True


def upload_usage_log_data(session_id: str, log_data: dict[str, Any], bucket: str = "local-storage") -> bool:
    path = storage_path_for_key(f"logs/{session_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log_data, indent=2))
    return True


def download_usage_log(session_id: str, bucket: str = "local-storage") -> dict[str, Any] | None:
    path = storage_path_for_key(f"logs/{session_id}.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_usage_logs(bucket: str = "local-storage", prefix: str = "logs/") -> list[str]:
    logs_dir = activity_history_dir()
    if not logs_dir.exists():
        return []
    return [f"logs/{path.name}" for path in logs_dir.glob("*.json")]
