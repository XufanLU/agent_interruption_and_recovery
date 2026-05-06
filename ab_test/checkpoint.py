from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from data_paths import checkpoints_dir


_checkpoint_ctx: ContextVar[dict[str, Any] | None] = ContextVar("checkpoint_ctx", default=None)


async def check_if_paused() -> None:
    return None


async def save_checkpoint_from_tool(checkpoint_name: str) -> None:
    ctx = _checkpoint_ctx.get()
    storage = ctx.get("storage") if ctx else None
    if storage is not None and hasattr(storage, "save_lightweight"):
        storage.save_lightweight(checkpoint_name)


class FileCheckpointStorage:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.root = checkpoints_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.root / "meta.json"

    def _load_meta(self) -> list[dict[str, Any]]:
        if not self.meta_file.exists():
            return []
        try:
            data = json.loads(self.meta_file.read_text())
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _save_meta(self, meta: list[dict[str, Any]]) -> None:
        self.meta_file.write_text(json.dumps(meta, indent=2))

    def save_lightweight(self, checkpoint_name: str) -> str:
        checkpoint_id = f"{self.session_id}_{checkpoint_name}"
        meta = self._load_meta()
        meta.append(
            {
                "checkpoint_id": checkpoint_id,
                "session_id": self.session_id,
                "name": checkpoint_name,
                "lightweight": True,
            }
        )
        self._save_meta(meta)
        return checkpoint_id


def cleanup_session_checkpoints(session_id: str) -> None:
    root = checkpoints_dir()
    if not root.exists():
        return
    for path in root.glob(f"{session_id}_*.json"):
        path.unlink(missing_ok=True)


def cleanup_fit_cache(xas_path: str | None = None) -> None:
    if not xas_path:
        return
    root = checkpoints_dir()
    if not root.exists():
        return
    for path in root.glob("fit_cache_*.json"):
        path.unlink(missing_ok=True)


def restore_session_from_checkpoint(session_id: str, storage_dir: Path, messages: list[Any]) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / f"{session_id}.json").write_text(json.dumps({"messages": messages}, indent=2))
