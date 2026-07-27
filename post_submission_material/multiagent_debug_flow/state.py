from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class InjectedDebugInterruption(RuntimeError):
    """Raised when the experiment injects an interruption at a handoff point."""


@dataclass
class WorkflowState:
    task_id: str
    stage: str = "initialized"
    bug_report: str = ""
    diagnosis: str | None = None
    patch_version: int = 0
    modified_files: list[str] = field(default_factory=list)
    test_command: str = f"{sys.executable} -m pytest -q"
    last_test_passed: bool | None = None
    last_test_output: str = ""
    reviewer_feedback: list[str] = field(default_factory=list)
    approved: bool = False
    next_agent: str = "developer"
    next_action: str = "diagnose"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        return cls(**data)


class JsonCheckpointStore:
    def __init__(self, root: Path, run_id: str):
        self.root = root
        self.run_id = run_id
        self.path = self.root / f"{run_id}.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: WorkflowState) -> None:
        self.path.write_text(json.dumps(state.to_dict(), indent=2))

    def load(self) -> WorkflowState | None:
        if not self.path.exists():
            return None
        return WorkflowState.from_dict(json.loads(self.path.read_text()))

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
