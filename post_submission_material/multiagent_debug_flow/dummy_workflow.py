from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .results import append_result, display_log_path, resolve_results_log_path
from .state import InjectedDebugInterruption, JsonCheckpointStore, WorkflowState
from .tasks import DebugTask, get_task, task_metadata


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, *, input_tokens: int, output_tokens: int, requests: int = 1) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += requests
        self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class DebugMetrics:
    wall_time_s: float = 0.0
    diagnoses: int = 0
    file_reads: int = 0
    file_writes: int = 0
    test_runs: int = 0
    review_rounds: int = 0
    patches_attempted: int = 0
    repeated_diagnoses: int = 0
    duplicated_patch_writes: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["token_usage"] = asdict(self.token_usage)
        return data

def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _setup_workspace(task: DebugTask, workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    _write_file(workspace / "src" / "__init__.py", "")
    _write_file(workspace / task.module_path, task.buggy_source)
    _write_file(workspace / task.test_path, task.test_source)


def _run_tests(workspace: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout


def _estimate_read_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _read_task_files(task: DebugTask, workspace: Path, metrics: DebugMetrics) -> str:
    module_text = (workspace / task.module_path).read_text()
    test_text = (workspace / task.test_path).read_text()
    metrics.file_reads += 2
    metrics.token_usage.add(
        input_tokens=_estimate_read_tokens(module_text) + _estimate_read_tokens(test_text),
        output_tokens=20,
    )
    return module_text + "\n" + test_text


def _developer_diagnose(task: DebugTask, workspace: Path, state: WorkflowState, metrics: DebugMetrics) -> None:
    _read_task_files(task, workspace, metrics)
    metrics.diagnoses += 1
    if metrics.diagnoses > 1:
        metrics.repeated_diagnoses += 1
    state.diagnosis = (
        "The implementation handles only the happy path. It must preserve the normal "
        "case while adding edge-case guards required by the tests."
    )
    state.stage = "diagnosed"
    state.next_agent = "developer"
    state.next_action = "apply_patch_v1"
    metrics.token_usage.add(input_tokens=35, output_tokens=45)


def _developer_apply_patch(task: DebugTask, workspace: Path, state: WorkflowState, metrics: DebugMetrics) -> None:
    patch_source = task.patch_v1_source if state.patch_version == 0 else task.patch_v2_source
    next_version = state.patch_version + 1
    module_path = workspace / task.module_path
    if module_path.read_text() == patch_source:
        metrics.duplicated_patch_writes += 1
    _write_file(module_path, patch_source)
    metrics.file_writes += 1
    metrics.patches_attempted += 1
    state.patch_version = next_version
    state.modified_files = [task.module_path]
    state.stage = f"patched_v{next_version}"
    state.next_agent = "developer"
    state.next_action = "run_tests"
    metrics.token_usage.add(input_tokens=45, output_tokens=_estimate_read_tokens(patch_source))


def _developer_run_tests(workspace: Path, state: WorkflowState, metrics: DebugMetrics) -> None:
    passed, output = _run_tests(workspace)
    metrics.test_runs += 1
    state.last_test_passed = passed
    state.last_test_output = output[-2000:]
    state.stage = f"tested_v{state.patch_version}"
    state.next_agent = "reviewer"
    state.next_action = "review_patch"
    metrics.token_usage.add(input_tokens=80, output_tokens=20)


def _reviewer_review(task: DebugTask, state: WorkflowState, metrics: DebugMetrics) -> None:
    metrics.review_rounds += 1
    if state.patch_version == 1:
        state.approved = False
        state.reviewer_feedback = list(task.reviewer_feedback)
        state.stage = "review_feedback_received"
        state.next_agent = "developer"
        state.next_action = "revise_patch"
        metrics.token_usage.add(input_tokens=120, output_tokens=55)
        return

    state.approved = bool(state.last_test_passed)
    state.reviewer_feedback = [] if state.approved else ["Tests still fail; inspect the remaining failure."]
    state.stage = "approved" if state.approved else "review_feedback_received"
    state.next_agent = "done" if state.approved else "developer"
    state.next_action = "finalize" if state.approved else "revise_patch"
    metrics.token_usage.add(input_tokens=100, output_tokens=25)


def _validate_resume_state(task: DebugTask, workspace: Path, state: WorkflowState) -> None:
    module_path = workspace / task.module_path
    test_path = workspace / task.test_path
    if not module_path.exists() or not test_path.exists():
        raise ValueError("Cannot resume: expected task files are missing.")
    if state.stage == "review_feedback_received" and not state.reviewer_feedback:
        raise ValueError("Cannot resume: reviewer feedback checkpoint is empty.")
    if state.patch_version == 1 and module_path.read_text() != task.patch_v1_source:
        raise ValueError("Cannot resume: workspace no longer matches patch version 1.")


def run_arm(
    *,
    task_id: str = "normalize_scores",
    behavior: str,
    workspace: Path,
    checkpoint_root: Path,
    run_id: str,
    inject_interruption: bool = True,
) -> dict[str, Any]:
    """Run one experiment arm.

    behavior must be "baseline_restart_from_zero" or "checkpoint_resume".
    The measured time and counts include the interrupted partial run plus the
    successful completion run, matching the A/B shape used by the XAS example.
    """

    if behavior not in {"baseline_restart_from_zero", "checkpoint_resume"}:
        raise ValueError(f"Unsupported behavior: {behavior}")

    task = get_task(task_id)
    checkpoint = JsonCheckpointStore(checkpoint_root, run_id)
    metrics = DebugMetrics()
    started = time.perf_counter()

    def run_until_done(*, resume: bool, allow_interrupt: bool) -> WorkflowState:
        state = checkpoint.load() if resume else None
        if state is None:
            _setup_workspace(task, workspace)
            metrics.file_writes += 2
            state = WorkflowState(task_id=task.task_id, bug_report=task.bug_report)
            checkpoint.save(state)
        else:
            _validate_resume_state(task, workspace, state)

        while not state.approved:
            if state.stage == "initialized":
                _developer_diagnose(task, workspace, state, metrics)
                checkpoint.save(state)
            elif state.stage == "diagnosed":
                _developer_apply_patch(task, workspace, state, metrics)
                checkpoint.save(state)
            elif state.stage.startswith("patched_v"):
                _developer_run_tests(workspace, state, metrics)
                checkpoint.save(state)
            elif state.stage.startswith("tested_v"):
                _reviewer_review(task, state, metrics)
                checkpoint.save(state)
                if allow_interrupt and state.stage == "review_feedback_received":
                    raise InjectedDebugInterruption("Interrupted after reviewer feedback before developer revision")
            elif state.stage == "review_feedback_received":
                _developer_apply_patch(task, workspace, state, metrics)
                checkpoint.save(state)
            else:
                raise RuntimeError(f"Unknown workflow stage: {state.stage}")

        return state

    try:
        run_until_done(resume=False, allow_interrupt=inject_interruption)
    except InjectedDebugInterruption:
        if behavior == "baseline_restart_from_zero":
            checkpoint.clear()
            run_until_done(resume=False, allow_interrupt=False)
        else:
            run_until_done(resume=True, allow_interrupt=False)

    metrics.wall_time_s = time.perf_counter() - started
    final_passed, final_output = _run_tests(workspace)
    metrics.test_runs += 1
    metrics.token_usage.add(input_tokens=20, output_tokens=10)

    final_state = checkpoint.load()
    return {
        "behavior": behavior,
        "task_id": task.task_id,
        "interruption": "after_reviewer_feedback" if inject_interruption else None,
        "final_stage": final_state.stage if final_state else "missing",
        "approved": bool(final_state and final_state.approved),
        "final_tests_passed": final_passed,
        "final_test_output": final_output[-2000:],
        "metrics": metrics.to_dict(),
    }


def run_ab_experiment(
    *,
    task_id: str = "normalize_scores",
    root: Path,
    run_id_prefix: str = "debug_ab",
) -> dict[str, Any]:
    baseline = run_arm(
        task_id=task_id,
        behavior="baseline_restart_from_zero",
        workspace=root / "baseline_workspace",
        checkpoint_root=root / "checkpoints",
        run_id=f"{run_id_prefix}_{task_id}_baseline",
    )
    checkpointed = run_arm(
        task_id=task_id,
        behavior="checkpoint_resume",
        workspace=root / "resume_workspace",
        checkpoint_root=root / "checkpoints",
        run_id=f"{run_id_prefix}_{task_id}_resume",
    )

    return {
        "schema_version": 1,
        "workflow": "multiagent_debug_flow",
        "task_id": task_id,
        "task_metadata": task_metadata(task_id),
        "interruption": "after_reviewer_feedback",
        "arms": {
            "A": baseline,
            "B": checkpointed,
        },
        "B_minus_A_wall_time_s": (
            checkpointed["metrics"]["wall_time_s"] - baseline["metrics"]["wall_time_s"]
        ),
        "B_minus_A_total_tokens": (
            checkpointed["metrics"]["token_usage"]["total_tokens"]
            - baseline["metrics"]["token_usage"]["total_tokens"]
        ),
        "B_minus_A_test_runs": (
            checkpointed["metrics"]["test_runs"] - baseline["metrics"]["test_runs"]
        ),
        "B_minus_A_repeated_diagnoses": (
            checkpointed["metrics"]["repeated_diagnoses"] - baseline["metrics"]["repeated_diagnoses"]
        ),
        "B_minus_A_file_writes": (
            checkpointed["metrics"]["file_writes"] - baseline["metrics"]["file_writes"]
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the multi-agent debug flow A/B example.")
    parser.add_argument("--task-id", default="normalize_scores")
    parser.add_argument("--root", type=Path, default=Path(".debug_flow_runs"))
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    summary = run_ab_experiment(task_id=args.task_id, root=args.root)
    if not args.no_log:
        log_path = resolve_results_log_path(args.log_path)
        summary["result_log_path"] = display_log_path(log_path)
        append_result(summary, log_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
