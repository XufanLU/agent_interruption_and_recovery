from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .openai_agents import create_debug_agents
from .process_logging import ProcessLogger, display_process_log_path, new_experiment_id, resolve_process_log_path
from .results import append_result, display_log_path, resolve_results_log_path
from .state import InjectedDebugInterruption, JsonCheckpointStore, WorkflowState
from .tasks import DebugTask, get_task, task_metadata
from .usage_tracking import add_token_usage, empty_token_usage, extract_model_token_usage


@dataclass
class OpenAIDebugMetrics:
    wall_time_s: float = 0.0
    token_usage: dict[str, int] = field(default_factory=empty_token_usage)
    agent_runs: int = 0
    diagnoses: int = 0
    file_writes: int = 0
    test_runs: int = 0
    review_rounds: int = 0
    patches_attempted: int = 0
    repeated_diagnoses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _tests_passed_from_output(output: str) -> bool:
    lowered = output.lower()
    return "status=passed" in lowered or " passed" in lowered and " failed" not in lowered


def _run_workspace_tests(workspace: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout


async def _run_agent(
    agent: Any,
    prompt: str,
    metrics: OpenAIDebugMetrics,
    *,
    process_logger: ProcessLogger | None,
    arm: str,
    stage: str,
    task_id: str,
):
    from agents import Runner

    agent_name = str(getattr(agent, "name", agent.__class__.__name__))
    if process_logger is not None:
        process_logger.log(
            "agent_input",
            arm=arm,
            stage=stage,
            task_id=task_id,
            agent=agent_name,
            prompt=prompt,
        )

    result = await Runner.run(agent, prompt)
    metrics.agent_runs += 1
    token_usage = extract_model_token_usage(result)
    add_token_usage(metrics.token_usage, token_usage)

    if process_logger is not None:
        process_logger.log(
            "agent_output",
            arm=arm,
            stage=stage,
            task_id=task_id,
            agent=agent_name,
            output=_final_output(result),
            token_usage=token_usage,
        )
    return result


def _final_output(result: Any) -> str:
    value = getattr(result, "final_output", None)
    if value is None:
        value = getattr(result, "output", None)
    return "" if value is None else str(value)


async def _run_until_done(
    *,
    task: DebugTask,
    workspace: Path,
    checkpoint: JsonCheckpointStore,
    metrics: OpenAIDebugMetrics,
    resume: bool,
    allow_interrupt: bool,
    process_logger: ProcessLogger | None,
    arm: str,
) -> WorkflowState:
    state = checkpoint.load() if resume else None
    if state is None:
        _setup_workspace(task, workspace)
        state = WorkflowState(task_id=task.task_id, bug_report=task.bug_report)
        checkpoint.save(state)
    elif state.stage == "review_feedback_received" and not state.reviewer_feedback:
        raise ValueError("Cannot resume: reviewer feedback checkpoint is empty.")

    developer, reviewer = create_debug_agents(task, workspace)

    while not state.approved:
        if state.stage == "initialized":
            prompt = (
                "Start the debugging task. Read the source and tests, diagnose the bug, "
                "write patch version 1 using the tool, and run the tests."
            )
            result = await _run_agent(
                developer,
                prompt,
                metrics,
                process_logger=process_logger,
                arm=arm,
                stage=state.stage,
                task_id=task.task_id,
            )
            tests_passed, test_output = _run_workspace_tests(workspace)
            if process_logger is not None:
                process_logger.log(
                    "workspace_test_result",
                    arm=arm,
                    stage="tested_v1",
                    task_id=task.task_id,
                    tests_passed=tests_passed,
                    output=test_output[-3000:],
                )
            state.diagnosis = _final_output(result)[-1000:]
            state.patch_version = 1
            state.modified_files = [task.module_path]
            state.last_test_passed = tests_passed
            state.last_test_output = test_output[-2000:]
            state.stage = "tested_v1"
            state.next_agent = "reviewer"
            state.next_action = "review_patch"
            metrics.diagnoses += 1
            metrics.file_writes += 1
            metrics.patches_attempted += 1
            metrics.test_runs += 1
            checkpoint.save(state)
        elif state.stage == "tested_v1":
            prompt = "Review patch_version=1. Tests did not fully satisfy the review contract. Return feedback."
            result = await _run_agent(
                reviewer,
                prompt,
                metrics,
                process_logger=process_logger,
                arm=arm,
                stage=state.stage,
                task_id=task.task_id,
            )
            feedback = _final_output(result)
            state.reviewer_feedback = [line.strip("- ") for line in feedback.splitlines() if line.strip().startswith("-")]
            if not state.reviewer_feedback:
                state.reviewer_feedback = list(task.reviewer_feedback)
            state.stage = "review_feedback_received"
            state.next_agent = "developer"
            state.next_action = "revise_patch"
            metrics.review_rounds += 1
            checkpoint.save(state)
            if allow_interrupt:
                raise InjectedDebugInterruption("Interrupted after reviewer feedback before developer revision")
        elif state.stage == "review_feedback_received":
            prompt = (
                "Resume from reviewer feedback. Do not redo the first diagnosis. "
                f"Reviewer feedback: {state.reviewer_feedback}. "
                "Write patch version 2 using the tool and run the tests."
            )
            result = await _run_agent(
                developer,
                prompt,
                metrics,
                process_logger=process_logger,
                arm=arm,
                stage=state.stage,
                task_id=task.task_id,
            )
            tests_passed, test_output = _run_workspace_tests(workspace)
            if process_logger is not None:
                process_logger.log(
                    "workspace_test_result",
                    arm=arm,
                    stage="tested_v2",
                    task_id=task.task_id,
                    tests_passed=tests_passed,
                    output=test_output[-3000:],
                )
            output = _final_output(result)
            state.patch_version = 2
            state.last_test_output = (test_output or output)[-2000:]
            state.last_test_passed = tests_passed
            state.stage = "tested_v2"
            state.next_agent = "reviewer"
            state.next_action = "review_patch"
            metrics.file_writes += 1
            metrics.patches_attempted += 1
            metrics.test_runs += 1
            checkpoint.save(state)
        elif state.stage == "tested_v2":
            prompt = f"Review patch_version=2 with tests_passed={bool(state.last_test_passed)}."
            result = await _run_agent(
                reviewer,
                prompt,
                metrics,
                process_logger=process_logger,
                arm=arm,
                stage=state.stage,
                task_id=task.task_id,
            )
            review = _final_output(result)
            state.approved = "APPROVED" in review.upper() or bool(state.last_test_passed)
            state.reviewer_feedback = [] if state.approved else [review]
            state.stage = "approved" if state.approved else "review_feedback_received"
            state.next_agent = "done" if state.approved else "developer"
            state.next_action = "finalize" if state.approved else "revise_patch"
            metrics.review_rounds += 1
            checkpoint.save(state)
        else:
            raise RuntimeError(f"Unknown workflow stage: {state.stage}")

    return state


async def run_openai_arm_async(
    *,
    task_id: str = "normalize_scores",
    behavior: str,
    workspace: Path,
    checkpoint_root: Path,
    run_id: str,
    inject_interruption: bool = True,
    process_log_path: str | Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    if behavior not in {"baseline_restart_from_zero", "checkpoint_resume"}:
        raise ValueError(f"Unsupported behavior: {behavior}")

    task = get_task(task_id)
    checkpoint = JsonCheckpointStore(checkpoint_root, run_id)
    metrics = OpenAIDebugMetrics()
    process_logger = (
        ProcessLogger(process_log_path, experiment_id=experiment_id) if process_log_path is not None else None
    )
    started = time.perf_counter()
    arm_label = "A" if behavior == "baseline_restart_from_zero" else "B"

    try:
        await _run_until_done(
            task=task,
            workspace=workspace,
            checkpoint=checkpoint,
            metrics=metrics,
            resume=False,
            allow_interrupt=inject_interruption,
            process_logger=process_logger,
            arm=arm_label,
        )
    except InjectedDebugInterruption:
        if process_logger is not None:
            process_logger.log(
                "interruption",
                arm=arm_label,
                task_id=task.task_id,
                behavior=behavior,
                interruption="after_reviewer_feedback",
            )
        if behavior == "baseline_restart_from_zero":
            metrics.repeated_diagnoses += 1
            checkpoint.clear()
            if process_logger is not None:
                process_logger.log(
                    "recovery_decision",
                    arm=arm_label,
                    task_id=task.task_id,
                    behavior=behavior,
                    decision="discard_checkpoint_and_restart",
                )
            await _run_until_done(
                task=task,
                workspace=workspace,
                checkpoint=checkpoint,
                metrics=metrics,
                resume=False,
                allow_interrupt=False,
                process_logger=process_logger,
                arm=arm_label,
            )
        else:
            if process_logger is not None:
                process_logger.log(
                    "recovery_decision",
                    arm=arm_label,
                    task_id=task.task_id,
                    behavior=behavior,
                    decision="load_checkpoint_and_resume",
                )
            await _run_until_done(
                task=task,
                workspace=workspace,
                checkpoint=checkpoint,
                metrics=metrics,
                resume=True,
                allow_interrupt=False,
                process_logger=process_logger,
                arm=arm_label,
            )

    metrics.wall_time_s = time.perf_counter() - started
    final_state = checkpoint.load()
    return {
        "behavior": behavior,
        "task_id": task_id,
        "interruption": "after_reviewer_feedback" if inject_interruption else None,
        "final_stage": final_state.stage if final_state else "missing",
        "approved": bool(final_state and final_state.approved),
        "metrics": metrics.to_dict(),
    }


def run_openai_arm(**kwargs) -> dict[str, Any]:
    return asyncio.run(run_openai_arm_async(**kwargs))


async def run_ab_experiment_async(
    *,
    task_id: str = "normalize_scores",
    root: Path,
    run_id_prefix: str = "openai_debug_ab",
    process_log_path: str | Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    experiment_id = experiment_id or new_experiment_id(f"openai_debug_flow_{task_id}")
    baseline = await run_openai_arm_async(
        task_id=task_id,
        behavior="baseline_restart_from_zero",
        workspace=root / "baseline_workspace",
        checkpoint_root=root / "checkpoints",
        run_id=f"{run_id_prefix}_{task_id}_baseline",
        process_log_path=process_log_path,
        experiment_id=experiment_id,
    )
    checkpointed = await run_openai_arm_async(
        task_id=task_id,
        behavior="checkpoint_resume",
        workspace=root / "resume_workspace",
        checkpoint_root=root / "checkpoints",
        run_id=f"{run_id_prefix}_{task_id}_resume",
        process_log_path=process_log_path,
        experiment_id=experiment_id,
    )
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "workflow": "multiagent_debug_flow_openai_agents",
        "task_id": task_id,
        "task_metadata": task_metadata(task_id),
        "interruption": "after_reviewer_feedback",
        "arms": {"A": baseline, "B": checkpointed},
        "B_minus_A_wall_time_s": checkpointed["metrics"]["wall_time_s"] - baseline["metrics"]["wall_time_s"],
        "B_minus_A_total_tokens": (
            checkpointed["metrics"]["token_usage"]["total_tokens"]
            - baseline["metrics"]["token_usage"]["total_tokens"]
        ),
        "B_minus_A_test_runs": checkpointed["metrics"]["test_runs"] - baseline["metrics"]["test_runs"],
        "B_minus_A_agent_runs": checkpointed["metrics"]["agent_runs"] - baseline["metrics"]["agent_runs"],
    }
    if process_log_path is not None:
        summary["process_log_path"] = display_process_log_path(process_log_path)
    return summary


def run_ab_experiment(**kwargs) -> dict[str, Any]:
    return asyncio.run(run_ab_experiment_async(**kwargs))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the OpenAI Agents SDK debug flow A/B example.")
    parser.add_argument("--task-id", default="normalize_scores")
    parser.add_argument("--root", type=Path, default=Path(".debug_flow_runs/openai"))
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--process-log-path", type=Path, default=None)
    parser.add_argument("--no-process-log", action="store_true")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    process_log_path = None if args.no_process_log else resolve_process_log_path(args.process_log_path)
    summary = run_ab_experiment(
        task_id=args.task_id,
        root=args.root,
        process_log_path=process_log_path,
        experiment_id=args.experiment_id,
    )
    if not args.no_log:
        log_path = resolve_results_log_path(args.log_path)
        summary["result_log_path"] = display_log_path(log_path)
        append_result(summary, log_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
