from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .tasks import DebugTask


def create_debug_agents(task: DebugTask, workspace: Path):
    """Create OpenAI Agents SDK developer/reviewer agents for one debug workspace."""

    from agents import Agent, function_tool

    model_name = os.getenv("DEBUG_FLOW_MODEL") or os.getenv("AB_TEST_MODEL") or os.getenv("OPENAI_MODEL")

    def _resolve_task_path(relative_path: str) -> Path:
        allowed = {task.module_path, task.test_path}
        if relative_path not in allowed:
            raise ValueError(f"Allowed paths are: {sorted(allowed)}")
        return workspace / relative_path

    @function_tool
    def read_debug_file(relative_path: str) -> str:
        """Read one allowed task file: the source module or test file."""
        return _resolve_task_path(relative_path).read_text()

    @function_tool
    def write_patch_version(version: int) -> str:
        """Write the task's patch version 1 or 2 to the source module."""
        if version == 1:
            source = task.patch_v1_source
        elif version == 2:
            source = task.patch_v2_source
        else:
            raise ValueError("version must be 1 or 2")
        module_path = workspace / task.module_path
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(source)
        return f"wrote patch version {version} to {task.module_path}"

    @function_tool
    def run_debug_tests() -> str:
        """Run pytest for the generated debugging workspace and return output."""
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        return f"status={status}\n{completed.stdout[-3000:]}"

    @function_tool
    def review_current_patch(patch_version: int, tests_passed: bool) -> str:
        """Review the current patch and return approval or concrete feedback."""
        if patch_version == 1:
            return "NEEDS_REVISION\n" + "\n".join(f"- {item}" for item in task.reviewer_feedback)
        if tests_passed:
            return "APPROVED\nPatch version 2 addresses reviewer feedback and tests pass."
        return "NEEDS_REVISION\nTests still fail; developer must inspect the remaining failure."

    developer_instructions = f"""
You are the Developer Agent in a small software debugging workflow.

Task:
{task.bug_report}

Allowed source file: {task.module_path}
Allowed test file: {task.test_path}

Follow the user request exactly. Use tools rather than inventing file contents:
- For first patch requests, read both files, diagnose briefly, call write_patch_version with version=1, then call run_debug_tests.
- For revision requests, use the provided reviewer feedback, call write_patch_version with version=2, then call run_debug_tests.
- Keep your final response concise and include whether tests passed.
"""

    reviewer_instructions = f"""
You are the Reviewer Agent in a developer/reviewer debugging workflow.

Review patch attempts for this task:
{task.bug_report}

Use review_current_patch. For patch version 1, request revision using the known review findings.
For patch version 2, approve only if tests_passed is true. Keep the final response concise.
"""

    agent_kwargs = {}
    if model_name:
        agent_kwargs["model"] = model_name

    developer = Agent(
        name="Debug Developer",
        instructions=developer_instructions,
        tools=[read_debug_file, write_patch_version, run_debug_tests],
        **agent_kwargs,
    )
    reviewer = Agent(
        name="Debug Reviewer",
        instructions=reviewer_instructions,
        tools=[review_current_patch],
        **agent_kwargs,
    )
    return developer, reviewer
