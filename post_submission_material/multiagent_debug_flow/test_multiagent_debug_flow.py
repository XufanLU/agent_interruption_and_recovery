from __future__ import annotations

from pathlib import Path

import pytest

from multiagent_debug_flow.dummy_workflow import run_ab_experiment, run_arm


@pytest.mark.parametrize("task_id", ["normalize_scores", "parse_duration"])
def test_multiagent_debug_flow_ab_resume_beats_restart(task_id: str, tmp_path: Path):
    summary = run_ab_experiment(task_id=task_id, root=tmp_path)

    baseline = summary["arms"]["A"]
    checkpointed = summary["arms"]["B"]

    assert baseline["final_tests_passed"]
    assert checkpointed["final_tests_passed"]
    assert baseline["approved"]
    assert checkpointed["approved"]

    assert checkpointed["metrics"]["diagnoses"] < baseline["metrics"]["diagnoses"]
    assert checkpointed["metrics"]["patches_attempted"] < baseline["metrics"]["patches_attempted"]
    assert checkpointed["metrics"]["review_rounds"] < baseline["metrics"]["review_rounds"]
    assert checkpointed["metrics"]["file_writes"] < baseline["metrics"]["file_writes"]
    assert checkpointed["metrics"]["token_usage"]["total_tokens"] < baseline["metrics"]["token_usage"]["total_tokens"]

    assert summary["B_minus_A_total_tokens"] < 0
    assert summary["B_minus_A_file_writes"] < 0


def test_resume_preserves_reviewer_feedback_handoff(tmp_path: Path):
    result = run_arm(
        task_id="normalize_scores",
        behavior="checkpoint_resume",
        workspace=tmp_path / "workspace",
        checkpoint_root=tmp_path / "checkpoints",
        run_id="resume_feedback",
    )

    assert result["final_stage"] == "approved"
    assert result["final_tests_passed"]
    assert result["metrics"]["repeated_diagnoses"] == 0
    assert result["metrics"]["duplicated_patch_writes"] == 0
